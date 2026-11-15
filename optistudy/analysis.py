"""Gelişim analizi: net trendi, tutarlılık, net tahmini, backtest, hedef takibi,
sıralama tahmini ve konu önceliği."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.proportion import proportion_confint

from optistudy.scoring import WRONG_PENALTY, add_net_column

# Tüm derslerin toplam neti için kullanılan etiket
TOTAL_LABEL = "Toplam"

MIN_EXAMS_FOR_TREND = 3
CONSISTENCY_WINDOW = 5
MIN_EXAMS_FOR_CONSISTENCY = 2
TREND_CONFIDENCE = 0.95

VERDICT_INCREASE = "increase"
VERDICT_DECREASE = "decrease"
VERDICT_NO_CLEAR_CHANGE = "no_clear_change"

VERDICT_MESSAGES: dict[str, str] = {
    VERDICT_INCREASE: "Netlerin anlamlı şekilde artıyor.",
    VERDICT_DECREASE: "Netlerin anlamlı şekilde düşüyor.",
    VERDICT_NO_CLEAR_CHANGE: "Henüz net bir artış yok.",
}


@dataclass(frozen=True)
class TrendResult:
    """Net ~ zaman regresyonunun özeti. Eğim birimi: net/hafta."""

    n_exams: int
    start_date: date
    intercept: float  # ilk deneme tarihindeki tahmini net
    slope_per_week: float
    ci_low: float
    ci_high: float

    @property
    def verdict(self) -> str:
        """Güven aralığı sıfırı içeriyorsa net bir değişim yok sayılır."""
        if self.ci_low > 0:
            return VERDICT_INCREASE
        if self.ci_high < 0:
            return VERDICT_DECREASE
        return VERDICT_NO_CLEAR_CHANGE

    @property
    def message(self) -> str:
        return VERDICT_MESSAGES[self.verdict]

    def fitted(self, dates: pd.Series) -> pd.Series:
        """Verilen tarihlerde regresyon doğrusunun değeri."""
        return self.intercept + self.slope_per_week * _weeks_since(dates, self.start_date)


def _weeks_since(dates: pd.Series, start: date) -> pd.Series:
    return (pd.to_datetime(dates) - pd.Timestamp(start)).dt.days / 7.0


def net_series(results: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Bir dersin (veya `TOTAL_LABEL` ile toplamın) deneme başına netini döndürür.

    Girdi: `db.subject_results_frame` biçiminde uzun tablo.
    Çıktı: `exam_id`, `date`, `publisher`, `net`; tarihe göre sıralı.
    """
    scored = add_net_column(results)
    if subject != TOTAL_LABEL:
        scored = scored[scored["subject"] == subject]
    series = (
        scored.groupby(["exam_id", "date", "publisher"], as_index=False)["net"].sum()
        .sort_values(["date", "exam_id"])
        .reset_index(drop=True)
    )
    return series


def _fit_ols(series: pd.DataFrame) -> tuple[RegressionResultsWrapper, date] | None:
    """`net ~ hafta` OLS modelini kurar. 2'den az farklı tarih varsa `None`."""
    dates = pd.to_datetime(series["date"])
    if dates.nunique() < 2:
        return None
    start = dates.min().date()
    x = _weeks_since(dates, start).to_numpy(dtype=float)
    y = series["net"].to_numpy(dtype=float)
    return sm.OLS(y, sm.add_constant(x)).fit(), start


def fit_trend(series: pd.DataFrame) -> TrendResult | None:
    """`net ~ hafta` basit doğrusal regresyonu (OLS) kurar.

    Varsayımlar: net zamanla doğrusal değişir; hatalar bağımsız, sabit varyanslı ve
    yaklaşık normaldir; yayınevi zorluk farkı yok sayılır. Eğimin %95 güven aralığı
    t dağılımıyla hesaplanır.
    En az 3 deneme ve en az 2 farklı tarih yoksa `None` döner.
    """
    if len(series) < MIN_EXAMS_FOR_TREND:
        return None
    model = _fit_ols(series)
    if model is None:
        return None
    fit, start = model
    with np.errstate(divide="ignore", invalid="ignore"):  # tam uyumda (artık = 0) uyarıyı sustur
        low, high = fit.conf_int(alpha=1 - TREND_CONFIDENCE)[1]

    return TrendResult(
        n_exams=len(series),
        start_date=start,
        intercept=float(fit.params[0]),
        slope_per_week=float(fit.params[1]),
        ci_low=float(low),
        ci_high=float(high),
    )


def subject_trend(results: pd.DataFrame, subject: str) -> TrendResult | None:
    """Bir dersin (veya toplamın) net trendi. Varsayımlar için bkz. `fit_trend`."""
    return fit_trend(net_series(results, subject))


def consistency(results: pd.DataFrame, subject: str) -> float | None:
    """Son 5 denemedeki netlerin örneklem standart sapması (ddof=1).

    Küçük değer = daha tutarlı. Varsayım: son 5 deneme yakın dönemi temsil eder;
    yayınevi farkı yok sayılır. 2'den az deneme varsa `None` döner.
    """
    nets = net_series(results, subject)["net"].tail(CONSISTENCY_WINDOW)
    if len(nets) < MIN_EXAMS_FOR_CONSISTENCY:
        return None
    return float(nets.std(ddof=1))


SUMMARY_COLUMNS: list[str] = [
    "subject", "n_exams", "last_net", "mean_net",
    "slope_per_week", "ci_low", "ci_high", "verdict", "std_last5",
]


def progress_summary(results: pd.DataFrame) -> pd.DataFrame | None:
    """Her ders ve toplam için tek satırlık gelişim özeti.

    Sütunlar: `SUMMARY_COLUMNS`. Trend veya tutarlılık için veri yetersizse ilgili
    hücreler boş (NaN/None) kalır. Hiç deneme yoksa `None` döner.
    """
    if results.empty:
        return None
    rows = []
    for subject in [*pd.unique(results["subject"]), TOTAL_LABEL]:
        series = net_series(results, subject)
        trend = fit_trend(series)
        std = consistency(results, subject)
        rows.append({
            "subject": subject,
            "n_exams": len(series),
            "last_net": float(series["net"].iloc[-1]),
            "mean_net": float(series["net"].mean()),
            "slope_per_week": trend.slope_per_week if trend else np.nan,
            "ci_low": trend.ci_low if trend else np.nan,
            "ci_high": trend.ci_high if trend else np.nan,
            "verdict": trend.verdict if trend else None,
            "std_last5": std if std is not None else np.nan,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


# --- Net tahmini -----------------------------------------------------------

FORECAST_LEVEL = 0.80
MIN_EXAMS_FOR_FORECAST = 5
COVERAGE_CI_LEVEL = 0.95
# Sınav günü tahmini, ampirik kapsamanın Wilson güven aralığının alt sınırı bu değere
# ulaşınca açılır. Sabit bir deneme sayısı değildir: tahminler ne kadar sık tutarsa
# o kadar az backtest adımı yeter.
MIN_COVERAGE_LOWER_BOUND = 0.60


@dataclass(frozen=True)
class NetForecast:
    """Belirli bir tarih için net tahmini ve tahmin aralığı (prediction interval)."""

    target_date: date
    point: float
    low: float
    high: float
    n_exams: int
    level: float = FORECAST_LEVEL


def net_bounds(results: pd.DataFrame, subject: str) -> tuple[float, float]:
    """Bir dersin (veya toplamın) alabileceği en düşük ve en yüksek net.

    Hepsi yanlış: −soru/4; hepsi doğru: soru sayısı.
    """
    counts = results[["subject", "question_count"]].drop_duplicates()
    if subject != TOTAL_LABEL:
        counts = counts[counts["subject"] == subject]
    total = float(counts["question_count"].sum())
    return -total * WRONG_PENALTY, total


def predict_net(
    series: pd.DataFrame,
    target_date: date,
    bounds: tuple[float, float] | None = None,
    level: float = FORECAST_LEVEL,
) -> NetForecast | None:
    """`net ~ hafta` regresyonuyla `target_date` için netin tahmin aralığını verir.

    Tahmin aralığı (güven aralığı DEĞİL): tek bir yeni denemenin netinin düşeceği
    aralık, ŷ ± t(1−α/2; n−2) · s · √(1 + 1/n + (x₀−x̄)²/Sxx).
    Varsayımlar: doğrusal trend hedef tarihe kadar sürer; hatalar bağımsız, sabit
    varyanslı ve normaldir; yayınevi zorluk farkı yok sayılır. Hedef tarih uzaklaştıkça
    doğrusallık varsayımı zayıflar.
    `bounds` verilirse tahmin ve aralık bu sınırlara kırpılır (alınamayacak net gösterilmez).
    5 denemenin altında veya 2'den az farklı tarih varsa `None` döner.
    """
    if len(series) < MIN_EXAMS_FOR_FORECAST:
        return None
    last = pd.to_datetime(series["date"]).max().date()
    if target_date < last:
        raise ValueError("Tahmin tarihi son denemeden önce olamaz.")
    model = _fit_ols(series)
    if model is None:
        return None
    fit, start = model

    x0 = float(_weeks_since(pd.Series([target_date]), start).iloc[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        frame = fit.get_prediction(np.array([[1.0, x0]])).summary_frame(alpha=1 - level)
    point, low, high = (float(frame[c].iloc[0]) for c in ("mean", "obs_ci_lower", "obs_ci_upper"))
    if bounds is not None:
        point, low, high = (float(np.clip(v, *bounds)) for v in (point, low, high))
    return NetForecast(target_date, point, low, high, n_exams=len(series), level=level)


@dataclass(frozen=True)
class BacktestResult:
    """Rolling-origin backtest sonucu. `steps`: date, actual, point, low, high, hit."""

    steps: pd.DataFrame
    level: float = FORECAST_LEVEL

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def hits(self) -> int:
        return int(self.steps["hit"].sum())

    @property
    def coverage(self) -> float:
        return self.hits / self.n

    @property
    def coverage_ci(self) -> tuple[float, float]:
        """Ampirik kapsamanın Wilson güven aralığı (küçük örneklemde normal yaklaşımdan güvenilir)."""
        low, high = proportion_confint(self.hits, self.n, alpha=1 - COVERAGE_CI_LEVEL, method="wilson")
        return float(low), float(high)


def backtest(
    series: pd.DataFrame,
    bounds: tuple[float, float] | None = None,
    level: float = FORECAST_LEVEL,
) -> BacktestResult | None:
    """Rolling-origin backtest: ilk k denemeyle (k ≥ 5) k+1'inci denemeyi tahmin eder.

    Her adımda gerçek net tahmin aralığına düştü mü (`hit`) kaydedilir. Ampirik kapsama
    hedef seviyeye (%80) yakınsa aralıklar dürüst demektir.
    Varsayımlar `predict_net` ile aynı. Hiç adım yoksa (≤ 5 deneme) `None` döner.
    """
    series = series.sort_values("date").reset_index(drop=True)
    rows = []
    for k in range(MIN_EXAMS_FOR_FORECAST, len(series)):
        target = series.iloc[k]
        target_date = pd.Timestamp(target["date"]).date()
        forecast = predict_net(series.iloc[:k], target_date, bounds, level)
        if forecast is None:
            continue
        actual = float(target["net"])
        rows.append({
            "date": target_date, "actual": actual,
            "point": forecast.point, "low": forecast.low, "high": forecast.high,
            "hit": forecast.low <= actual <= forecast.high,
        })
    if not rows:
        return None
    return BacktestResult(pd.DataFrame(rows), level=level)


def exam_day_ready(result: BacktestResult | None) -> bool:
    """Sınav günü tahmini açılabilir mi?

    Kural: ampirik kapsamanın %95 Wilson aralığının alt sınırı ≥ `MIN_COVERAGE_LOWER_BOUND`.
    Yani aralıkların %80 hedefinin çok altında kalmadığına dair yeterli kanıt olmalı.
    Gereğinden geniş aralıklar (kapsama ≈ %100) temkinli ama dürüst sayılır, engellenmez.
    """
    return result is not None and result.coverage_ci[0] >= MIN_COVERAGE_LOWER_BOUND


def combine_forecasts(
    parts: list[NetForecast],
    constant: float = 0.0,
    bounds: tuple[float, float] | None = None,
) -> NetForecast:
    """Bağımsız bileşen tahminlerini (ör. puanın TYT ve AYT kısımları) tek tahminde birleştirir.

    Nokta tahmin = sabit + Σ bileşen; aralık yarı genişliği = √(Σ yarı genişlik²).
    Varsayımlar: bileşenlerin hataları bağımsız ve yaklaşık normaldir; aynı güven düzeyindedir.
    Bileşenler kırpılmamış olmalıdır; kırpma (`bounds`) birleşik sonuca uygulanır.
    """
    point = constant + sum(p.point for p in parts)
    half = float(np.sqrt(sum(((p.high - p.low) / 2) ** 2 for p in parts)))
    low, high = point - half, point + half
    if bounds is not None:
        point, low, high = (float(np.clip(v, *bounds)) for v in (point, low, high))
    return NetForecast(
        target_date=max(p.target_date for p in parts), point=point, low=low, high=high,
        n_exams=min(p.n_exams for p in parts), level=parts[0].level,
    )


def predict_combined_exam_day(
    components: list[tuple[pd.DataFrame, date]],
    constant: float,
    bounds: tuple[float, float] | None = None,
) -> NetForecast | None:
    """Farklı oturumlardan gelen bileşenlerle sınav günü puan tahmini (ör. SAY = TYT + AYT).

    `components`: (bileşen serisi, o oturumun sınav tarihi) çiftleri. Her bileşenin backtest
    kapsaması ayrı ayrı yeterli olmalı (bkz. `exam_day_ready`); değilse `None` döner.
    """
    parts = []
    for series, exam_date in components:
        if not exam_day_ready(backtest(series)):
            return None
        forecast = predict_net(series, exam_date)
        if forecast is None:
            return None
        parts.append(forecast)
    return combine_forecasts(parts, constant, bounds)


def predict_exam_day(
    series: pd.DataFrame,
    exam_date: date,
    bounds: tuple[float, float] | None = None,
) -> NetForecast | None:
    """Sınav günü net tahmini. Backtest kapsaması yeterli değilse `None` döner.

    Varsayımlar `predict_net` ile aynı; sınav günü uzaksa doğrusal trend iyimser olabilir.
    """
    if not exam_day_ready(backtest(series, bounds)):
        return None
    return predict_net(series, exam_date, bounds)


# --- Hedef takibi ------------------------------------------------------------

GOAL_REACHED = "reached"
GOAL_ON_TRACK = "on_track"
GOAL_BEHIND = "behind"
GOAL_UNCLEAR = "unclear"

# Kısa etiket (rozet) ve bir cümlelik açıklama
GOAL_MESSAGES: dict[str, str] = {
    GOAL_REACHED: "Hedefin üstündesin",
    GOAL_ON_TRACK: "Bu hızla yetişiyorsun",
    GOAL_BEHIND: "Hızlanman gerekiyor",
    GOAL_UNCLEAR: "Henüz belli değil",
}
GOAL_DETAILS: dict[str, str] = {
    GOAL_REACHED: "Trendin hedefin üstünde; seviyeni korumaya odaklan.",
    GOAL_ON_TRACK: "Gereken hız, mevcut hızının %95 aralığının altında.",
    GOAL_BEHIND: "Gereken hız, mevcut hızının %95 aralığının üstünde.",
    GOAL_UNCLEAR: "Gereken hız, mevcut hızının %95 aralığının içinde; birkaç deneme daha netleştirir.",
}


@dataclass(frozen=True)
class GoalProgress:
    """Hedef net ile mevcut trendin karşılaştırması. Hız birimi: net/hafta."""

    target_net: float
    current_level: float    # trend doğrusunun son deneme tarihindeki değeri
    weeks_left: float       # son denemeden sınav gününe kalan hafta
    required_slope: float   # hedefe sınav günü ulaşmak için gereken ortalama hız
    trend: TrendResult

    @property
    def gap(self) -> float:
        """Hedefe kalan net (negatifse hedef zaten aşılmış)."""
        return self.target_net - self.current_level

    @property
    def share_done(self) -> float:
        """Hedefin ne kadarına ulaşıldığı, 0–1 arasında (ilerleme çubuğu için)."""
        return float(np.clip(self.current_level / self.target_net, 0.0, 1.0))

    @property
    def status(self) -> str:
        """Gereken hız, eğimin %95 güven aralığıyla karşılaştırılır.

        Gereken hız aralığın altındaysa mevcut hız yeterli, üstündeyse yetersiz sayılır;
        aralığın içindeyse veri karar vermeye yetmez.
        """
        if self.gap <= 0:
            return GOAL_REACHED
        if self.required_slope <= self.trend.ci_low:
            return GOAL_ON_TRACK
        if self.required_slope > self.trend.ci_high:
            return GOAL_BEHIND
        return GOAL_UNCLEAR

    @property
    def message(self) -> str:
        return GOAL_MESSAGES[self.status]

    @property
    def detail(self) -> str:
        return GOAL_DETAILS[self.status]


def goal_progress(series: pd.DataFrame, target_net: float, exam_date: date) -> GoalProgress | None:
    """Hedef nete sınav gününe kadar ulaşmak için gereken haftalık hızı hesaplar.

    Mevcut seviye olarak son denemenin ham neti değil, trend doğrusunun o tarihteki değeri
    kullanılır (tek denemenin dalgalanmasından etkilenmesin diye). Gereken hız =
    (hedef − mevcut seviye) / kalan hafta. Bu bir tahmin değil, basit bir hız karşılaştırmasıdır;
    sınav günü net tahmini ayrıca backtest koşuluna bağlıdır.
    Varsayımlar `fit_trend` ile aynı. Trend kurulamıyorsa (3'ten az deneme) veya sınav günü
    son denemeden önce/aynı gündeyse `None` döner.
    """
    trend = fit_trend(series)
    if trend is None:
        return None
    last = pd.to_datetime(series["date"]).max()
    weeks_left = (pd.Timestamp(exam_date) - last).days / 7.0
    if weeks_left <= 0:
        return None
    current = float(trend.fitted(pd.Series([last])).iloc[0])
    return GoalProgress(
        target_net=float(target_net), current_level=current, weeks_left=weeks_left,
        required_slope=(target_net - current) / weeks_left, trend=trend,
    )


GOAL_FORECAST_MESSAGES: dict[str, str] = {
    "below": "Hedefin tahmin aralığının altında: bu gidişle hedefi geçme olasılığın yüksek (%90'dan fazla).",
    "inside": "Hedefin tahmin aralığının içinde: ulaşabilirsin ama garanti değil.",
    "above": "Hedefin tahmin aralığının üstünde: bu gidişle ulaşma olasılığın düşük (%10'dan az).",
}


def goal_vs_forecast(target_net: float, forecast: NetForecast) -> str:
    """Hedefin sınav günü tahmin aralığına göre konumu: 'below', 'inside' veya 'above'.

    %80 tahmin aralığı simetrik olduğundan her iki uçta %10 kalır; hedef alt ucun altındaysa
    hedefin geçilme olasılığı > %90, üst ucun üstündeyse < %10 olur. Varsayımlar `predict_net` ile aynı.
    """
    if target_net < forecast.low:
        return "below"
    if target_net > forecast.high:
        return "above"
    return "inside"


def focus_expected_loss(focus: pd.DataFrame) -> float:
    """Haftalık odak listesindeki konuların sınav başına toplam beklenen kayıp sorusu."""
    return float(focus["priority"].sum())


# --- Sıralama tahmini --------------------------------------------------------

@dataclass(frozen=True)
class RankEstimate:
    """Puan aralığına karşılık gelen başarı sırası aralığı (tek sayı değil)."""

    year: int              # kullanılan ÖSYM tablosunun yılı
    score_type: str        # "TYT" (sınav puanı) veya "Y-TYT" (yerleştirme puanı)
    score_low: float
    score_high: float
    rank_best: int         # yüksek puanın sırası (küçük sayı = daha iyi)
    rank_worst: int        # düşük puanın sırası


def score_to_rank(score: float, reference: pd.DataFrame) -> float:
    """Puanı, "X puan ve üstü aday sayısı" tablosunda doğrusal interpolasyonla sıraya çevirir.

    Varsayım: iki tablo satırı arasında adaylar puana eşit dağılmıştır. Tablonun üstündeki
    puanlar en iyi sıraya (en az 1), altındakiler en kötü sıraya sabitlenir.
    """
    ref = reference.sort_values("score")
    rank = float(np.interp(score, ref["score"], ref["rank"]))
    return max(rank, 1.0)


def rank_range(
    score_low: float, score_high: float, reference: pd.DataFrame, score_type: str,
) -> RankEstimate | None:
    """Puan aralığının iki ucunu sıralamaya çevirir. Tablo boşsa `None` döner.

    Varsayımlar: sınav yılındaki aday sayısı ve puan dağılımı tablonun yılıyla aynıdır.
    """
    if reference.empty:
        return None
    return RankEstimate(
        year=int(reference["year"].iloc[0]),
        score_type=score_type,
        score_low=score_low,
        score_high=score_high,
        rank_best=round(score_to_rank(score_high, reference)),
        rank_worst=round(score_to_rank(score_low, reference)),
    )


# --- Konu önceliği -----------------------------------------------------------

# Beta önselinin gücü: genel hata oranının kaç soruluk veri ağırlığında olduğu.
# Bir konuda bu kadar soru çözülene kadar tahmin, dersin genel hata oranına yakın kalır.
PRIOR_STRENGTH = 10.0
WEEKLY_TOP_N = 5
WEIGHT_OFFICIAL = "official"   # ağırlık ÖSYM'nin yayımladığı soru sayısı
WEIGHT_OBSERVED = "observed"   # ağırlık öğrencinin denemelerindeki ortalama soru sayısı

PRIORITY_COLUMNS: list[str] = [
    "subject", "topic", "weight", "weight_source", "n_questions", "errors",
    "raw_error_rate", "adjusted_error_rate", "priority",
]


def subject_error_rates(results: pd.DataFrame) -> dict[str, float]:
    """Her ders için genel hata oranı: (yanlış + boş) / (doğru + yanlış + boş), tüm denemeler."""
    totals = results.groupby("subject")[["correct", "wrong", "blank"]].sum()
    attempted = totals.sum(axis=1)
    rates = (totals["wrong"] + totals["blank"]) / attempted
    return {s: float(r) for s, r in rates.items() if attempted[s] > 0}


def topic_priority(
    topic_results: pd.DataFrame,
    topics: pd.DataFrame,
    subject_results: pd.DataFrame,
    prior_strength: float = PRIOR_STRENGTH,
) -> pd.DataFrame | None:
    """Konu öncelik skoru: P_k = w_k · (Y_k + B_k + α) / (N_k + α + β).

    - N_k: konuda girilen toplam soru (D + Y + B), Y_k + B_k: hata (yanlış + boş).
    - α, β: Beta önseli; ortalaması öğrencinin o dersteki genel hata oranı p (ders
      sonuçlarından), gücü `prior_strength`: α = κ·p, β = κ·(1 − p). Az verili konular
      ders ortalamasına çekilir, böylece 2 soruluk bir şanssızlık listeyi ele geçirmez.
    - w_k: ÖSYM'nin yayımladığı soru sayısı; yoksa öğrencinin konu sonucu girdiği
      denemelerde o konudan gelen ortalama soru sayısı.
    P_k, "bu konuda sınav başına beklenen kayıp soru" olarak okunabilir.
    Varsayımlar: denemeler arası konu zorluğu ve yayınevi farkı yok sayılır; hata oranı
    zamanla değişmez (tüm denemeler eşit ağırlıklı); boş ve yanlış aynı sayılır.
    Ağırlığı bilinmeyen ve verisi olmayan konular sonda, `priority` boş olarak döner.
    Hiç konu sonucu yoksa `None` döner.
    """
    if topic_results.empty:
        return None

    agg = (
        topic_results.groupby(["subject", "topic"])[["correct", "wrong", "blank"]].sum()
        .reset_index()
    )
    df = topics.merge(agg, on=["subject", "topic"], how="left").fillna(
        {"correct": 0, "wrong": 0, "blank": 0}
    )
    df["n_questions"] = df["correct"] + df["wrong"] + df["blank"]
    df["errors"] = df["wrong"] + df["blank"]

    exams_with_data = topic_results.groupby("subject")["exam_id"].nunique()
    observed = df["n_questions"] / df["subject"].map(exams_with_data)
    official = df["weight"].notna()
    df["weight_source"] = np.where(official, WEIGHT_OFFICIAL, WEIGHT_OBSERVED)
    df["weight"] = df["weight"].where(official, observed.where(df["n_questions"] > 0))

    prior = df["subject"].map(subject_error_rates(subject_results))
    alpha = prior_strength * prior
    beta = prior_strength * (1 - prior)
    df["raw_error_rate"] = (df["errors"] / df["n_questions"]).where(df["n_questions"] > 0)
    df["adjusted_error_rate"] = (df["errors"] + alpha) / (df["n_questions"] + alpha + beta)
    df["priority"] = df["weight"] * df["adjusted_error_rate"]

    df = df.sort_values("priority", ascending=False, na_position="last").reset_index(drop=True)
    return df[PRIORITY_COLUMNS]


def weekly_focus(priorities: pd.DataFrame, n: int = WEEKLY_TOP_N) -> pd.DataFrame:
    """Haftalık çalışma listesi: önceliği hesaplanabilen ilk `n` konu."""
    return priorities[priorities["priority"].notna()].head(n).reset_index(drop=True)


def latest_with_change(series: pd.DataFrame) -> tuple[float, float | None] | None:
    """Son denemenin neti ve bir önceki denemeye göre farkı. Tek denemede fark `None`; boşsa `None`."""
    if series.empty:
        return None
    nets = series.sort_values(["date"])["net"].to_numpy(dtype=float)
    return float(nets[-1]), (float(nets[-1] - nets[-2]) if len(nets) > 1 else None)
