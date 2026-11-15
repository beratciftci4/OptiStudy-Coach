"""Net ve TYT puanı hesabı.

TYT kuralı: 4 yanlış 1 doğruyu götürür, boşlar nete etki etmez.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

WRONG_PENALTY: float = 0.25


def net(correct: int, wrong: int) -> float:
    """Tek bir ders (veya konu) için net: D − Y/4."""
    if correct < 0 or wrong < 0:
        raise ValueError("Doğru ve yanlış sayısı negatif olamaz.")
    return correct - wrong * WRONG_PENALTY


def add_net_column(results: pd.DataFrame) -> pd.DataFrame:
    """`correct` ve `wrong` sütunlarından `net` sütununu ekleyip yeni DataFrame döndürür.

    Girdi değiştirilmez.
    """
    missing = {"correct", "wrong"} - set(results.columns)
    if missing:
        raise KeyError(f"Eksik sütun(lar): {sorted(missing)}")
    out = results.copy()
    out["net"] = out["correct"] - out["wrong"] * WRONG_PENALTY
    return out


def exam_totals(results: pd.DataFrame) -> pd.DataFrame:
    """Ders bazlı sonuçlardan deneme başına toplam neti hesaplar.

    Girdi: en az `exam_id`, `correct`, `wrong` sütunları (varsa `date` korunur).
    Çıktı: deneme başına bir satır; `total_correct`, `total_wrong`, `total_net`.
    """
    scored = add_net_column(results)
    keys = ["exam_id", "date"] if "date" in scored.columns else ["exam_id"]
    totals = (
        scored.groupby(keys, as_index=False)
        .agg(
            total_correct=("correct", "sum"),
            total_wrong=("wrong", "sum"),
            total_net=("net", "sum"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return totals


def exam_net_table(results: pd.DataFrame) -> pd.DataFrame:
    """Uzun formattaki ders sonuçlarını deneme başına tek satırlık net tablosuna çevirir.

    Girdi: `date`, `publisher`, `subject`, `correct`, `wrong` (varsa `exam_id` korunur).
    Çıktı: anahtar sütunlar + her ders için net + `Toplam`. Ders sırası girdideki sırayla aynıdır.
    """
    scored = add_net_column(results)
    keys = [c for c in ("exam_id", "date", "publisher") if c in scored.columns]
    subjects = list(pd.unique(scored["subject"]))
    if scored.empty:
        return pd.DataFrame(columns=[*keys, "Toplam"])
    table = scored.pivot_table(index=keys, columns="subject", values="net", aggfunc="sum")
    table = table.reindex(columns=subjects)
    table["Toplam"] = table[subjects].sum(axis=1)
    table.columns.name = None
    return table.reset_index().sort_values(keys).reset_index(drop=True)


# --- TYT puanı ---------------------------------------------------------------

TYT_SCORE_MIN = 100.0
TYT_SCORE_MAX = 500.0
MIN_RAW_FOR_TYT = 0.5          # Türkçe veya Matematikten en az bu kadar net gerekir (Kılavuz Tablo 1D)
TYT_GATE_SUBJECTS = ("Türkçe", "Matematik")
OBP_PER_DIPLOMA_POINT = 5.0    # OBP = diploma notu × 5
OBP_COEFFICIENT = 0.12         # yerleştirme puanı = TYT puanı + 0,12 × OBP


@dataclass(frozen=True)
class TytScale:
    """Ağırlıklı standart puandan (A-TYT) 100–500 TYT puanına doğrusal dönüşüm: P = intercept + slope · A."""

    year: int
    intercept: float
    slope: float


def weighted_standard_score(nets: Mapping[str, float], stats: pd.DataFrame) -> float:
    """ÖSYM yöntemiyle ağırlıklı standart puan (A-TYT).

    Her test: SP = 50 + 10 · (net − ortalama) / standart sapma (son sınıf adaylarının
    ortalaması ve standart sapması). A-TYT = Σ ağırlık · SP.
    """
    total = 0.0
    for row in stats.itertuples(index=False):
        sp = 50 + 10 * (nets[row.subject] - row.mean_final_year) / row.sd_final_year
        total += row.weight * sp
    return total


def distribution_mean(reference: pd.DataFrame) -> float:
    """"X puan ve üstü aday sayısı" tablosundan ortalama puanı yaklaşık hesaplar.

    Her aralıktaki adaylar aralığın orta noktasında sayılır; en üst satır (ör. tam 500)
    kendi puanında sayılır.
    """
    ref = reference.sort_values("score", ascending=False)
    scores = ref["score"].to_numpy(dtype=float)
    cumulative = ref["rank"].to_numpy(dtype=float)
    counts = np.diff(np.concatenate([[0.0], cumulative]))
    mids = np.concatenate([[scores[0]], (scores[1:] + scores[:-1]) / 2])
    return float(np.dot(counts, mids) / cumulative[-1])


def calibrate_tyt_scale(stats: pd.DataFrame, reference: pd.DataFrame) -> TytScale:
    """A-TYT → TYT puanı dönüşümünü resmi verilerden kalibre eder.

    ÖSYM A-TYT'yi "en küçüğü 100, en büyüğü 500" olacak şekilde ölçekler ama o yılın en
    küçük ve en büyük A-TYT değerini yayımlamaz. Bu yüzden iki bilinen noktadan doğrusal
    ölçek kurulur:
      1. Tam net → 500 (en yüksek A-TYT tam nettir; 2026'da 500 alan aday var).
      2. Tüm adayların ortalama netleri → dağılım tablosundan hesaplanan ortalama puan.
    Varsayımlar: ortalama netlerin A-TYT'si adayların ortalama A-TYT'sine eşittir
    (A-TYT doğrusal olduğu için doğru; ancak puanı hesaplanmayan az sayıda düşük netli aday
    ortalamayı hafifçe kaydırır); dağılım tablosunun orta nokta yaklaşımı yeterince doğrudur.
    """
    full = {r.subject: float(r.question_count) for r in stats.itertuples(index=False)}
    mean_all = {r.subject: float(r.mean_all) for r in stats.itertuples(index=False)}
    a_full = weighted_standard_score(full, stats)
    a_mean = weighted_standard_score(mean_all, stats)
    p_mean = distribution_mean(reference)
    slope = (TYT_SCORE_MAX - p_mean) / (a_full - a_mean)
    intercept = TYT_SCORE_MAX - slope * a_full
    return TytScale(year=int(stats["year"].iloc[0]), intercept=intercept, slope=slope)


def tyt_score_linear(nets: Mapping[str, float], stats: pd.DataFrame, scale: TytScale) -> float:
    """Kırpılmamış TYT puanı (netlerin doğrusal fonksiyonu). Trend ve tahmin için kullanılır."""
    return scale.intercept + scale.slope * weighted_standard_score(nets, stats)


def tyt_score(nets: Mapping[str, float], stats: pd.DataFrame, scale: TytScale) -> float | None:
    """Ders netlerinden yaklaşık TYT puanı (100–500).

    Türkçe ve Matematik netinin ikisi de 0,5'in altındaysa ÖSYM puan hesaplamaz → `None`.
    Varsayımlar için bkz. `calibrate_tyt_scale`; puan geçen yılın test istatistiklerine dayanır.
    """
    if all(nets[s] < MIN_RAW_FOR_TYT for s in TYT_GATE_SUBJECTS):
        return None
    return float(np.clip(tyt_score_linear(nets, stats, scale), TYT_SCORE_MIN, TYT_SCORE_MAX))


def placement_score(tyt: float, diploma_grade: float) -> float:
    """Yerleştirme puanı: TYT puanı + 0,12 × OBP (OBP = diploma notu × 5).

    Kırık OBP (önceki yıl yerleşenlerde katsayının yarıya inmesi) ve meslek lisesi ek
    puanı hesaba katılmaz.
    """
    return tyt + OBP_COEFFICIENT * OBP_PER_DIPLOMA_POINT * diploma_grade


def exam_score_series(results: pd.DataFrame, stats: pd.DataFrame, scale: TytScale) -> pd.DataFrame:
    """Her deneme için (kırpılmamış) TYT puanı: `exam_id`, `date`, `publisher`, `net`.

    Sütun adı `net` tutuldu ki trend/tahmin/backtest fonksiyonları aynen kullanılabilsin.
    Puan ders netlerinin doğrusal fonksiyonu olduğundan bu, ağırlıklı bir toplam net serisidir.
    """
    table = exam_net_table(results)
    subjects = list(stats["subject"])
    scores = [
        tyt_score_linear(dict(zip(subjects, row)), stats, scale)
        for row in table[subjects].itertuples(index=False)
    ]
    return table[["exam_id", "date", "publisher"]].assign(net=scores)


# --- Alan puanları (SAY, EA, SÖZ, DİL) ---------------------------------------

# Ağırlıklı puanın hesaplanması için gereken asgari ham puan (Kılavuz Tablo 1D):
# her gruptaki testlerden en az birinde ≥ 0,5 ham puan. Bir test birden fazla dersten oluşabilir.
FIELD_GATES: dict[str, list[list[tuple[str, ...]]]] = {
    "SAY": [[("Türkçe",), ("Matematik",)], [("AYT Matematik",), ("Fizik", "Kimya", "Biyoloji")]],
    "EA": [[("Türkçe",), ("Matematik",)], [("AYT Matematik",), ("Edebiyat", "Tarih-1", "Coğrafya-1")]],
    "SÖZ": [[("Türkçe",), ("Matematik",)],
            [("Edebiyat", "Tarih-1", "Coğrafya-1"), ("Tarih-2", "Coğrafya-2", "Felsefe Grubu", "Din Kültürü")]],
    "DİL": [[("Türkçe",), ("Matematik",)], [("Yabancı Dil",)]],
}


def score_type_stats(
    score_type: str,
    tyt_stats: pd.DataFrame,
    ayt_stats: pd.DataFrame,
    weights: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Bir alan puan türü için ders bazlı ağırlık ve istatistik tablosu.

    Çıktı `calibrate_tyt_scale` ve `tyt_score_linear` ile aynı biçimdedir (subject, weight,
    question_count, mean_final_year, sd_final_year, mean_all, year, session); böylece aynı
    standart puan ve kalibrasyon kodu her puan türünde kullanılır.
    Nüfus düzeltmesi: AYT/YDT testlerinin "tüm adaylar" ortalaması, o puan türünde puanı
    hesaplanan adaylara göre ölçeklenir (n_tüm / puanı olan aday). Puanı hesaplanmayanların
    tanım gereği bu testlerde neti ≈ 0 olduğu varsayılır. TYT testlerine düzeltme uygulanmaz.
    """
    w = weights.loc[weights["score_type"] == score_type, ["subject", "weight"]]
    holders = float(reference["rank"].max())
    tyt = tyt_stats.drop(columns=["weight"], errors="ignore").merge(w, on="subject")
    tyt = tyt.assign(session="TYT")
    ayt = ayt_stats.merge(w, on="subject").rename(columns={"valid_question_count": "question_count"})
    ayt = ayt.assign(mean_all=ayt["mean_all"] * ayt["n_all"] / holders, session="AYT")
    cols = ["year", "subject", "session", "weight", "question_count", "mean_final_year", "sd_final_year", "mean_all"]
    out = pd.concat([tyt[cols], ayt[cols]], ignore_index=True)
    if not np.isclose(out["weight"].sum(), 1.0):
        raise ValueError(f"{score_type} ağırlıklarının toplamı 1 olmalı ({out['weight'].sum():.3f}).")
    return out


def passes_gates(nets: Mapping[str, float], score_type: str) -> bool:
    """ÖSYM'nin asgari ham puan şartı (Tablo 1D) sağlanıyor mu?"""
    gates = [[("Türkçe",), ("Matematik",)]] if score_type == "TYT" else FIELD_GATES[score_type]
    return all(
        any(sum(nets.get(s, 0.0) for s in test) >= MIN_RAW_FOR_TYT for test in group)
        for group in gates
    )


def field_score(
    nets: Mapping[str, float], stats: pd.DataFrame, scale: TytScale, score_type: str,
) -> float | None:
    """Ders netlerinden yaklaşık alan puanı (100–500). Şart sağlanmazsa `None`.

    Varsayımlar için bkz. `score_type_stats` ve `calibrate_tyt_scale`.
    """
    if not passes_gates(nets, score_type):
        return None
    return float(np.clip(tyt_score_linear(nets, stats, scale), TYT_SCORE_MIN, TYT_SCORE_MAX))


def score_constant(scale: TytScale) -> float:
    """Puanın netlerden bağımsız kısmı: P = sabit + Σ bileşen (Σ ağırlık = 1 iken sabit + 50·eğim)."""
    return scale.intercept + 50.0 * scale.slope


def component_score_series(results: pd.DataFrame, stats: pd.DataFrame, scale: TytScale) -> pd.DataFrame:
    """Puanın bir oturuma (TYT veya AYT) düşen kısmının deneme bazlı serisi (`net` sütununda).

    Bileşen = eğim · Σ ağırlık · 10 · (net − ortalama) / standart sapma, sadece `results`'ta
    bulunan derslerden. Bütün puan = `score_constant(scale)` + TYT bileşeni + AYT bileşeni.
    """
    table = exam_net_table(results)
    part = stats[stats["subject"].isin(table.columns)]
    value = sum(
        scale.slope * r.weight * 10 * (table[r.subject] - r.mean_final_year) / r.sd_final_year
        for r in part.itertuples(index=False)
    )
    return table[["exam_id", "date", "publisher"]].assign(net=value)
