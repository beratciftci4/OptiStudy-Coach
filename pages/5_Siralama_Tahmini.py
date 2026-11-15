"""Sıralama tahmini: TYT ve alan puanı (SAY/EA/SÖZ/DİL) için ayrı ayrı başarı sırası aralığı."""

from contextlib import closing
from datetime import date

import streamlit as st

from optistudy import db
from optistudy.analysis import (
    MIN_COVERAGE_LOWER_BOUND, MIN_EXAMS_FOR_FORECAST, NetForecast, backtest,
    predict_combined_exam_day, predict_exam_day, rank_range,
)
from optistudy.data import (
    FIELD_NAMES, FIELD_SESSION, FIELD_SUBJECTS, load_ayt_test_stats, load_exam_dates,
    load_score_weights, load_tyt_test_stats, next_exam,
)
from optistudy.scoring import (
    TYT_SCORE_MAX, TYT_SCORE_MIN, calibrate_tyt_scale, component_score_series,
    exam_score_series, placement_score, score_constant, score_type_stats,
)

SCORE_BOUNDS = (TYT_SCORE_MIN, TYT_SCORE_MAX)


def save_diploma_grade() -> None:
    with closing(db.open_app_db()) as conn:
        db.set_diploma_grade(conn, st.session_state.diploma_grade)


def _fmt_rank(best: int, worst: int) -> str:
    return f"{best:,} – {worst:,}".replace(",", ".")


def _coverage_text(series, label: str) -> str:
    """Bir bileşenin backtest durumunu kısa cümleyle anlatır."""
    n = series["exam_id"].nunique()
    bt = backtest(series)
    if bt is None:
        return f"{label}: {n} deneme var, backtest için en az {MIN_EXAMS_FOR_FORECAST + 1} gerekli."
    return (
        f"{label}: {bt.n} tahminden {bt.hits} tanesi tuttu "
        f"(alt sınır %{bt.coverage_ci[0] * 100:.0f}, gereken %{MIN_COVERAGE_LOWER_BOUND * 100:.0f})."
    )


def show_rank_card(
    title: str, score_type: str, forecast: NetForecast | None, reasons: list[str], grade: float | None,
    exam_reference, placement_reference,
) -> None:
    """Bir puan türü için sonuç kartı: puan aralığı, sınav puanı sırası, (varsa) yerleştirme sırası."""
    with st.container(border=True, height="stretch"):
        st.subheader(title)
        if forecast is None:
            st.info(
                " ".join(reasons) + " Deneme girdikçe tekrar kontrol edilir.",
                title="Henüz kapalı", icon=":material/lock:",
            )
            return
        exam_rank = rank_range(forecast.low, forecast.high, exam_reference, score_type)
        st.metric("Tahmini puan (%80 aralık)", f"{forecast.low:.0f} – {forecast.high:.0f}")
        st.metric("Başarı sırası (sınav puanı)", _fmt_rank(exam_rank.rank_best, exam_rank.rank_worst))
        if grade is not None:
            p_low, p_high = placement_score(forecast.low, grade), placement_score(forecast.high, grade)
            placement_rank = rank_range(p_low, p_high, placement_reference, f"Y-{score_type}")
            st.metric(
                "Başarı sırası (yerleştirme puanı)",
                _fmt_rank(placement_rank.rank_best, placement_rank.rank_worst),
                help=f"Yerleştirme puanı: {p_low:.0f} – {p_high:.0f} (OBP eklenmiş)",
            )
        else:
            st.caption("Yerleştirme puanı sırası için diploma notunu gir.")
        st.caption(f"ÖSYM {exam_rank.year}-YKS puan dağılımı kullanıldı.")


st.title("Sıralama tahmini", icon=":material/leaderboard:")

with closing(db.open_app_db()) as conn:
    field = db.get_field(conn)
    tyt_results = db.subject_results_frame(conn, "TYT")
    session = FIELD_SESSION[field] if field else None
    field_results = db.subject_results_frame(conn, session) if session else None
    references = {t: db.rank_reference_frame(conn, t) for t in ("TYT", "Y-TYT")}
    if field:
        references[field] = db.rank_reference_frame(conn, field)
        references[f"Y-{field}"] = db.rank_reference_frame(conn, f"Y-{field}")
    saved_grade = db.get_diploma_grade(conn)

st.number_input(
    "Diploma notu (opsiyonel, 50–100)", min_value=50.0, max_value=100.0, step=0.5,
    value=saved_grade, placeholder="ör. 82.5", key="diploma_grade", on_change=save_diploma_grade,
    help="Tercihte geçerli yerleştirme puanı = sınav puanı + 0,12 × (diploma notu × 5).",
)
grade = st.session_state.get("diploma_grade")

exam_dates = load_exam_dates()
today = date.today()
tyt_exam = next_exam(exam_dates, today, "TYT")
tyt_stats = load_tyt_test_stats()

# --- TYT puanı -----------------------------------------------------------
tyt_forecast, tyt_reasons = None, []
if tyt_exam is None:
    tyt_reasons.append("Tanımlı bir sonraki TYT tarihi yok.")
elif tyt_results.empty:
    tyt_reasons.append("Henüz TYT denemesi yok.")
else:
    tyt_scale = calibrate_tyt_scale(tyt_stats, references["TYT"])
    tyt_scores = exam_score_series(tyt_results, tyt_stats, tyt_scale)
    tyt_forecast = predict_exam_day(tyt_scores, tyt_exam["date"], SCORE_BOUNDS)
    if tyt_forecast is None:
        tyt_reasons.append(_coverage_text(tyt_scores, "TYT puanı"))

# --- Alan puanı ----------------------------------------------------------
field_forecast, field_reasons = None, []
if field is not None:
    field_exam = next_exam(exam_dates, today, session)
    stats = score_type_stats(field, tyt_stats, load_ayt_test_stats(), load_score_weights(), references[field])
    scale = calibrate_tyt_scale(stats, references[field])
    field_part = field_results[field_results["subject"].isin(FIELD_SUBJECTS[field])]
    if tyt_exam is None or field_exam is None:
        field_reasons.append("Tanımlı bir sonraki sınav tarihi yok.")
    elif tyt_results.empty or field_part.empty:
        field_reasons.append(f"{FIELD_NAMES[field]} puanı için hem TYT hem {session} denemesi gerekli.")
    else:
        tyt_component = component_score_series(tyt_results, stats, scale)
        field_component = component_score_series(field_part, stats, scale)
        field_forecast = predict_combined_exam_day(
            [(tyt_component, tyt_exam["date"]), (field_component, field_exam["date"])],
            score_constant(scale), SCORE_BOUNDS,
        )
        if field_forecast is None:
            field_reasons += [
                _coverage_text(tyt_component, "TYT kısmı"),
                _coverage_text(field_component, f"{session} kısmı"),
            ]

if tyt_exam is not None:
    st.caption(f"Sınav: {tyt_exam['date']:%d.%m.%Y} ({(tyt_exam['date'] - today).days} gün sonra). "
               f"Kaynak: {tyt_exam['source']}.")

cols = st.columns(2)
with cols[0]:
    show_rank_card("TYT", "TYT", tyt_forecast, tyt_reasons, grade, references["TYT"], references["Y-TYT"])
with cols[1]:
    if field is None:
        with st.container(border=True, height="stretch"):
            st.subheader("Alan puanı")
            st.info("Alan sıralamasını görmek için ana sayfadan alanını seç.", icon=":material/info:")
    else:
        show_rank_card(
            f"{FIELD_NAMES[field]} ({field})", field, field_forecast, field_reasons, grade,
            references[field], references[f"Y-{field}"],
        )

st.caption(
    "Puanlar ÖSYM yöntemiyle (standart puan + kılavuzdaki test ağırlıkları) ve 2026 test istatistikleriyle "
    "yaklaşık hesaplanır; 100–500 ölçeği resmi dağılımdan kalibre edilir. Sınavın zorluğu veya aday sayısı "
    "değişirse sıralama kayar. Alan puanında TYT ve AYT kısımları ayrı tahmin edilip birleştirilir."
)
st.warning(
    "TYT sıralaması TYT puan türüyle öğrenci alan ön lisans programları içindir; lisans programlarında "
    "alan puanının sıralaması geçerlidir. Kırık OBP ve meslek lisesi ek puanı hesaba katılmadı.",
    icon=":material/school:",
)
