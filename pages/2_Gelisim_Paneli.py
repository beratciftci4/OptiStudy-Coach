"""Gelişim paneli: ders bazlı net trendi ve tutarlılık."""

from contextlib import closing

import streamlit as st

from optistudy import db
from optistudy.analysis import (
    MIN_EXAMS_FOR_TREND, TOTAL_LABEL, VERDICT_DECREASE, VERDICT_INCREASE, VERDICT_MESSAGES,
    consistency, progress_summary, subject_trend,
)
from optistudy.charts import PLOTLY_CONFIG, chart_colors, consistency_figure, net_trend_figure
from optistudy.data import FIELD_SESSION, FIELD_SUBJECTS

st.title("Gelişim paneli", icon=":material/monitoring:")
COLORS = chart_colors(st.context.theme.type == "dark")

with closing(db.open_app_db()) as conn:
    field = db.get_field(conn)
    sessions = ["TYT"] + ([FIELD_SESSION[field]] if field else [])
    exam_type = st.segmented_control(
        "Deneme türü", options=sessions, default="TYT", key="panel_exam_type", required=True,
    ) if len(sessions) > 1 else "TYT"
    results = db.subject_results_frame(conn, exam_type)
    if exam_type != "TYT":
        # Alan değiştirildiyse eski alanın testleri karışmasın
        results = results[results["subject"].isin(FIELD_SUBJECTS[field])]

summary = progress_summary(results)
if summary is None:
    st.info(f"Henüz {exam_type} denemesi yok. Panel için daha fazla deneme gir.", icon=":material/info:")
    st.page_link("pages/1_Deneme_Girisi.py", label="Deneme gir", icon=":material/edit_note:")
    st.stop()

subject = st.segmented_control(
    "Ders", options=[TOTAL_LABEL, *results["subject"].unique()], default=TOTAL_LABEL,
    key="panel_subject" if exam_type == "TYT" else f"panel_subject_{exam_type}", required=True,
)

row = summary.set_index("subject").loc[subject]
trend = subject_trend(results, subject)
std = consistency(results, subject)

with st.container(horizontal=True):
    st.metric("Son net", f"{row.last_net:.2f}", border=True)
    st.metric("Ortalama net", f"{row.mean_net:.2f}", border=True)
    st.metric(
        "Eğim", f"{trend.slope_per_week:+.2f} net/hafta" if trend else "—", border=True,
        help="Doğrusal regresyonla haftalık ortalama net değişimi.",
    )
    st.metric(
        "Tutarlılık", f"±{std:.2f} net" if std is not None else "—", border=True,
        help="Son 5 denemenin standart sapması. Küçük olması daha tutarlı olduğun anlamına gelir.",
    )

fig = net_trend_figure(results, subject, COLORS)
st.plotly_chart(fig, config=PLOTLY_CONFIG, alt=f"{subject} için deneme netleri ve trend çizgisi")

if trend is None:
    st.info(
        f"Trend için en az {MIN_EXAMS_FOR_TREND} deneme gerekli (şu an {int(row.n_exams)}). "
        "Daha fazla deneme gir.",
        icon=":material/info:",
    )
else:
    interval = (
        f"%95 güven aralığı: {trend.ci_low:+.2f} ile {trend.ci_high:+.2f} net/hafta "
        f"({trend.n_exams} deneme)."
    )
    if trend.verdict == VERDICT_INCREASE:
        st.success(interval, title=trend.message, icon=":material/trending_up:")
    elif trend.verdict == VERDICT_DECREASE:
        st.warning(interval, title=trend.message, icon=":material/trending_down:")
    else:
        st.info(
            interval + " Aralık sıfırı içerdiği için değişim tesadüfi olabilir.",
            title=trend.message, icon=":material/trending_flat:",
        )

st.subheader("Tüm dersler", icon=":material/table_chart:")
cons_fig = consistency_figure(summary, COLORS)
if cons_fig is None:
    st.info("Tutarlılık için en az 2 deneme gerekli.", icon=":material/info:")
else:
    st.plotly_chart(cons_fig, config=PLOTLY_CONFIG, alt="Ders bazlı tutarlılık: son 5 denemenin standart sapması")

table = summary.assign(verdict=summary["verdict"].map(VERDICT_MESSAGES))
st.dataframe(table, hide_index=True, column_config={
    "subject": "Ders",
    "n_exams": st.column_config.NumberColumn("Deneme", format="%d"),
    "last_net": st.column_config.NumberColumn("Son net", format="%.2f"),
    "mean_net": st.column_config.NumberColumn("Ortalama", format="%.2f"),
    "slope_per_week": st.column_config.NumberColumn("Eğim (net/hafta)", format="%+.2f"),
    "ci_low": st.column_config.NumberColumn("GA alt", format="%+.2f"),
    "ci_high": st.column_config.NumberColumn("GA üst", format="%+.2f"),
    "verdict": "Yorum",
    "std_last5": st.column_config.NumberColumn("Std. sapma (son 5)", format="%.2f"),
}, alt="Ders bazlı gelişim özeti tablosu")
