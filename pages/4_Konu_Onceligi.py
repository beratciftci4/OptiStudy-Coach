"""Haftalık konu önceliği: Bayesçi düzeltilmiş skorla ilk 5 konu."""

from contextlib import closing
from datetime import date

import streamlit as st

from optistudy import db
from optistudy.analysis import (
    TOTAL_LABEL, focus_expected_loss, goal_progress, net_series,
    PRIOR_STRENGTH, WEEKLY_TOP_N, WEIGHT_OFFICIAL, topic_priority, weekly_focus,
)
from optistudy.charts import PLOTLY_CONFIG, chart_colors, priority_figure
from optistudy.data import FIELD_SESSION, FIELD_SUBJECTS, load_exam_dates, next_exam

WEIGHT_LABELS = {WEIGHT_OFFICIAL: "ÖSYM", "observed": "Denemelerin"}

st.title("Konu önceliği", icon=":material/target:")
COLORS = chart_colors(st.context.theme.type == "dark")

with closing(db.open_app_db()) as conn:
    field = db.get_field(conn)
    sessions = ["TYT"] + ([FIELD_SESSION[field]] if field else [])
    exam_type = st.segmented_control(
        "Deneme türü", options=sessions, default="TYT", key="priority_exam_type", required=True,
        help="TYT ve alan denemelerinin soruları farklı puan ağırlığında olduğu için ayrı listelenir.",
    ) if len(sessions) > 1 else "TYT"
    topic_results = db.topic_results_frame(conn, exam_type)
    topics = db.topics_frame(conn, exam_type)
    subject_results = db.subject_results_frame(conn, exam_type)
    goal = db.get_goal(conn, exam_type)
    if exam_type != "TYT":
        own = list(FIELD_SUBJECTS[field])
        topic_results = topic_results[topic_results["subject"].isin(own)]
        topics = topics[topics["subject"].isin(own)]
        subject_results = subject_results[subject_results["subject"].isin(own)]

priorities = topic_priority(topic_results, topics, subject_results)
if priorities is None:
    st.info(
        "Konu önceliği için konu bazlı sonuç gerekiyor. Deneme girerken her dersin altındaki "
        "\"Konu bazlı sonuç\" bölümünü doldur ya da konu CSV'si yükle.",
        icon=":material/info:",
    )
    st.page_link("pages/1_Deneme_Girisi.py", label="Deneme gir", icon=":material/edit_note:")
    st.stop()

focus = weekly_focus(priorities)
n_exams = topic_results["exam_id"].nunique()
st.markdown(
    f"Konu bazlı sonuç girdiğin **{n_exams} denemeye** göre, sınavda en çok soru kaybettiğin "
    f"**{len(focus)} konu**. Bu hafta bunlara öncelik ver."
)

exam = next_exam(load_exam_dates(), date.today(), exam_type)
progress = (
    goal_progress(net_series(subject_results, TOTAL_LABEL), goal, exam["date"])
    if goal is not None and exam is not None else None
)
if progress is not None and progress.gap > 0:
    st.info(
        f"Bu {len(focus)} konu sınav başına toplam ~{focus_expected_loss(focus):.1f} soru kaybettiriyor; "
        f"geri kazandığın her soru yaklaşık 1 net. Hedefe giden gereken hız: "
        f"{progress.required_slope:+.2f} net/hafta.",
        title=f"Hedefine {progress.gap:.1f} net var ({goal:g} net)", icon=":material/flag:",
    )
elif goal is None:
    st.caption("Ana sayfadan hedef net belirlersen bu listeyi hedefinle birlikte görürsün.")

for i, row in enumerate(focus.itertuples(index=False), start=1):
    with st.container(border=True):
        cols = st.columns([3, 1, 1, 1], vertical_alignment="center")
        cols[0].markdown(f"**{i}. {row.topic}**  \n:gray[{row.subject}]")
        cols[1].metric("Beklenen kayıp", f"{row.priority:.1f} soru")
        cols[2].metric("Hata oranı", f"%{row.adjusted_error_rate * 100:.0f}",
                       help=f"Ham oran: %{row.raw_error_rate * 100:.0f} ({int(row.errors)}/{int(row.n_questions)} soru)")
        cols[3].metric("Soru ağırlığı", f"{row.weight:.1f}",
                       help=f"Kaynak: {WEIGHT_LABELS[row.weight_source]} ortalaması"
                       if row.weight_source != WEIGHT_OFFICIAL else "Kaynak: ÖSYM kılavuzu")

fig = priority_figure(focus, COLORS)
if fig is not None:
    st.plotly_chart(fig, config=PLOTLY_CONFIG, alt="Haftalık öncelikli konular ve beklenen kayıp soru")

with st.expander("Nasıl hesaplanıyor?", icon=":material/help:"):
    st.markdown(
        f"""
- **Beklenen kayıp** = soru ağırlığı × düzeltilmiş hata oranı. Yani o konuda sınav başına
  kaç soruyu yanlış veya boş bırakmanın beklendiği.
- **Düzeltilmiş hata oranı**: (yanlış + boş + α) / (soru + α + β). α ve β, o dersteki genel
  hata oranından türetilir ve {PRIOR_STRENGTH:.0f} soruluk bir ön bilgi gibi davranır. Az soru
  çözdüğün bir konuda tek bir kötü deneme listeyi bozmasın diye oran ders ortalamasına çekilir.
- **Soru ağırlığı**: Sosyal ve Fen için ÖSYM kılavuzundaki resmi soru sayısı. Türkçe ve
  Matematikte ÖSYM konu bazlı sayı yayımlamadığı için senin denemelerindeki ortalama soru sayısı.
- Boş ve yanlış aynı sayılır, yayınevleri arası zorluk farkı yok sayılır.
"""
    )

st.subheader("Tüm konular", icon=":material/table_chart:")
table = priorities.assign(weight_source=priorities["weight_source"].map(WEIGHT_LABELS))
st.dataframe(table, hide_index=True, column_config={
    "subject": "Ders",
    "topic": "Konu",
    "weight": st.column_config.NumberColumn("Ağırlık (soru)", format="%.1f"),
    "weight_source": "Ağırlık kaynağı",
    "n_questions": st.column_config.NumberColumn("Girilen soru", format="%d"),
    "errors": st.column_config.NumberColumn("Yanlış + boş", format="%d"),
    "raw_error_rate": st.column_config.NumberColumn("Ham hata", format="percent"),
    "adjusted_error_rate": st.column_config.NumberColumn("Düzeltilmiş hata", format="percent"),
    "priority": st.column_config.NumberColumn("Beklenen kayıp", format="%.2f"),
}, alt="Tüm konuların öncelik tablosu")
st.caption(
    f"İlk {WEEKLY_TOP_N} konu haftalık listeye alınır. Ağırlığı bilinmeyen ve hiç verisi "
    "olmayan konuların önceliği hesaplanamaz."
)
