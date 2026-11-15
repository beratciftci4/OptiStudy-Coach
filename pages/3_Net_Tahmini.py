"""Net tahmini: bir sonraki deneme, backtest ve sınav günü tahmini (TYT ve alan denemeleri)."""

from contextlib import closing
from datetime import date, timedelta

import streamlit as st

from optistudy import db
from optistudy.analysis import (
    GOAL_FORECAST_MESSAGES, goal_vs_forecast,
    MIN_COVERAGE_LOWER_BOUND, MIN_EXAMS_FOR_FORECAST, TOTAL_LABEL,
    backtest, exam_day_ready, net_bounds, net_series, predict_exam_day, predict_net,
)
from optistudy.charts import PLOTLY_CONFIG, backtest_figure, chart_colors, forecast_figure
from optistudy.data import FIELD_SESSION, FIELD_SUBJECTS, load_exam_dates, next_exam

st.title("Net tahmini", icon=":material/query_stats:")
COLORS = chart_colors(st.context.theme.type == "dark")

with closing(db.open_app_db()) as conn:
    field = db.get_field(conn)
    sessions = ["TYT"] + ([FIELD_SESSION[field]] if field else [])
    exam_type = st.segmented_control(
        "Deneme türü", options=sessions, default="TYT", key="forecast_exam_type", required=True,
    ) if len(sessions) > 1 else "TYT"
    results = db.subject_results_frame(conn, exam_type)
    goal = db.get_goal(conn, exam_type)
    if exam_type != "TYT":
        results = results[results["subject"].isin(FIELD_SUBJECTS[field])]

n_exams = results["exam_id"].nunique()
if n_exams < MIN_EXAMS_FOR_FORECAST:
    st.info(
        f"Tahmin için en az {MIN_EXAMS_FOR_FORECAST} {exam_type} denemesi gerekli (şu an {n_exams}). "
        "Daha fazla deneme gir.",
        icon=":material/info:",
    )
    st.page_link("pages/1_Deneme_Girisi.py", label="Deneme gir", icon=":material/edit_note:")
    st.stop()

subject = st.segmented_control(
    "Ders", options=[TOTAL_LABEL, *results["subject"].unique()], default=TOTAL_LABEL,
    key="forecast_subject" if exam_type == "TYT" else f"forecast_subject_{exam_type}", required=True,
)
series = net_series(results, subject)
bounds = net_bounds(results, subject)
last_date = series["date"].max()

# --- Bir sonraki deneme --------------------------------------------------
st.subheader("Bir sonraki deneme", icon=":material/event_upcoming:")
target = st.date_input(
    "Bir sonraki denemeni ne zaman çözeceksin?",
    value=max(date.today(), last_date + timedelta(days=1)),
    min_value=last_date, format="DD.MM.YYYY", key=f"forecast_target_{exam_type}",
)
forecast = predict_net(series, target, bounds)
if forecast is None:
    st.info(
        "Trend kurulamıyor: denemelerin en az iki farklı tarihte çözülmüş olması gerekli. "
        "Daha fazla deneme gir.",
        icon=":material/info:",
    )
    st.stop()

with st.container(horizontal=True, vertical_alignment="center"):
    st.metric(
        "Tahmini net", f"{forecast.point:.1f}", border=True,
        help="Doğrusal trendin o tarihteki değeri.",
    )
    st.metric(
        "%80 tahmin aralığı", f"{forecast.low:.1f} – {forecast.high:.1f}", border=True,
        help="Bir sonraki denemenin neti 10 denemenin yaklaşık 8'inde bu aralığa düşer.",
    )
    st.badge("Düşük güven", icon=":material/warning:", color="orange")

st.caption(
    f"{forecast.n_exams} denemeye dayanıyor. Varsayımlar: netler doğrusal değişiyor, "
    "yayınevleri arası zorluk farkı yok sayılıyor."
)

# --- Sınav günü ----------------------------------------------------------
st.subheader("Sınav günü", icon=":material/flag:")
exam = next_exam(load_exam_dates(), date.today(), exam_type)
bt = backtest(series, bounds)
exam_forecast = predict_exam_day(series, exam["date"], bounds) if exam else None

forecasts = [forecast, *([exam_forecast] if exam_forecast else [])]
st.plotly_chart(
    forecast_figure(
        results, subject, forecasts, COLORS, target_net=goal if subject == TOTAL_LABEL else None,
    ),
    config=PLOTLY_CONFIG,
    alt=f"{subject} için geçmiş netler, trend ve tahmin aralıkları",
)

if exam is None:
    st.info("Tanımlı bir sonraki sınav tarihi yok (`data/reference/exam_dates.csv`).", icon=":material/info:")
elif exam_forecast is not None:
    days_left = (exam["date"] - date.today()).days
    with st.container(horizontal=True):
        st.metric(f"{exam_type} {exam['year']} tahmini net", f"{exam_forecast.point:.1f}", border=True)
        st.metric("%80 tahmin aralığı", f"{exam_forecast.low:.1f} – {exam_forecast.high:.1f}", border=True)
    st.caption(f"Sınav tarihi: {exam['date']:%d.%m.%Y} ({days_left} gün sonra). Kaynak: {exam['source']}.")
    if goal is not None and subject == TOTAL_LABEL:
        position = goal_vs_forecast(goal, exam_forecast)
        alert = {"below": st.success, "inside": st.info, "above": st.warning}[position]
        alert(GOAL_FORECAST_MESSAGES[position], title=f"Hedefin: {goal:g} net", icon=":material/flag:")
    st.warning(
        "Sınava aylar varken doğrusal trendi o kadar ileri uzatmak iyimser bir varsayımdır: "
        "netler zamanla yavaşlayarak artar. Aralığı tek bir sayıdan daha ciddiye al.",
        icon=":material/info:",
    )
else:
    reason = (
        "Henüz backtest yapılacak kadar deneme yok."
        if bt is None else
        f"Geçmiş tahminlerin {bt.n} tanesinden {bt.hits} tanesi tuttu (%{bt.coverage * 100:.0f}). "
        f"Bu oranın %95 güven aralığının alt sınırı %{bt.coverage_ci[0] * 100:.0f}; "
        f"açılması için en az %{MIN_COVERAGE_LOWER_BOUND * 100:.0f} olmalı."
    )
    st.info(
        f"{reason} Deneme girdikçe tekrar kontrol edilir.",
        title=f"{exam_type} {exam['year']} ({exam['date']:%d.%m.%Y}) tahmini henüz kapalı",
        icon=":material/lock:",
    )

# --- Backtest ------------------------------------------------------------
st.subheader("Tahminler ne kadar güvenilir?", icon=":material/verified:")
if bt is None:
    st.info(f"Backtest için en az {MIN_EXAMS_FOR_FORECAST + 1} deneme gerekli.", icon=":material/info:")
else:
    low, high = bt.coverage_ci
    with st.container(horizontal=True):
        st.metric("Tutan tahmin", f"{bt.hits} / {bt.n}", border=True)
        st.metric(
            "Ampirik kapsama", f"%{bt.coverage * 100:.0f}", border=True,
            help="Hedef %80. Çok düşükse aralıklar fazla dar, yani tahminler aşırı iddialı.",
        )
        st.metric(
            "Kapsamanın %95 aralığı", f"%{low * 100:.0f} – %{high * 100:.0f}", border=True,
            help="Wilson aralığı. Az sayıda tahminle geniş olması normaldir.",
        )
    st.plotly_chart(backtest_figure(bt, COLORS), config=PLOTLY_CONFIG, alt="Backtest: tahmin aralıkları ve gerçekleşen netler")
    st.caption(
        "Her noktada sadece o tarihe kadarki denemelerle tahmin yapıldı ve gerçek sonuçla karşılaştırıldı."
    )
    if exam_day_ready(bt):
        st.success("Kapsama yeterli: sınav günü tahmini açık.", icon=":material/check_circle:")

st.page_link("pages/5_Siralama_Tahmini.py", label="Sıralama tahminine geç", icon=":material/leaderboard:")
