"""OptiStudy Streamlit giriş noktası. Sadece arayüz; hesaplama `optistudy/` içinde."""

from contextlib import closing
from datetime import date

import streamlit as st

from optistudy import db
from optistudy.analysis import (
    GOAL_BEHIND, GOAL_ON_TRACK, GOAL_REACHED, TOTAL_LABEL, VERDICT_DECREASE, VERDICT_INCREASE,
    focus_expected_loss, goal_progress, latest_with_change, net_series, subject_trend, topic_priority,
    weekly_focus,
)
from optistudy.data import (
    FIELD_NAMES, FIELD_SESSION, FIELD_SUBJECTS, FIELDS, load_exam_dates, load_subjects, next_exam,
    session_max_net,
)

st.set_page_config(page_title="OptiStudy", page_icon=":material/school:", layout="wide")

ENTRY_PAGE = "pages/1_Deneme_Girisi.py"


def save_field() -> None:
    with closing(db.open_app_db()) as conn:
        db.set_field(conn, st.session_state.home_field)


GOAL_STYLE = {
    GOAL_REACHED: ("green", ":material/emoji_events:"),
    GOAL_ON_TRACK: ("green", ":material/check_circle:"),
    GOAL_BEHIND: ("orange", ":material/speed:"),
}


def save_goal(exam_type: str, max_net: float) -> None:
    with closing(db.open_app_db()) as conn:
        db.set_goal(conn, exam_type, st.session_state[f"goal_input_{exam_type}"], max_net)
    st.toast("Hedef kaydedildi.", icon=":material/flag:")


def clear_goal(exam_type: str) -> None:
    with closing(db.open_app_db()) as conn:
        db.set_goal(conn, exam_type, None)
    st.session_state[f"goal_input_{exam_type}"] = None


def goal_editor(exam_type: str, goal: float | None, max_net: float) -> None:
    """Hedef belirleme/değiştirme açılır kutusu."""
    key = f"goal_input_{exam_type}"
    with st.popover("Hedefi değiştir" if goal else "Hedef belirle", icon=":material/flag:"):
        st.number_input(
            f"{exam_type} hedef toplam net", min_value=1.0, max_value=max_net, step=1.0,
            value=goal, placeholder=f"ör. {max_net * 0.6:.0f}", key=key,
            help=f"En fazla {max_net:g} net.",
        )
        with st.container(horizontal=True):
            st.button(
                "Kaydet", type="primary", icon=":material/save:", key=f"goal_save_{exam_type}",
                on_click=save_goal, args=(exam_type, max_net), disabled=st.session_state.get(key) is None,
            )
            if goal is not None:
                st.button(
                    "Hedefi kaldır", icon=":material/delete:", key=f"goal_clear_{exam_type}",
                    on_click=clear_goal, args=(exam_type,),
                )


def goal_section(exam_type: str, results, goal: float | None, max_net: float, focus) -> None:
    """Hedefe ilerleme: çubuk, gereken/mevcut hız ve hedefin bu haftaki konularla bağlantısı."""
    if goal is None:
        st.caption("Hedef net belirlersen sınava kadar haftada kaç net artman gerektiğini gösteririm.")
        goal_editor(exam_type, goal, max_net)
        return

    exam = next_exam(load_exam_dates(), date.today(), exam_type)
    progress = goal_progress(net_series(results, TOTAL_LABEL), goal, exam["date"]) if exam else None
    if progress is None:
        st.progress(0.0, text=f"Hedef: **{goal:g} net**")
        st.caption(
            "Takvimde sınav tarihi yok." if exam is None else
            "Hedefe giden hızı hesaplamak için en az 3 deneme gerekli."
        )
    else:
        st.progress(
            progress.share_done,
            text=f"Hedef **{goal:g} net** · trendine göre şu an **{progress.current_level:.1f}**",
        )
        color, icon = GOAL_STYLE.get(progress.status, ("gray", ":material/help:"))
        st.badge(progress.message, icon=icon, color=color, help=progress.detail)
        if progress.gap > 0:
            t = progress.trend
            st.caption(
                f"Gereken hız: **{progress.required_slope:+.2f} net/hafta** ({progress.weeks_left:.0f} hafta). "
                f"Şu anki hızın: {t.slope_per_week:+.2f} (%95 aralık {t.ci_low:+.2f} – {t.ci_high:+.2f}). "
                f"{progress.detail}"
            )
            if focus is not None:
                st.caption(
                    f"Hedefine {progress.gap:.1f} net var. Aşağıdaki {len(focus)} konu sınav başına toplam "
                    f"~{focus_expected_loss(focus):.1f} soru kaybettiriyor; geri kazandığın her soru "
                    "yaklaşık 1 net (yanlıştan doğruya 1,25)."
                )
        else:
            st.caption(progress.detail)
    goal_editor(exam_type, goal, max_net)


def countdown(exam_type: str) -> None:
    """Bir sonraki sınava kalan gün kartı. Takvimde sınav yoksa hiçbir şey göstermez."""
    exam = next_exam(load_exam_dates(), date.today(), exam_type)
    if exam is None:
        return
    st.metric(
        f"{exam_type} sınavına", f"{(exam['date'] - date.today()).days} gün",
        icon=":material/event:", border=True, width="stretch",
        help=f"{exam['date']:%d.%m.%Y}. Kaynak: {exam['source']}.",
    )


def session_summary(
    title: str, results, topic_results, topics, goal: float | None, max_net: float,
) -> None:
    """Bir oturumun (TYT veya alan) kısa özeti: deneme sayısı, son net, trend, hedef ve ilk 3 konu."""
    with st.container(border=True, height="stretch"):
        st.subheader(title, icon=":material/assignment:")
        n = results["exam_id"].nunique()
        series = net_series(results, TOTAL_LABEL)
        latest = latest_with_change(series)
        with st.container(horizontal=True):
            st.metric(f"{title} denemesi", n, width="content")
            if latest is not None:
                last, change = latest
                st.metric(
                    "Son net", f"{last:.2f}", delta=None if change is None else f"{change:+.2f}",
                    chart_data=series["net"].round(2).tolist() if n >= 2 else None, chart_type="area",
                    delta_description="önceki denemeye göre",
                )
        if n == 0:
            st.caption(f"Henüz {title} denemesi yok.")
            st.page_link(ENTRY_PAGE, label="İlk denemeni gir", icon=":material/arrow_forward:")
            goal_section(title, results, goal, max_net, None)
            return

        trend = subject_trend(results, TOTAL_LABEL)
        if trend is None:
            st.caption("Trend için en az 3 deneme gerekli.")
        else:
            color, icon = {
                VERDICT_INCREASE: ("green", ":material/trending_up:"),
                VERDICT_DECREASE: ("orange", ":material/trending_down:"),
            }.get(trend.verdict, ("gray", ":material/trending_flat:"))
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.badge(trend.message, icon=icon, color=color)
                st.caption(f"Eğim: {trend.slope_per_week:+.2f} net/hafta")

        priorities = topic_priority(topic_results, topics, results)
        focus = None if priorities is None else weekly_focus(priorities, n=3)
        goal_section(title, results, goal, max_net, focus)
        if focus is None:
            st.caption("Konu önerisi için deneme girerken konu bazlı sonuç da gir.")
        else:
            lines = [
                f"{i}. {r.topic} :gray[({r.subject}, ~{r.priority:.1f} soru kaybı)]"
                for i, r in enumerate(focus.itertuples(index=False), start=1)
            ]
            st.markdown("**Bu hafta odaklan:**\n\n" + "\n".join(lines))


def home() -> None:
    st.title("OptiStudy", icon=":material/school:")
    st.caption("Deneme sonuçlarını *bu hafta ne çalışmalıyım?* sorusunun cevabına çeviren kişisel YKS koçun.")

    with closing(db.open_app_db()) as conn:
        field = db.get_field(conn)
        goals = db.goals_dict(conn)
        session = FIELD_SESSION[field] if field else None
        data = {"TYT": (db.subject_results_frame(conn, "TYT"), db.topic_results_frame(conn, "TYT"),
                        db.topics_frame(conn, "TYT"))}
        if session:
            own = list(FIELD_SUBJECTS[field])
            r, tr, t = (db.subject_results_frame(conn, session), db.topic_results_frame(conn, session),
                        db.topics_frame(conn, session))
            data[session] = (r[r["subject"].isin(own)], tr[tr["subject"].isin(own)], t[t["subject"].isin(own)])

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.segmented_control(
            "Alanın", options=list(FIELDS), default=field, key="home_field", on_change=save_field,
            format_func=lambda f: f"{FIELD_NAMES[f]} ({f})",
            help="AYT/YDT denemelerinde hangi testleri gireceğini ve alan sıralamanı belirler.",
        )
        for exam_type in data:
            countdown(exam_type)
    if field is None:
        st.caption("Alanını seçersen AYT (Dil için YDT) denemelerini de girip alan sıralamanı görebilirsin.")

    if all(results.empty for results, _, _ in data.values()):
        st.info(
            "Deneme girdikçe netlerin, trendin ve bu hafta odaklanman gereken konular burada görünür.",
            title="Başlamak için ilk denemeni gir", icon=":material/rocket_launch:",
        )

    cols = st.columns(len(data))
    for col, (title, (results, topic_results, topics)) in zip(cols, data.items()):
        with col:
            session_summary(
                title, results, topic_results, topics, goals.get(title),
                session_max_net(load_subjects(), title, field),
            )

    st.subheader("Hızlı erişim")
    with st.container(horizontal=True):
        st.page_link(ENTRY_PAGE, label="Deneme gir", icon=":material/edit_note:")
        st.page_link("pages/4_Konu_Onceligi.py", label="Bu hafta ne çalışmalıyım?", icon=":material/target:")
        st.page_link("pages/2_Gelisim_Paneli.py", label="Gelişimim", icon=":material/monitoring:")
        st.page_link("pages/3_Net_Tahmini.py", label="Net tahmini", icon=":material/query_stats:")
        st.page_link("pages/5_Siralama_Tahmini.py", label="Sıralama tahmini", icon=":material/leaderboard:")


page = st.navigation(
    {
        "": [st.Page(home, title="Ana sayfa", icon=":material/home:", default=True)],
        "Veri": [st.Page(ENTRY_PAGE, title="Deneme girişi", icon=":material/edit_note:")],
        "Analiz": [
            st.Page("pages/2_Gelisim_Paneli.py", title="Gelişim paneli", icon=":material/monitoring:"),
            st.Page("pages/3_Net_Tahmini.py", title="Net tahmini", icon=":material/query_stats:"),
            st.Page("pages/5_Siralama_Tahmini.py", title="Sıralama tahmini", icon=":material/leaderboard:"),
            st.Page("pages/4_Konu_Onceligi.py", title="Konu önceliği", icon=":material/target:"),
        ],
    },
    position="sidebar",
)
page.run()
