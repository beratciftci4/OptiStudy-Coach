"""Deneme girişi: TYT ve alan denemesi (AYT/YDT), elle giriş, CSV içe aktarma, silme."""

from contextlib import closing
from datetime import date, datetime

import streamlit as st

from optistudy import db
from optistudy.data import (
    FIELD_NAMES, FIELD_SESSION, ValidationError, load_subjects, load_topics, read_exam_csv,
    read_topic_csv, required_subjects, validate_counts, validate_topic_totals,
)
from optistudy.scoring import exam_net_table, net

SUBJECTS = load_subjects()
QUESTION_COUNTS = dict(zip(SUBJECTS["name"], SUBJECTS["question_count"]))
TOPICS = load_topics()
FIELDS = (("correct", "Doğru"), ("wrong", "Yanlış"), ("blank", "Boş"))


def _key(subject: str, field: str) -> str:
    return f"entry_{subject}_{field}"


def _topic_key(subject: str, topic: str, field: str) -> str:
    return f"topic_{subject}_{topic}_{field}"


def _entered_results(subjects: list[str]) -> dict[str, tuple[int, int, int]]:
    return {s: tuple(int(st.session_state.get(_key(s, f), 0)) for f, _ in FIELDS) for s in subjects}


def _entered_topics(subjects: list[str]) -> dict[tuple[str, str], tuple[int, int, int]]:
    """Girilen konu sonuçları; hepsi sıfır olan konular dahil edilmez."""
    out = {}
    for subject, topic in TOPICS[["subject", "name"]].itertuples(index=False):
        if subject not in subjects:
            continue
        values = tuple(int(st.session_state.get(_topic_key(subject, topic, f), 0)) for f, _ in FIELDS)
        if sum(values):
            out[(subject, topic)] = values
    return out


def _reset_inputs() -> None:
    for name in SUBJECTS["name"]:
        for f, _ in FIELDS:
            st.session_state[_key(name, f)] = 0
    for subject, topic in TOPICS[["subject", "name"]].itertuples(index=False):
        for f, _ in FIELDS:
            st.session_state[_topic_key(subject, topic, f)] = 0
    st.session_state.entry_notes = ""


def save_exam(exam_type: str, subjects: list[str]) -> None:
    """Kaydet butonunun geri çağrısı: kaydeder, başarılıysa formu sıfırlar."""
    try:
        with closing(db.open_app_db()) as conn:
            db.add_exam(
                conn,
                st.session_state.entry_date.isoformat(),
                st.session_state.entry_publisher.strip(),
                _entered_results(subjects),
                st.session_state.entry_notes.strip() or None,
                topic_results=_entered_topics(subjects),
                exam_type=exam_type,
            )
    except ValueError as exc:
        st.session_state.entry_error = str(exc)
        return
    st.session_state.entry_error = None
    st.session_state.entry_saved = exam_type
    _reset_inputs()


def import_csv(frame) -> None:
    try:
        with closing(db.open_app_db()) as conn:
            ids = db.import_exam_frame(conn, frame)
    except ValueError as exc:
        st.session_state.import_error = str(exc)
        return
    st.session_state.import_error = None
    st.session_state.import_count = len(ids)
    st.session_state.csv_upload_version += 1  # yükleme alanını temizle


def import_topic_csv(frame) -> None:
    try:
        with closing(db.open_app_db()) as conn:
            n = db.import_topic_frame(conn, frame)
    except ValueError as exc:
        st.session_state.topic_import_error = str(exc)
        return
    st.session_state.topic_import_error = None
    st.session_state.topic_import_count = n
    st.session_state.topic_upload_version += 1


def start_fresh() -> None:
    """Sıfırdan başla butonunun geri çağrısı: veritabanını yedek adıyla kenara alır."""
    backup = db.start_fresh(db.app_db_path(), datetime.now().strftime("%Y%m%d-%H%M%S"))
    st.session_state.fresh_confirm = False
    st.session_state.fresh_backup = backup.name if backup else ""


def delete_selected() -> None:
    with closing(db.open_app_db()) as conn:
        db.delete_exam(conn, st.session_state.delete_exam_id)
    st.session_state.delete_confirm = False
    st.toast("Deneme silindi.", icon=":material/delete:")


st.session_state.setdefault("entry_error", None)
st.session_state.setdefault("import_error", None)
st.session_state.setdefault("csv_upload_version", 0)
st.session_state.setdefault("topic_import_error", None)
st.session_state.setdefault("topic_upload_version", 0)

with closing(db.open_app_db()) as conn:
    student_field = db.get_field(conn)

st.title("Deneme girişi", icon=":material/edit_note:")
manual_tab, csv_tab, manage_tab = st.tabs([
    ":material/keyboard: Elle giriş", ":material/upload_file: CSV içe aktar", ":material/database: Veri yönetimi",
])

# --- Elle giriş ----------------------------------------------------------
with manual_tab:
    sessions = ["TYT"] + ([FIELD_SESSION[student_field]] if student_field else [])
    exam_type = st.segmented_control(
        "Deneme türü", options=sessions, default="TYT", key="entry_exam_type", required=True,
    )
    if student_field is None:
        st.caption("AYT veya YDT denemesi girmek için ana sayfadan alanını seç.")
    else:
        st.caption(f"Alanın: {FIELD_NAMES[student_field]} ({student_field})")

    subjects = required_subjects(SUBJECTS, exam_type, student_field)

    with st.container(horizontal=True, border=True):
        st.date_input("Tarih", value=date.today(), key="entry_date", format="DD.MM.YYYY")
        st.text_input("Yayınevi", key="entry_publisher", placeholder="ör. Yayınevi A")

    errors: list[str] = []
    total_net = 0.0
    entered = _entered_results(subjects)
    for subject in subjects:
        qc = int(QUESTION_COUNTS[subject])
        with st.container(border=True):
            cols = st.columns([2, 1, 1, 1, 1], vertical_alignment="bottom")
            cols[0].markdown(f"**{subject}**  \n:gray[{qc} soru]")
            for col, (field, label) in zip(cols[1:4], FIELDS):
                col.number_input(label, min_value=0, max_value=qc, step=1, key=_key(subject, field))
            c, w, b = entered[subject]
            try:
                validate_counts(c, w, b, qc, label=subject)
                subject_net = net(c, w)
                total_net += subject_net
                cols[4].metric("Net", f"{subject_net:.2f}")
            except ValidationError as exc:
                errors.append(str(exc))
                cols[4].metric("Net", "—")

            subject_topics = list(TOPICS.loc[TOPICS["subject"] == subject, "name"])
            topic_rows = []
            with st.expander("Konu bazlı sonuç (opsiyonel)", icon=":material/list:"):
                for topic in subject_topics:
                    tcols = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
                    tcols[0].markdown(topic)
                    for col, (field, label) in zip(tcols[1:], FIELDS):
                        col.number_input(
                            label, min_value=0, max_value=qc, step=1,
                            key=_topic_key(subject, topic, field),
                        )
                    topic_rows.append({
                        f: int(st.session_state.get(_topic_key(subject, topic, f), 0)) for f, _ in FIELDS
                    })
                st.caption("Sadece hatırladığın konuları girmen yeterli; toplam ders sonucunu aşamaz.")
            try:
                validate_topic_totals(
                    {"correct": c, "wrong": w, "blank": b}, topic_rows, label=f"{subject} konuları",
                )
            except ValidationError as exc:
                errors.append(str(exc))

    st.text_input("Not (opsiyonel)", key="entry_notes")

    for message in errors:
        st.error(message, icon=":material/error:")

    publisher_missing = not st.session_state.entry_publisher.strip()
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.metric("Toplam net", f"{total_net:.2f}", width="content")
        st.space("stretch")
        if publisher_missing:
            st.caption("Kaydetmek için yayınevi adını gir.")
        st.button(
            "Denemeyi kaydet", type="primary", icon=":material/save:",
            on_click=save_exam, args=(exam_type, subjects), disabled=bool(errors) or publisher_missing,
            help="Yayınevi adı girilmeden kaydedilemez." if publisher_missing else None,
        )
    if st.session_state.entry_error:
        st.error(st.session_state.entry_error, icon=":material/error:")
    if saved := st.session_state.pop("entry_saved", None):
        st.success(f"{saved} denemesi kaydedildi.", icon=":material/check_circle:")

# --- CSV içe aktarma -----------------------------------------------------
with csv_tab:
    st.caption(
        "Sütunlar: `date, publisher, subject, correct, wrong, blank` ve opsiyonel `exam_type` "
        "(TYT, AYT veya YDT; yoksa TYT). Tarih YYYY-AA-GG biçiminde. TYT denemesinde 4 dersin, "
        "AYT/YDT denemesinde alanının tüm testlerinin olması gerekir."
    )
    uploaded = st.file_uploader(
        "CSV dosyası", type="csv", key=f"csv_upload_{st.session_state.csv_upload_version}",
    )
    if st.session_state.pop("import_count", None) is not None:
        st.success("Denemeler içe aktarıldı.", icon=":material/check_circle:")
    if uploaded is not None:
        try:
            frame = read_exam_csv(uploaded, SUBJECTS, field=student_field)
        except ValidationError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            n_exams_in_file = frame.groupby(["date", "publisher", "exam_type"]).ngroups
            st.markdown(f"**{n_exams_in_file} deneme bulundu.** Kontrol edip onayla:")
            for et, part in frame.groupby("exam_type", sort=False):
                st.markdown(f"{et}")
                st.dataframe(exam_net_table(part.drop(columns="exam_type")), hide_index=True, column_config={
                    "date": st.column_config.DateColumn("Tarih", format="DD.MM.YYYY"),
                    "publisher": "Yayınevi",
                })
            st.button(
                "İçe aktar", type="primary", icon=":material/upload:",
                on_click=import_csv, args=(frame,),
            )
    if st.session_state.import_error:
        st.error(st.session_state.import_error, icon=":material/error:")

    st.subheader("Konu bazlı sonuçlar (opsiyonel)")
    st.caption(
        "Sütunlar: `date, publisher, subject, topic, correct, wrong, blank`. "
        "Denemenin ders sonuçları önceden kayıtlı olmalı; konu adları konu listesiyle aynı yazılmalı."
    )
    topic_uploaded = st.file_uploader(
        "Konu CSV dosyası", type="csv", key=f"topic_upload_{st.session_state.topic_upload_version}",
    )
    if (n_topic_rows := st.session_state.pop("topic_import_count", None)) is not None:
        st.success(f"{n_topic_rows} konu sonucu içe aktarıldı.", icon=":material/check_circle:")
    if topic_uploaded is not None:
        try:
            topic_frame = read_topic_csv(topic_uploaded)
        except ValidationError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            n_exams_in_file = topic_frame.groupby(["date", "publisher"]).ngroups
            st.markdown(f"**{n_exams_in_file} deneme için {len(topic_frame)} satır bulundu.**")
            st.dataframe(topic_frame, hide_index=True, column_config={
                "date": st.column_config.DateColumn("Tarih", format="DD.MM.YYYY"),
                "publisher": "Yayınevi", "subject": "Ders", "topic": "Konu",
                "correct": "Doğru", "wrong": "Yanlış", "blank": "Boş",
            })
            st.button(
                "Konu sonuçlarını içe aktar", type="primary", icon=":material/upload:",
                on_click=import_topic_csv, args=(topic_frame,),
            )
    if st.session_state.topic_import_error:
        st.error(st.session_state.topic_import_error, icon=":material/error:")

    with st.expander("Konu listesi", icon=":material/list:"):
        st.dataframe(TOPICS[["subject", "name"]], hide_index=True, column_config={
            "subject": "Ders", "name": "Konu",
        })

# --- Veri yönetimi ------------------------------------------------------
with manage_tab:
    st.subheader("Dışa aktar", icon=":material/download:")
    st.caption("Tüm verini yukarıdaki biçimde indir; yedek almak veya başka bilgisayara taşımak için geri yükleyebilirsin.")
    with closing(db.open_app_db()) as conn:
        export_exams = db.export_exam_frame(conn)
        export_topics = db.export_topic_frame(conn)
    with st.container(horizontal=True):
        st.download_button(
            "Denemeleri indir (CSV)", data=export_exams.to_csv(index=False).encode("utf-8-sig"),
            file_name="optistudy_denemeler.csv", mime="text/csv", icon=":material/download:",
            disabled=export_exams.empty,
        )
        st.download_button(
            "Konu sonuçlarını indir (CSV)", data=export_topics.to_csv(index=False).encode("utf-8-sig"),
            file_name="optistudy_konular.csv", mime="text/csv", icon=":material/download:",
            disabled=export_topics.empty,
        )

    st.subheader("Sıfırdan başla", icon=":material/restart_alt:")
    st.caption(
        "Örnek veriyle denedikten sonra kendi gerçek denemelerine geçmek için. Kayıtlı tüm denemeler "
        "kenara alınır ve boş bir başlangıç yapılır. Hiçbir şey silinmez: mevcut veritabanı proje klasöründe "
        "yedek dosyası olarak kalır. Alanın ve diploma notun korunur."
    )
    if (backup_name := st.session_state.pop("fresh_backup", None)) is not None:
        st.success(
            f"Boş bir başlangıç yapıldı. Eski veriler: {backup_name}" if backup_name else "Zaten boş bir başlangıçtasın.",
            icon=":material/check_circle:",
        )
    st.checkbox("Kayıtlı tüm denemeleri kenara alıp sıfırdan başlamak istiyorum.", key="fresh_confirm")
    st.button(
        "Sıfırdan başla", icon=":material/restart_alt:", on_click=start_fresh,
        disabled=not st.session_state.get("fresh_confirm", False),
    )

# --- Kayıtlı denemeler ---------------------------------------------------
st.subheader("Kayıtlı denemeler", icon=":material/history:")
with closing(db.open_app_db()) as conn:
    results = db.subject_results_frame(conn)

if results.empty:
    st.info("Henüz deneme girilmedi.", icon=":material/info:")
else:
    labels: dict[int, str] = {}
    for et, part in results.groupby("exam_type", sort=False):
        table = exam_net_table(part.drop(columns="exam_type")).sort_values(["date", "exam_id"], ascending=False)
        st.markdown(f"**{et}** :gray[· {len(table)} deneme]")
        st.dataframe(table, hide_index=True, column_config={
            "exam_id": None,
            "date": st.column_config.DateColumn("Tarih", format="DD.MM.YYYY"),
            "publisher": "Yayınevi",
            **{c: st.column_config.NumberColumn(c, format="%.2f") for c in table.columns[3:]},
        })
        labels.update({
            row.exam_id: f"{et} · {row.date:%d.%m.%Y} · {row.publisher} · {row.Toplam:.2f} net"
            for row in table.itertuples(index=False)
        })

    with st.expander("Deneme sil", icon=":material/delete:"):
        st.selectbox("Silinecek deneme", options=list(labels), format_func=labels.get, key="delete_exam_id")
        st.checkbox("Bu denemeyi ve tüm sonuçlarını kalıcı olarak silmek istiyorum.", key="delete_confirm")
        st.button(
            "Sil", icon=":material/delete:", on_click=delete_selected,
            disabled=not st.session_state.get("delete_confirm", False),
        )
