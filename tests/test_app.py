"""Arayüz duman testleri: sayfalar hatasız açılıyor ve temel akışlar çalışıyor mu."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from optistudy import db
from optistudy.data import read_exam_csv, read_topic_csv

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
ENTRY = "pages/1_Deneme_Girisi.py"
PANEL = "pages/2_Gelisim_Paneli.py"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("OPTISTUDY_DB", str(path))
    return path


@pytest.fixture
def sample_db(db_path):
    conn = db.open_app_db(db_path)
    db.import_exam_frame(conn, read_exam_csv(ROOT / "data" / "sample" / "exams.csv"))
    db.import_topic_frame(conn, read_topic_csv(ROOT / "data" / "sample" / "topic_results.csv"))
    conn.close()
    return db_path


def open_page(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


def test_home_empty(db_path):
    at = open_page()
    assert next(m.value for m in at.metric if m.label == "TYT denemesi") == "0"
    assert any(m.label == "TYT sınavına" and m.value.endswith(" gün") for m in at.metric)


def test_panel_empty_asks_for_more_exams(db_path):
    at = open_page(PANEL)
    assert "daha fazla deneme gir" in at.info[0].value


def test_panel_with_sample_data(sample_db):
    at = open_page(PANEL)
    assert len(at.metric) == 4
    at.segmented_control(key="panel_subject").set_value("Matematik").run()
    assert not at.exception


def test_entry_save_and_delete(db_path):
    at = open_page(ENTRY)
    save = at.button[0]
    assert save.disabled  # yayınevi boş

    at.text_input(key="entry_publisher").input("Yayınevi X").run()
    at.number_input(key="entry_Türkçe_correct").set_value(30).run()
    at.number_input(key="entry_Türkçe_wrong").set_value(8).run()
    assert not at.button[0].disabled
    at.button[0].click().run()
    assert not at.exception
    assert at.success[0].value == "TYT denemesi kaydedildi."
    assert at.number_input(key="entry_Türkçe_correct").value == 0  # form sıfırlandı

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 1

    at.checkbox(key="delete_confirm").check().run()
    delete_button = next(b for b in at.button if b.label == "Sil")
    delete_button.click().run()
    assert not at.exception
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 0
    conn.close()


def test_entry_blocks_invalid_counts(db_path):
    at = open_page(ENTRY)
    at.text_input(key="entry_publisher").input("Yayınevi X").run()
    at.number_input(key="entry_Fen_correct").set_value(15).run()
    at.number_input(key="entry_Fen_wrong").set_value(10).run()
    assert at.error and "20 soru" in at.error[0].value
    assert at.button[0].disabled


FORECAST = "pages/3_Net_Tahmini.py"
RANK = "pages/5_Siralama_Tahmini.py"


def test_forecast_page_needs_five_exams(db_path):
    at = open_page(FORECAST)
    assert "en az 5 TYT denemesi" in at.info[0].value


def test_forecast_page_with_sample_data(sample_db):
    at = open_page(FORECAST)
    assert any("Düşük güven" in m.value for m in at.markdown)
    labels = [m.label for m in at.metric]
    assert "Tahmini net" in labels and "Tutan tahmin" in labels
    # Örnek veride toplam için kapsama yetersiz → sınav günü kilitli
    assert any("henüz kapalı" in i.proto.title for i in at.info)
    assert not any(label.startswith("TYT 2027") for label in labels)

    # Fen'de 7 tahminin 7'si tutuyor → sınav günü tahmini açık
    at.segmented_control(key="forecast_subject").set_value("Fen").run()
    assert not at.exception
    assert any(m.label.startswith("TYT 2027") for m in at.metric)


def test_forecast_page_target_date_change(sample_db):
    at = open_page(FORECAST)
    before = next(m.value for m in at.metric if m.label == "%80 tahmin aralığı")
    at.date_input(key="forecast_target_TYT").set_value(date(2027, 3, 1)).run()
    assert not at.exception
    after = next(m.value for m in at.metric if m.label == "%80 tahmin aralığı")
    assert before != after


@pytest.fixture
def steady_db(db_path):
    """Düzenli artan sahte veri: Matematik her hafta +0,25 net, Türkçede ±1 dönüşümlü sapma."""
    conn = db.open_app_db(db_path)
    for i in range(12):
        bump = 1 if i % 2 else -1
        db.add_exam(conn, f"2026-{9 + (i * 7) // 30:02d}-{1 + (i * 7) % 30:02d}", f"Y{i}", {
            "Türkçe": (25 + bump, 4, 0),
            "Sosyal": (10, 2, 0),
            "Matematik": (10, 12 - i, 0),
            "Fen": (6, 2, 0),
        })
    conn.close()
    return db_path


def test_ranking_locked_on_sample_data(sample_db):
    at = open_page(RANK)
    assert any(i.proto.title == "Henüz kapalı" for i in at.info)


def test_ranking_open_and_diploma_grade_saved(steady_db):
    at = open_page(RANK)
    labels = [m.label for m in at.metric]
    assert "Başarı sırası (sınav puanı)" in labels
    assert "Başarı sırası (yerleştirme puanı)" not in labels
    assert any("ön lisans" in w.value for w in at.warning)

    at.number_input(key="diploma_grade").set_value(85.0).run()
    assert not at.exception
    assert "Başarı sırası (yerleştirme puanı)" in [m.label for m in at.metric]
    conn = db.connect(steady_db)
    assert db.get_diploma_grade(conn) == 85.0
    conn.close()

    rank = next(m.value for m in at.metric if m.label == "Başarı sırası (sınav puanı)")
    best, worst = (int(x.replace(".", "")) for x in rank.split(" – "))
    assert 1 <= best < worst


PRIORITY = "pages/4_Konu_Onceligi.py"


def test_priority_page_without_topic_data(db_path):
    at = open_page(PRIORITY)
    assert "konu bazlı sonuç" in at.info[0].value


def test_priority_page_with_sample_data(sample_db):
    at = open_page(PRIORITY)
    lost = [m for m in at.metric if m.label == "Beklenen kayıp"]
    assert len(lost) == 5
    values = [float(m.value.split()[0]) for m in lost]
    assert values == sorted(values, reverse=True)
    assert any("Problemler" in m.value for m in at.markdown)  # örnek veride en zayıf konu


def test_entry_saves_topic_results(db_path):
    at = open_page(ENTRY)
    at.text_input(key="entry_publisher").input("Yayınevi X").run()
    at.number_input(key="entry_Fen_correct").set_value(8).run()
    at.number_input(key="entry_Fen_wrong").set_value(6).run()
    at.number_input(key="topic_Fen_Fizik_correct").set_value(3).run()
    at.number_input(key="topic_Fen_Fizik_wrong").set_value(4).run()
    assert not at.button[0].disabled
    at.button[0].click().run()
    assert not at.exception
    assert at.number_input(key="topic_Fen_Fizik_wrong").value == 0  # form sıfırlandı
    conn = db.connect(db_path)
    tr = db.topic_results_frame(conn)
    conn.close()
    assert tr[["topic", "correct", "wrong", "blank"]].values.tolist() == [["Fizik", 3, 4, 0]]


def test_entry_blocks_topic_totals_above_subject(db_path):
    at = open_page(ENTRY)
    at.text_input(key="entry_publisher").input("Yayınevi X").run()
    at.number_input(key="entry_Fen_wrong").set_value(2).run()
    at.number_input(key="topic_Fen_Fizik_wrong").set_value(3).run()
    assert any("yanlış toplamı" in e.value for e in at.error)
    assert at.button[0].disabled


def test_forecast_page_same_date_exams_does_not_crash(db_path):
    # Eski denemeler aynı tarihle topluca girilmiş olabilir: trend kurulamaz
    conn = db.open_app_db(db_path)
    for i in range(5):
        db.add_exam(conn, "2026-09-06", f"Y{i}", {
            "Türkçe": (20 + i, 4, 0), "Sosyal": (10, 2, 0), "Matematik": (10, 4, 0), "Fen": (6, 2, 0),
        })
    conn.close()
    at = open_page(FORECAST)
    assert any("iki farklı tarihte" in i.value for i in at.info)


# --- AYT / alan ----------------------------------------------------------

def test_home_field_selection_is_saved(db_path):
    at = open_page()
    at.segmented_control(key="home_field").set_value("EA").run()
    assert not at.exception
    conn = db.connect(db_path)
    assert db.get_field(conn) == "EA"
    conn.close()
    assert [m.label for m in at.metric if m.label.endswith("denemesi")] == ["TYT denemesi", "AYT denemesi"]


def test_entry_ayt_exam_for_selected_field(db_path):
    conn = db.open_app_db(db_path)
    db.set_field(conn, "SAY")
    conn.close()
    at = open_page(ENTRY)
    at.segmented_control(key="entry_exam_type").set_value("AYT").run()
    labels = [m.value for m in at.markdown]
    assert any("AYT Matematik" in v for v in labels) and not any("**Türkçe**" in v for v in labels)
    at.text_input(key="entry_publisher").input("Yayınevi X").run()
    at.number_input(key="entry_Fizik_correct").set_value(6).run()
    at.number_input(key="entry_Fizik_wrong").set_value(4).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.success[0].value == "AYT denemesi kaydedildi."
    conn = db.connect(db_path)
    ayt = db.subject_results_frame(conn, "AYT")
    conn.close()
    assert sorted(ayt["subject"]) == sorted(["AYT Matematik", "Fizik", "Kimya", "Biyoloji"])


def test_entry_without_field_offers_only_tyt(db_path):
    at = open_page(ENTRY)
    assert at.segmented_control(key="entry_exam_type").options == ["TYT"]


@pytest.fixture
def steady_say_db(steady_db):
    """steady_db + SAY alanı ve düzenli artan AYT denemeleri (TYT'den bir gün sonra)."""
    conn = db.open_app_db(steady_db)
    db.set_field(conn, "SAY")
    tyt_dates = [r[0] for r in conn.execute("SELECT date FROM exams ORDER BY date")]
    for i, d in enumerate(tyt_dates):
        bump = 1 if i % 2 else -1
        next_day = (pd.Timestamp(d) + pd.Timedelta(days=1)).date().isoformat()
        db.add_exam(conn, next_day, f"Y{i}", {
            "AYT Matematik": (12 + bump, 12 - i, 0), "Fizik": (6, 2, 0), "Kimya": (5, 2, 0), "Biyoloji": (6, 2, 0),
        }, exam_type="AYT")
    conn.close()
    return steady_db


def test_rank_page_shows_tyt_and_field_separately(steady_say_db):
    at = open_page(RANK)
    subheaders = [h.value for h in at.subheader]
    assert subheaders == ["TYT", "Sayısal (SAY)"]
    ranks = [m for m in at.metric if m.label == "Başarı sırası (sınav puanı)"]
    assert len(ranks) == 2  # iki ayrı sıralama, iki ayrı kart
    for m in ranks:
        best, worst = (int(x.replace(".", "")) for x in m.value.split(" – "))
        assert 1 <= best < worst


def test_rank_page_without_field(steady_db):
    at = open_page(RANK)
    assert any("alanını seç" in i.value for i in at.info)


def test_home_weekly_summary(sample_db):
    at = open_page()
    labels = [m.label for m in at.metric if not m.label.endswith("sınavına")]
    assert labels[:2] == ["TYT denemesi", "Son net"]
    focus = next(m.value for m in at.markdown if m.value.startswith("**Bu hafta odaklan:**"))
    assert focus.count("soru kaybı") == 3 and "1. Problemler" in focus


@pytest.mark.parametrize("field", ["SAY", "EA", "SÖZ", "DİL"])
def test_every_field_end_to_end(steady_db, field):
    """Her alan için: alan denemeleri girilir, tüm sayfalar açılır, iki ayrı sıralama kartı çıkar."""
    from optistudy.data import FIELD_NAMES, FIELD_SESSION, FIELD_SUBJECTS, load_subjects
    qc = dict(zip(load_subjects()["name"], load_subjects()["question_count"]))
    conn = db.open_app_db(steady_db)
    db.set_field(conn, field)
    session = FIELD_SESSION[field]
    main, *rest = FIELD_SUBJECTS[field]
    tyt_dates = [r[0] for r in conn.execute("SELECT date FROM exams ORDER BY date")]
    for i, d in enumerate(tyt_dates):
        bump = 1 if i % 2 else -1
        next_day = (pd.Timestamp(d) + pd.Timedelta(days=1)).date().isoformat()
        results = {main: (qc[main] // 3 + bump, 12 - i, 0)}  # ana testte düzenli artış
        results.update({s: (qc[s] // 3, 1, 0) for s in rest})
        db.add_exam(conn, next_day, f"Y{i}", results, exam_type=session)
    conn.close()

    at = open_page()
    for page, key in [(ENTRY, "entry_exam_type"), ("pages/2_Gelisim_Paneli.py", "panel_exam_type"),
                      (FORECAST, "forecast_exam_type"), (PRIORITY, "priority_exam_type")]:
        at.switch_page(page).run()
        at.segmented_control(key=key).set_value(session).run()
        assert not at.exception, (page, at.exception)
    at.switch_page(RANK).run()
    assert not at.exception
    assert [h.value for h in at.subheader] == ["TYT", f"{FIELD_NAMES[field]} ({field})"]
    assert len([m for m in at.metric if m.label == "Başarı sırası (sınav puanı)"]) == 2


def test_entry_start_fresh_keeps_backup(sample_db):
    at = open_page(ENTRY)
    button = next(b for b in at.button if b.label == "Sıfırdan başla")
    assert button.disabled
    at.checkbox(key="fresh_confirm").check().run()
    next(b for b in at.button if b.label == "Sıfırdan başla").click().run()
    assert not at.exception
    assert any("Boş bir başlangıç" in s.value for s in at.success)
    assert len(list(sample_db.parent.glob("app-yedek-*.db"))) == 1
    conn = db.connect(sample_db)
    assert db.subject_results_frame(conn).empty
    conn.close()


def test_home_goal_set_progress_and_clear(sample_db):
    at = open_page()
    assert any("Hedef net belirlersen" in c.value for c in at.caption)
    at.number_input(key="goal_input_TYT").set_value(90.0).run()
    at.button(key="goal_save_TYT").click().run()
    assert not at.exception
    conn = db.connect(sample_db)
    assert db.get_goal(conn, "TYT") == 90.0
    conn.close()
    # Örnek veride 12 TYT denemesi var → hız karşılaştırması gösterilir
    assert any("Gereken hız" in c.value for c in at.caption)
    assert any("Hedefine" in c.value and "soru kaybettiriyor" in c.value for c in at.caption)

    at.button(key="goal_clear_TYT").click().run()
    assert not at.exception
    conn = db.connect(sample_db)
    assert db.get_goal(conn, "TYT") is None
    conn.close()


def test_goal_shown_on_priority_page(sample_db):
    conn = db.connect(sample_db)
    db.set_goal(conn, "TYT", 100)
    conn.close()
    at = open_page(PRIORITY)
    assert any(i.proto.title.startswith("Hedefine") for i in at.info)


def test_goal_line_on_forecast_page(sample_db):
    conn = db.connect(sample_db)
    db.set_goal(conn, "TYT", 100)
    conn.close()
    at = open_page(FORECAST)
    assert not at.exception
