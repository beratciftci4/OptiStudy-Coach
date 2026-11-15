"""AYT/YDT desteği: şema geçişi, alan seçimi, alan puanı ve birleşik tahmin testleri."""

import io
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optistudy import db
from optistudy.analysis import NetForecast, combine_forecasts, predict_combined_exam_day, predict_net
from optistudy.data import (
    FIELD_SUBJECTS, FIELDS, ValidationError, load_ayt_test_stats, load_rank_reference,
    load_score_weights, load_subjects, load_tyt_test_stats, read_exam_csv, required_subjects,
)
from optistudy.scoring import (
    calibrate_tyt_scale, component_score_series, field_score, passes_gates, score_constant,
    score_type_stats, tyt_score_linear,
)

# Aşama 1–5'teki (sadece TYT) şema: geçişi bununla test ediyoruz
OLD_SCHEMA = """
CREATE TABLE subjects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    question_count INTEGER NOT NULL CHECK (question_count > 0));
CREATE TABLE topics (id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subjects(id),
    name TEXT NOT NULL, weight REAL, UNIQUE (subject_id, name));
CREATE TABLE exams (id INTEGER PRIMARY KEY, date TEXT NOT NULL, publisher TEXT NOT NULL,
    exam_type TEXT NOT NULL DEFAULT 'TYT' CHECK (exam_type = 'TYT'), notes TEXT, UNIQUE (date, publisher));
CREATE TABLE subject_results (exam_id INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES subjects(id), correct INTEGER NOT NULL, wrong INTEGER NOT NULL,
    blank INTEGER NOT NULL, PRIMARY KEY (exam_id, subject_id));
CREATE TABLE topic_results (exam_id INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id), correct INTEGER NOT NULL DEFAULT 0,
    wrong INTEGER NOT NULL DEFAULT 0, blank INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (exam_id, topic_id));
CREATE TABLE rank_reference (year INTEGER NOT NULL, score_type TEXT NOT NULL, score REAL NOT NULL,
    rank INTEGER NOT NULL, PRIMARY KEY (year, score_type, score));
CREATE TABLE student_profile (id INTEGER PRIMARY KEY, diploma_grade REAL);
INSERT INTO subjects (id, name, question_count) VALUES (1,'Türkçe',40),(2,'Sosyal',20),(3,'Matematik',40),(4,'Fen',20);
INSERT INTO topics (id, subject_id, name, weight) VALUES (1, 4, 'Fizik', 7);
INSERT INTO exams (id, date, publisher, notes) VALUES (7, '2026-09-06', 'A', 'ilk'), (9, '2026-09-13', 'B', NULL);
INSERT INTO subject_results VALUES (7,1,30,8,2),(7,2,12,4,4),(7,3,20,12,8),(7,4,8,6,6),
                                   (9,1,31,8,1),(9,2,12,4,4),(9,3,21,12,7),(9,4,8,6,6);
INSERT INTO topic_results VALUES (7, 1, 3, 2, 1);
INSERT INTO student_profile VALUES (1, 82.5);
"""

TYT_EXAM = {"Türkçe": (30, 8, 2), "Sosyal": (12, 4, 4), "Matematik": (20, 12, 8), "Fen": (8, 6, 6)}
SAY_EXAM = {"AYT Matematik": (20, 8, 12), "Fizik": (6, 4, 4), "Kimya": (5, 4, 4), "Biyoloji": (7, 2, 4)}


# --- şema geçişi ---------------------------------------------------------

def test_migration_preserves_data_and_makes_backup(tmp_path):
    path = tmp_path / "optistudy.db"
    with sqlite3.connect(path) as old:
        old.executescript(OLD_SCHEMA)

    conn = db.open_app_db(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    # Veriler ve id'ler korundu
    assert conn.execute("SELECT id, date, publisher, exam_type, notes FROM exams ORDER BY id").fetchall() == [
        (7, "2026-09-06", "A", "TYT", "ilk"), (9, "2026-09-13", "B", "TYT", None),
    ]
    assert conn.execute("SELECT COUNT(*) FROM subject_results").fetchone()[0] == 8
    assert db.topic_results_frame(conn)["topic"].tolist() == ["Fizik"]
    assert db.get_diploma_grade(conn) == 82.5
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    # Yeni kısıtlar çalışıyor: aynı gün aynı yayınevinin AYT denemesi kaydedilebilir
    db.set_field(conn, "SAY")
    db.add_exam(conn, "2026-09-06", "A", SAY_EXAM, exam_type="AYT")
    # Silme hâlâ zincirleme çalışıyor (yabancı anahtarlar yeni tabloya bağlı)
    db.delete_exam(conn, 7)
    assert conn.execute("SELECT COUNT(*) FROM subject_results WHERE exam_id = 7").fetchone()[0] == 0
    assert db.topic_results_frame(conn).empty
    conn.close()

    backup = tmp_path / "optistudy-v1-yedek.db"
    assert backup.exists()
    with sqlite3.connect(backup) as b:
        assert b.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 2


def test_migration_runs_once(tmp_path):
    path = tmp_path / "x.db"
    with sqlite3.connect(path) as old:
        old.executescript(OLD_SCHEMA)
    db.open_app_db(path).close()
    (tmp_path / "x-v1-yedek.db").unlink()
    db.open_app_db(path).close()  # ikinci açılışta geçiş yok → yeni yedek de yok
    assert not (tmp_path / "x-v1-yedek.db").exists()


def test_new_database_needs_no_migration(tmp_path):
    db.open_app_db(tmp_path / "new.db").close()
    assert not (tmp_path / "new-v1-yedek.db").exists()


# --- alan ve zorunlu dersler ---------------------------------------------

def test_required_subjects():
    subjects = load_subjects()
    assert required_subjects(subjects, "TYT", None) == ["Türkçe", "Sosyal", "Matematik", "Fen"]
    assert required_subjects(subjects, "AYT", "SAY") == list(FIELD_SUBJECTS["SAY"])
    assert required_subjects(subjects, "YDT", "DİL") == ["Yabancı Dil"]
    with pytest.raises(ValidationError, match="alanını seç"):
        required_subjects(subjects, "AYT", None)
    with pytest.raises(ValidationError, match="YDT"):
        required_subjects(subjects, "AYT", "DİL")


def test_field_subjects_add_up_to_full_tests():
    qc = dict(zip(load_subjects()["name"], load_subjects()["question_count"]))
    assert sum(qc[s] for s in FIELD_SUBJECTS["SAY"]) == 80   # Matematik 40 + Fen 40
    assert sum(qc[s] for s in FIELD_SUBJECTS["EA"]) == 80    # Matematik 40 + TDE-Sos1 40
    assert sum(qc[s] for s in FIELD_SUBJECTS["SÖZ"]) == 80   # TDE-Sos1 40 + Sos2 40
    assert sum(qc[s] for s in FIELD_SUBJECTS["DİL"]) == 80


def test_add_ayt_exam_requires_field(conn):
    with pytest.raises(ValueError, match="alanını seç"):
        db.add_exam(conn, "2026-09-07", "A", SAY_EXAM, exam_type="AYT")
    db.set_field(conn, "SAY")
    exam_id = db.add_exam(conn, "2026-09-07", "A", SAY_EXAM, exam_type="AYT")
    db.add_exam(conn, "2026-09-07", "A", TYT_EXAM)  # aynı gün TYT de olur
    ayt = db.subject_results_frame(conn, "AYT")
    assert set(ayt["subject"]) == set(FIELD_SUBJECTS["SAY"]) and set(ayt["exam_id"]) == {exam_id}
    assert set(db.subject_results_frame(conn, "TYT")["subject"]) == set(TYT_EXAM)
    with pytest.raises(db.DuplicateExamError, match="AYT"):
        db.add_exam(conn, "2026-09-07", "A", SAY_EXAM, exam_type="AYT")


def test_ayt_exam_rejects_other_field_subjects(conn):
    db.set_field(conn, "EA")
    with pytest.raises(ValueError, match="Eksik"):
        db.add_exam(conn, "2026-09-07", "A", SAY_EXAM, exam_type="AYT")


def test_field_roundtrip_and_validation(conn):
    assert db.get_field(conn) is None
    for f in FIELDS:
        db.set_field(conn, f)
        assert db.get_field(conn) == f
    with pytest.raises(ValueError):
        db.set_field(conn, "MF")


def test_read_exam_csv_with_exam_type():
    header = "date,publisher,exam_type,subject,correct,wrong,blank\n"
    rows = "".join(f"2026-09-06,A,TYT,{s},{c},{w},{b}\n" for s, (c, w, b) in TYT_EXAM.items())
    rows += "".join(f"2026-09-06,A,AYT,{s},{c},{w},{b}\n" for s, (c, w, b) in SAY_EXAM.items())
    with pytest.raises(ValidationError, match="alanını seç"):
        read_exam_csv(io.BytesIO((header + rows).encode("utf-8")))
    df = read_exam_csv(io.BytesIO((header + rows).encode("utf-8")), field="SAY")
    assert df.groupby(["date", "publisher", "exam_type"]).ngroups == 2
    with pytest.raises(ValidationError, match="ait olmayan"):
        read_exam_csv(io.BytesIO((header + rows).encode("utf-8")), field="EA")


def test_import_csv_and_topics_for_ayt(conn):
    db.set_field(conn, "SAY")
    header = "date,publisher,exam_type,subject,correct,wrong,blank\n"
    rows = "".join(f"2026-09-06,A,AYT,{s},{c},{w},{b}\n" for s, (c, w, b) in SAY_EXAM.items())
    db.import_exam_frame(conn, read_exam_csv(io.BytesIO((header + rows).encode()), field="SAY"))
    topics = pd.DataFrame([{
        "date": date(2026, 9, 6), "publisher": "A", "subject": "Fizik",
        "topic": "Modern fizik", "correct": 1, "wrong": 1, "blank": 0,
    }])
    assert db.import_topic_frame(conn, topics) == 1
    assert db.topic_results_frame(conn, "AYT")["topic"].tolist() == ["Modern fizik"]


# --- alan puanı ----------------------------------------------------------

def _field_setup(score_type: str):
    ref = load_rank_reference()
    ref = ref[ref["score_type"] == score_type]
    stats = score_type_stats(score_type, load_tyt_test_stats(), load_ayt_test_stats(), load_score_weights(), ref)
    return stats, calibrate_tyt_scale(stats, ref)


@pytest.mark.parametrize("score_type", FIELDS)
def test_field_full_nets_score_500(score_type):
    stats, scale = _field_setup(score_type)
    assert stats["weight"].sum() == pytest.approx(1.0)
    assert stats.loc[stats["session"] == "TYT", "weight"].sum() == pytest.approx(0.40)
    full = dict(zip(stats["subject"], stats["question_count"]))
    assert field_score(full, stats, scale, score_type) == pytest.approx(500.0)


def test_say_net_values_are_plausible():
    stats, scale = _field_setup("SAY")
    s = stats.set_index("subject")
    per_net = {k: scale.slope * s.loc[k, "weight"] * 10 / s.loc[k, "sd_final_year"] for k in s.index}
    # Bir AYT matematik neti yaklaşık 3 puan, bir TYT neti AYT netinden az etkili
    assert 2.5 < per_net["AYT Matematik"] < 3.5
    assert per_net["Türkçe"] < per_net["AYT Matematik"]


def test_population_correction_hand_computed():
    ref = pd.DataFrame({"year": [2026, 2026], "score": [100.0, 500.0], "rank": [1000, 1]})
    tyt = pd.DataFrame({"year": [2026], "subject": ["Türkçe"], "question_count": [40], "weight": [0.33],
                        "mean_final_year": [20.0], "sd_final_year": [8.0], "mean_all": [20.0]})
    ayt = pd.DataFrame({"year": [2026], "subject": ["Fizik"], "valid_question_count": [14],
                        "mean_final_year": [3.0], "sd_final_year": [3.0], "mean_all": [2.0], "n_all": [1500]})
    w = pd.DataFrame({"score_type": ["SAY", "SAY"], "subject": ["Türkçe", "Fizik"], "weight": [0.4, 0.6]})
    out = score_type_stats("SAY", tyt, ayt, w, ref).set_index("subject")
    assert out.loc["Fizik", "mean_all"] == pytest.approx(2.0 * 1500 / 1000)  # nüfus düzeltmesi
    assert out.loc["Türkçe", "mean_all"] == 20.0                           # TYT'ye uygulanmaz
    with pytest.raises(ValueError, match="toplamı"):
        score_type_stats("SAY", tyt, ayt, w.assign(weight=[0.4, 0.5]), ref)


def test_gates():
    assert passes_gates({"Türkçe": 1, "AYT Matematik": 0, "Fizik": 0.25, "Kimya": 0.25}, "SAY")
    assert not passes_gates({"Türkçe": 1, "AYT Matematik": 0.25, "Fizik": 0.0}, "SAY")
    assert not passes_gates({"Türkçe": 0, "Matematik": 0.25, "AYT Matematik": 10}, "SAY")
    assert passes_gates({"Matematik": 1, "Yabancı Dil": 0.5}, "DİL")


def test_components_add_up_to_score():
    stats, scale = _field_setup("SAY")
    nets_tyt = {"Türkçe": 28.0, "Sosyal": 11.0, "Matematik": 17.0, "Fen": 6.5}
    nets_ayt = {"AYT Matematik": 18.0, "Fizik": 5.0, "Kimya": 4.0, "Biyoloji": 6.5}

    def frame(nets):
        return pd.DataFrame([{"exam_id": 1, "date": date(2026, 9, 6), "publisher": "A",
                              "subject": s, "correct": n, "wrong": 0} for s, n in nets.items()])

    total = (score_constant(scale)
             + component_score_series(frame(nets_tyt), stats, scale)["net"].iloc[0]
             + component_score_series(frame(nets_ayt), stats, scale)["net"].iloc[0])
    assert total == pytest.approx(tyt_score_linear({**nets_tyt, **nets_ayt}, stats, scale))


# --- birleşik tahmin -----------------------------------------------------

def test_combine_forecasts_hand_computed():
    d = date(2027, 6, 20)
    a = NetForecast(d, point=10, low=7, high=13, n_exams=8)    # yarı genişlik 3
    b = NetForecast(d, point=20, low=16, high=24, n_exams=6)   # yarı genişlik 4
    c = combine_forecasts([a, b], constant=100)
    assert c.point == 130 and c.low == pytest.approx(125) and c.high == pytest.approx(135)  # √(9+16) = 5
    assert c.n_exams == 6
    clipped = combine_forecasts([a, b], constant=470, bounds=(100, 500))
    assert clipped.high == 500 and clipped.point == 500


def test_predict_combined_requires_both_components():
    start = date(2026, 9, 6)

    def steady(n):
        return pd.DataFrame({
            "exam_id": range(n),
            "date": [start + timedelta(days=7 * i) for i in range(n)],
            "net": [10 + i + (0.5 if i % 2 else -0.5) for i in range(n)],
        })

    good, short = steady(12), steady(5)
    exam = date(2027, 6, 20)
    assert predict_combined_exam_day([(good, exam), (short, exam)], 100) is None
    combined = predict_combined_exam_day([(good, exam), (good, exam)], 100)
    single = predict_net(good, exam)
    # İki aynı bileşen: nokta = sabit + 2·tek tahmin, yarı genişlik = √2·tek yarı genişlik
    assert combined.point == pytest.approx(100 + 2 * single.point)
    assert (combined.high - combined.low) == pytest.approx(np.sqrt(2) * (single.high - single.low))


# --- dışa aktarma --------------------------------------------------------

def test_export_roundtrip(tmp_path):
    from optistudy.data import read_topic_csv
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "data" / "sample"
    src = db.open_app_db(tmp_path / "src.db")
    db.set_field(src, "SAY")
    db.import_exam_frame(src, read_exam_csv(root / "exams.csv"))
    db.import_exam_frame(src, read_exam_csv(root / "ayt_exams.csv", field="SAY"))
    db.import_topic_frame(src, read_topic_csv(root / "topic_results.csv"))
    db.import_topic_frame(src, read_topic_csv(root / "ayt_topic_results.csv"))

    exams_csv, topics_csv = tmp_path / "exams.csv", tmp_path / "topics.csv"
    db.export_exam_frame(src).to_csv(exams_csv, index=False)
    db.export_topic_frame(src).to_csv(topics_csv, index=False)

    dst = db.open_app_db(tmp_path / "dst.db")
    db.set_field(dst, "SAY")
    db.import_exam_frame(dst, read_exam_csv(exams_csv, field="SAY"))
    db.import_topic_frame(dst, read_topic_csv(topics_csv))
    pd.testing.assert_frame_equal(db.export_exam_frame(src), db.export_exam_frame(dst))
    pd.testing.assert_frame_equal(db.export_topic_frame(src), db.export_topic_frame(dst))
    assert db.export_exam_frame(dst)["exam_type"].value_counts().to_dict() == {"TYT": 48, "AYT": 40}
