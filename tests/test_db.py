from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from optistudy import db
from optistudy.data import ValidationError, read_exam_csv
from optistudy.scoring import exam_totals

ROOT = Path(__file__).resolve().parent.parent


def test_schema_has_all_tables(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"subjects", "topics", "exams", "subject_results", "topic_results", "rank_reference"} <= names


def test_subjects_seeded(conn):
    rows = dict(conn.execute("SELECT name, question_count FROM subjects WHERE exam_type = 'TYT'"))
    assert rows == {"Türkçe": 40, "Sosyal": 20, "Matematik": 40, "Fen": 20}
    ayt = dict(conn.execute("SELECT name, question_count FROM subjects WHERE exam_type = 'AYT'"))
    assert sum(ayt.values()) == 160  # 4 test × 40 soru
    assert ayt["Edebiyat"] == 24 and ayt["Fizik"] == 14


def test_add_exam_and_total_net(conn, full_exam):
    db.add_exam(conn, "2026-09-06", "A", full_exam)
    totals = exam_totals(db.subject_results_frame(conn))
    assert totals["total_net"].tolist() == [62.5]


def test_add_exam_rejects_missing_subject(conn, full_exam):
    del full_exam["Fen"]
    with pytest.raises(ValueError, match="Eksik"):
        db.add_exam(conn, "2026-09-06", "A", full_exam)
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 0


def test_add_exam_invalid_counts_writes_nothing(conn, full_exam):
    full_exam["Fen"] = (15, 6, 6)  # 27 > 20
    with pytest.raises(ValidationError):
        db.add_exam(conn, "2026-09-06", "A", full_exam)
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM subject_results").fetchone()[0] == 0


def test_duplicate_exam_rejected(conn, full_exam):
    db.add_exam(conn, "2026-09-06", "A", full_exam)
    with pytest.raises(db.DuplicateExamError, match="zaten kayıtlı"):
        db.add_exam(conn, "2026-09-06", "A", full_exam)
    assert conn.execute("SELECT COUNT(*) FROM subject_results").fetchone()[0] == 4


def test_import_sample_roundtrip(conn):
    frame = read_exam_csv(ROOT / "data" / "sample" / "exams.csv")
    ids = db.import_exam_frame(conn, frame)
    out = db.subject_results_frame(conn)
    assert len(ids) == frame.groupby(["date", "publisher"]).ngroups
    assert len(out) == len(frame)
    assert out["correct"].sum() == frame["correct"].sum()


def _rows(exams: list[tuple[str, str]], results: dict[str, tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "publisher": p, "subject": s, "correct": c, "wrong": w, "blank": b}
        for d, p in exams
        for s, (c, w, b) in results.items()
    ])


def test_import_is_all_or_nothing_on_duplicate(conn, full_exam):
    db.add_exam(conn, "2026-09-13", "B", full_exam)
    frame = _rows([("2026-09-06", "A"), ("2026-09-13", "B")], full_exam)
    with pytest.raises(db.DuplicateExamError):
        db.import_exam_frame(conn, frame)
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM subject_results").fetchone()[0] == 4


def test_import_is_all_or_nothing_on_invalid(conn, full_exam):
    bad = {s: (50, 0, 0) for s in full_exam}
    frame = pd.concat([_rows([("2026-09-06", "A")], full_exam), _rows([("2026-09-13", "B")], bad)])
    with pytest.raises(ValidationError):
        db.import_exam_frame(conn, frame)
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 0


def test_delete_exam_cascades(conn, full_exam):
    keep = db.add_exam(conn, "2026-09-06", "A", full_exam)
    gone = db.add_exam(conn, "2026-09-13", "A", full_exam)
    assert db.delete_exam(conn, gone) is True
    assert conn.execute("SELECT id FROM exams").fetchall() == [(keep,)]
    assert conn.execute("SELECT COUNT(*) FROM subject_results").fetchone()[0] == 4
    assert db.delete_exam(conn, gone) is False


def test_open_app_db_seeds_subjects(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTISTUDY_DB", str(tmp_path / "t.db"))
    c = db.open_app_db()
    assert c.execute("SELECT COUNT(*) FROM subjects WHERE exam_type = 'TYT'").fetchone()[0] == 4
    c.close()
    assert (tmp_path / "t.db").exists()


def test_start_fresh_archives_data_and_keeps_profile(tmp_path, full_exam):
    path = tmp_path / "app.db"
    with closing(db.open_app_db(path)) as c:
        db.add_exam(c, "2026-09-06", "A", full_exam)
        db.set_field(c, "SAY")
        db.set_diploma_grade(c, 82.5)

    backup = db.start_fresh(path, "20261106-210000")

    assert backup == tmp_path / "app-yedek-20261106-210000.db"
    with closing(db.open_app_db(backup)) as c:
        assert len(db.subject_results_frame(c)) == 4  # eski veri yedekte duruyor
    with closing(db.open_app_db(path)) as c:
        assert db.subject_results_frame(c).empty
        assert db.get_field(c) == "SAY"
        assert db.get_diploma_grade(c) == 82.5


def test_start_fresh_without_file_does_nothing(tmp_path):
    assert db.start_fresh(tmp_path / "yok.db", "x") is None
    assert not (tmp_path / "yok.db").exists()


def test_start_fresh_refuses_to_overwrite_backup(tmp_path):
    path = tmp_path / "app.db"
    db.open_app_db(path).close()
    (tmp_path / "app-yedek-x.db").write_bytes(b"")
    with pytest.raises(FileExistsError):
        db.start_fresh(path, "x")
    assert path.exists()
