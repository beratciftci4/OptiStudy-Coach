import io
from pathlib import Path

import pytest

from optistudy.data import (
    ValidationError,
    read_exam_csv,
    validate_counts,
    validate_topic_totals,
)

ROOT = Path(__file__).resolve().parent.parent
HEADER = "date,publisher,subject,correct,wrong,blank\n"
VALID_ROWS = (
    "2026-09-06,A,Türkçe,30,8,2\n"
    "2026-09-06,A,Sosyal,12,4,4\n"
    "2026-09-06,A,Matematik,20,12,8\n"
    "2026-09-06,A,Fen,8,6,6\n"
)


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exam.csv"
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_validate_counts_ok():
    validate_counts(30, 8, 2, 40)
    validate_counts(0, 0, 0, 40)


@pytest.mark.parametrize("c, w, b", [(30, 8, 3), (41, 0, 0), (-1, 0, 0)])
def test_validate_counts_rejects(c, w, b):
    with pytest.raises(ValidationError):
        validate_counts(c, w, b, 40)


def test_topic_totals_cannot_exceed_subject():
    subject = {"correct": 20, "wrong": 12, "blank": 8}
    validate_topic_totals(subject, [{"wrong": 5}, {"wrong": 7}])
    with pytest.raises(ValidationError):
        validate_topic_totals(subject, [{"wrong": 8}, {"wrong": 5}])


def test_read_valid_csv(tmp_path):
    df = read_exam_csv(write(tmp_path, VALID_ROWS))
    assert len(df) == 4
    assert set(df["subject"]) == {"Türkçe", "Sosyal", "Matematik", "Fen"}


def test_read_rejects_missing_subject(tmp_path):
    body = "\n".join(VALID_ROWS.strip().split("\n")[:3]) + "\n"
    with pytest.raises(ValidationError, match="eksik ders"):
        read_exam_csv(write(tmp_path, body))


def test_read_rejects_too_many_answers(tmp_path):
    body = VALID_ROWS.replace("Fen,8,6,6", "Fen,10,6,6")
    with pytest.raises(ValidationError, match="20 soru"):
        read_exam_csv(write(tmp_path, body))


def test_read_rejects_unknown_subject(tmp_path):
    body = VALID_ROWS.replace("Fen,", "Astronomi,")
    with pytest.raises(ValidationError, match="Bilinmeyen"):
        read_exam_csv(write(tmp_path, body))


def test_read_rejects_bad_date(tmp_path):
    body = VALID_ROWS.replace("2026-09-06", "06.09.2026")
    with pytest.raises(ValidationError, match="YYYY-AA-GG"):
        read_exam_csv(write(tmp_path, body))


def test_read_accepts_uploaded_bytes_with_bom():
    # Arayüzden yüklenen dosya gibi: bellek içi, Excel BOM'lu
    buffer = io.BytesIO(("﻿" + HEADER + VALID_ROWS).encode("utf-8"))
    df = read_exam_csv(buffer)
    assert len(df) == 4
    assert "date" in df.columns


def test_read_rejects_empty_file():
    with pytest.raises(ValidationError, match="okunamadı"):
        read_exam_csv(io.BytesIO(b""))


def test_sample_data_is_valid():
    df = read_exam_csv(ROOT / "data" / "sample" / "exams.csv")
    assert df["date"].nunique() >= 10
