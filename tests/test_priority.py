"""Konu önceliği (Bayesçi düzeltilmiş skor), konu kaydı ve konu CSV testleri."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optistudy import db
from optistudy.analysis import (
    WEIGHT_OBSERVED, WEIGHT_OFFICIAL, subject_error_rates, topic_priority, weekly_focus,
)
from optistudy.charts import priority_figure
from optistudy.data import ValidationError, load_topics, read_exam_csv, read_topic_csv

ROOT = Path(__file__).resolve().parent.parent

TOPICS = pd.DataFrame({
    "subject": ["Fen", "Fen", "Matematik"],
    "topic": ["Fizik", "Kimya", "Problemler"],
    "weight": [7.0, 7.0, np.nan],          # Matematik: ağırlık yayımlanmıyor
})
# Ders sonuçları → genel hata oranı: Fen (10+10)/(20+20+...) aşağıda
SUBJECT_RESULTS = pd.DataFrame({
    "exam_id": [1, 2, 1, 2],
    "subject": ["Fen", "Fen", "Matematik", "Matematik"],
    "correct": [12, 12, 20, 20],
    "wrong": [6, 6, 10, 10],
    "blank": [2, 2, 10, 10],
})  # Fen: 16/40 = 0.4 ; Matematik: 40/80 = 0.5


def topic_rows(rows: list[tuple[int, str, str, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["exam_id", "subject", "topic", "correct", "wrong", "blank"])


def test_subject_error_rates():
    assert subject_error_rates(SUBJECT_RESULTS) == {"Fen": pytest.approx(0.4), "Matematik": pytest.approx(0.5)}


def test_priority_hand_computed():
    tr = topic_rows([
        (1, "Fen", "Fizik", 2, 3, 0), (2, "Fen", "Fizik", 2, 2, 1),   # N=10, hata=6
        (1, "Matematik", "Problemler", 4, 2, 0), (2, "Matematik", "Problemler", 5, 1, 0),  # N=12, hata=3
    ])
    p = topic_priority(tr, TOPICS, SUBJECT_RESULTS, prior_strength=10).set_index("topic")
    # Fizik: α = 10·0.4 = 4, β = 6 → (6 + 4) / (10 + 10) = 0.5 ; P = 7 · 0.5 = 3.5
    assert p.loc["Fizik", "adjusted_error_rate"] == pytest.approx(0.5)
    assert p.loc["Fizik", "priority"] == pytest.approx(3.5)
    assert p.loc["Fizik", "weight_source"] == WEIGHT_OFFICIAL
    # Problemler: ağırlık = 12 soru / 2 deneme = 6 ; α = β = 5 → (3 + 5) / (12 + 10) = 8/22
    assert p.loc["Problemler", "weight"] == pytest.approx(6.0)
    assert p.loc["Problemler", "weight_source"] == WEIGHT_OBSERVED
    assert p.loc["Problemler", "priority"] == pytest.approx(6 * 8 / 22)
    assert p.loc["Problemler", "raw_error_rate"] == pytest.approx(0.25)


def test_topic_without_data_uses_prior_if_weight_known():
    # Kimya hiç girilmedi: ağırlık resmi (7), oran = önsel ortalaması 0.4 → P = 2.8
    tr = topic_rows([(1, "Fen", "Fizik", 2, 3, 0)])
    p = topic_priority(tr, TOPICS, SUBJECT_RESULTS).set_index("topic")
    assert p.loc["Kimya", "n_questions"] == 0
    assert np.isnan(p.loc["Kimya", "raw_error_rate"])
    assert p.loc["Kimya", "priority"] == pytest.approx(7 * 0.4)
    # Problemler: ağırlık bilinmiyor ve veri yok → öncelik hesaplanamaz, listenin sonunda
    assert np.isnan(p.loc["Problemler", "priority"])
    assert topic_priority(tr, TOPICS, SUBJECT_RESULTS)["topic"].iloc[-1] == "Problemler"


def test_shrinkage_tames_small_samples():
    # Kimya'da 1 soru 1 hata (ham %100), Fizik'te 30 soru 15 hata (ham %50)
    tr = topic_rows([(1, "Fen", "Kimya", 0, 1, 0), (1, "Fen", "Fizik", 15, 15, 0)])
    p = topic_priority(tr, TOPICS, SUBJECT_RESULTS).set_index("topic")
    assert p.loc["Kimya", "raw_error_rate"] == 1.0
    # Kimya: (1 + 4) / (1 + 10) ≈ 0.45 ; Fizik: (15 + 4) / (30 + 10) = 0.475 → Fizik önde
    assert p.loc["Kimya", "adjusted_error_rate"] == pytest.approx(5 / 11)
    assert p.loc["Fizik", "priority"] > p.loc["Kimya", "priority"]


def test_priority_sorted_and_none_without_data():
    assert topic_priority(topic_rows([]), TOPICS, SUBJECT_RESULTS) is None
    tr = topic_rows([(1, "Fen", "Fizik", 0, 5, 0), (1, "Matematik", "Problemler", 10, 0, 0)])
    p = topic_priority(tr, TOPICS, SUBJECT_RESULTS)
    assert p["priority"].dropna().is_monotonic_decreasing


def test_weekly_focus_top_n_skips_unknown():
    tr = topic_rows([(1, "Fen", "Fizik", 2, 3, 0)])
    focus = weekly_focus(topic_priority(tr, TOPICS, SUBJECT_RESULTS), n=5)
    assert list(focus["topic"]) == ["Fizik", "Kimya"]  # Problemler hesaplanamıyor
    assert len(weekly_focus(topic_priority(tr, TOPICS, SUBJECT_RESULTS), n=1)) == 1


def test_priority_figure():
    tr = topic_rows([(1, "Fen", "Fizik", 2, 3, 0)])
    focus = weekly_focus(topic_priority(tr, TOPICS, SUBJECT_RESULTS))
    fig = priority_figure(focus)
    assert list(fig.data[0].y) == ["Fen · Kimya", "Fen · Fizik"]  # en öncelikli en üstte
    assert priority_figure(focus.iloc[0:0]) is None


# --- konu listesi ve kayıt -----------------------------------------------

def test_reference_topics_official_weights():
    t = load_topics().set_index(["subject", "name"])["weight"]
    assert t.loc[("Fen", "Fizik")] == 7 and t.loc[("Fen", "Biyoloji")] == 6
    assert t.loc["Sosyal"].sum() == 20 and t.loc["Fen"].sum() == 20
    assert t.loc["Türkçe"].isna().all() and t.loc["Matematik"].isna().all()  # bilerek boş


def test_add_exam_with_topics(conn, full_exam):
    topics = {("Fen", "Fizik"): (3, 2, 1), ("Fen", "Kimya"): (0, 0, 0)}
    exam_id = db.add_exam(conn, "2026-09-06", "A", full_exam, topic_results=topics)
    tr = db.topic_results_frame(conn)
    assert tr[["exam_id", "topic", "correct", "wrong", "blank"]].values.tolist() == [[exam_id, "Fizik", 3, 2, 1]]
    db.delete_exam(conn, exam_id)
    assert db.topic_results_frame(conn).empty


def test_topic_totals_cannot_exceed_subject(conn, full_exam):
    # Fen: (8, 6, 6) ; konu yanlışları 4 + 3 = 7 > 6
    topics = {("Fen", "Fizik"): (0, 4, 0), ("Fen", "Kimya"): (0, 3, 0)}
    with pytest.raises(ValidationError, match="yanlış toplamı"):
        db.add_exam(conn, "2026-09-06", "A", full_exam, topic_results=topics)
    assert conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0] == 0


def test_unknown_topic_rejected(conn, full_exam):
    with pytest.raises(ValueError, match="Bilinmeyen konu"):
        db.add_exam(conn, "2026-09-06", "A", full_exam, topic_results={("Fen", "Astronomi"): (1, 0, 0)})


def test_topics_seeded_with_null_weights(conn):
    db.seed_topics(conn, load_topics())
    t = db.topics_frame(conn, "TYT")
    assert len(t) == 18
    assert t.loc[t["topic"] == "Paragraf", "weight"].isna().all()


# --- konu CSV ------------------------------------------------------------

HEADER = "date,publisher,subject,topic,correct,wrong,blank\n"


def test_read_topic_csv_and_import(conn, full_exam):
    db.seed_topics(conn, load_topics())
    db.add_exam(conn, "2026-09-06", "A", full_exam)
    body = "2026-09-06,A,Fen,Fizik,3,2,1\n2026-09-06,A,Fen,Kimya,2,1,0\n"
    frame = read_topic_csv(io.BytesIO((HEADER + body).encode("utf-8")))
    assert db.import_topic_frame(conn, frame) == 2
    with pytest.raises(db.DuplicateExamError):
        db.import_topic_frame(conn, frame)


def test_import_topics_requires_existing_exam(conn):
    db.seed_topics(conn, load_topics())
    frame = read_topic_csv(io.BytesIO((HEADER + "2026-09-06,X,Fen,Fizik,1,0,0\n").encode()))
    with pytest.raises(ValueError, match="kayıtlı değil"):
        db.import_topic_frame(conn, frame)


@pytest.mark.parametrize("body, match", [
    ("2026-09-06,A,Fen,Astronomi,1,0,0\n", "Bilinmeyen konu"),
    ("2026-09-06,A,Fen,Fizik,-1,0,0\n", "negatif"),
    ("2026-09-06,A,Fen,Fizik,1,0,0\n2026-09-06,A,Fen,Fizik,1,0,0\n", "birden fazla"),
])
def test_read_topic_csv_rejects(body, match):
    with pytest.raises(ValidationError, match=match):
        read_topic_csv(io.BytesIO((HEADER + body).encode("utf-8")))


def test_sample_topic_results_match_subject_totals(conn):
    db.seed_topics(conn, load_topics())
    db.import_exam_frame(conn, read_exam_csv(ROOT / "data" / "sample" / "exams.csv"))
    n = db.import_topic_frame(conn, read_topic_csv(ROOT / "data" / "sample" / "topic_results.csv"))
    assert n > 0
    tr = db.topic_results_frame(conn).groupby(["exam_id", "subject"])[["correct", "wrong", "blank"]].sum()
    sr = db.subject_results_frame(conn).set_index(["exam_id", "subject"])[["correct", "wrong", "blank"]]
    pd.testing.assert_frame_equal(tr, sr.loc[tr.index], check_dtype=False)
