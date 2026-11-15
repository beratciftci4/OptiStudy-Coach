"""TYT puanı dönüşümü, yerleştirme puanı ve sıralama aralığı testleri."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optistudy import db
from optistudy.analysis import rank_range, score_to_rank
from optistudy.data import load_rank_reference, load_tyt_test_stats
from optistudy.scoring import (
    TytScale, calibrate_tyt_scale, distribution_mean, exam_score_series,
    placement_score, tyt_score, tyt_score_linear, weighted_standard_score,
)

# Elle hesaplanabilir iki testli küçük örnek
STATS = pd.DataFrame({
    "year": [2026, 2026],
    "subject": ["Türkçe", "Matematik"],
    "question_count": [40, 40],
    "weight": [0.5, 0.5],
    "mean_final_year": [20.0, 10.0],
    "sd_final_year": [10.0, 5.0],
    "mean_all": [20.0, 10.0],
})
# "X puan ve üstü aday sayısı": 500'de 10, 300 ve üstü 60, 100 ve üstü 100 aday
REF = pd.DataFrame({"year": [2026] * 3, "score": [500.0, 300.0, 100.0], "rank": [10, 60, 100]})


# --- standart puan ve ölçek ----------------------------------------------

def test_weighted_standard_score_hand_computed():
    # Türkçe 30 → SP = 50 + 10·(30−20)/10 = 60 ; Matematik 20 → SP = 50 + 10·(20−10)/5 = 70
    assert weighted_standard_score({"Türkçe": 30, "Matematik": 20}, STATS) == pytest.approx(65.0)


def test_distribution_mean_hand_computed():
    # 500: 10 aday (500'de), 300–500: 50 aday (400'de), 100–300: 40 aday (200'de)
    # (10·500 + 50·400 + 40·200) / 100 = 330
    assert distribution_mean(REF) == pytest.approx(330.0)


def test_calibration_hits_both_anchor_points():
    scale = calibrate_tyt_scale(STATS, REF)
    full = {"Türkçe": 40, "Matematik": 40}
    mean = {"Türkçe": 20, "Matematik": 10}
    assert tyt_score_linear(full, STATS, scale) == pytest.approx(500.0)
    assert tyt_score_linear(mean, STATS, scale) == pytest.approx(330.0)
    # A_full = 0.5·70 + 0.5·110 = 90 ; A_mean = 50 → eğim = (500−330)/40 = 4.25
    assert scale.slope == pytest.approx(4.25)


def test_tyt_score_requires_half_net_in_turkish_or_math():
    scale = TytScale(2026, 0.0, 5.0)
    assert tyt_score({"Türkçe": 0.25, "Matematik": 0.0}, STATS, scale) is None
    assert tyt_score({"Türkçe": 0.5, "Matematik": 0.0}, STATS, scale) is not None


def test_tyt_score_clipped_to_100_500():
    assert tyt_score({"Türkçe": 1, "Matematik": 0}, STATS, TytScale(2026, -1000, 1)) == 100
    assert tyt_score({"Türkçe": 40, "Matematik": 40}, STATS, TytScale(2026, 1000, 1)) == 500


def test_placement_score():
    # 300 + 0,12 × (80 × 5) = 300 + 48
    assert placement_score(300, 80) == pytest.approx(348.0)


def test_real_2026_calibration_is_sane():
    stats = load_tyt_test_stats()
    ref = load_rank_reference()
    scale = calibrate_tyt_scale(stats, ref[ref["score_type"] == "TYT"])
    full = {"Türkçe": 40, "Sosyal": 20, "Matematik": 40, "Fen": 20}
    assert tyt_score(full, stats, scale) == pytest.approx(500.0)
    # Bir net Türkçe ≈ eğim · 0,33 · 10 / 8,4 → yaklaşık 3,2 puan
    per_net = scale.slope * 0.33 * 10 / 8.4
    assert 2.5 < per_net < 4.0
    # Puanı hesaplanan en düşük aday 100 alır; ağırlıklı puanı ortalamanın çok altında olmalı
    a_at_100 = (100 - scale.intercept) / scale.slope
    assert 20 < a_at_100 < 40


def test_exam_score_series_matches_per_exam_score():
    scale = TytScale(2026, 0.0, 5.0)
    results = pd.DataFrame({
        "exam_id": [1, 1, 2, 2],
        "date": [date(2026, 9, 6)] * 2 + [date(2026, 9, 13)] * 2,
        "publisher": ["A"] * 4,
        "subject": ["Türkçe", "Matematik"] * 2,
        "correct": [30, 20, 32, 24],
        "wrong": [0, 0, 8, 0],
    })
    series = exam_score_series(results, STATS, scale)
    assert list(series.columns) == ["exam_id", "date", "publisher", "net"]
    # Deneme 1: A = 65 → 325 ; Deneme 2: Türkçe 30, Matematik 24 → A = 0.5·60 + 0.5·78 = 69 → 345
    assert list(series["net"]) == pytest.approx([325.0, 345.0])


# --- sıralama ------------------------------------------------------------

def test_score_to_rank_interpolates():
    # 300 → 60 ; 400 → 60 ile 10 arasının ortası = 35 ; 200 → 80
    assert score_to_rank(300, REF) == 60
    assert score_to_rank(400, REF) == pytest.approx(35)
    assert score_to_rank(200, REF) == pytest.approx(80)


def test_score_to_rank_outside_table():
    assert score_to_rank(50, REF) == 100
    assert score_to_rank(520, REF) == 10
    one = pd.DataFrame({"year": [2026, 2026], "score": [100.0, 500.0], "rank": [100, 0]})
    assert score_to_rank(500, one) == 1.0  # sıra en az 1


def test_rank_range_returns_interval_not_single_number():
    est = rank_range(200, 400, REF, "TYT")
    assert (est.rank_best, est.rank_worst) == (35, 80)
    assert est.year == 2026 and est.score_type == "TYT"


def test_rank_range_empty_reference():
    assert rank_range(200, 400, REF.iloc[0:0], "TYT") is None


def test_real_reference_ranks_monotonic():
    ref = load_rank_reference()
    tyt = ref[ref["score_type"] == "TYT"]
    ranks = [score_to_rank(s, tyt) for s in np.arange(100, 501, 10)]
    assert all(a >= b for a, b in zip(ranks, ranks[1:]))
    # Resmi tablodan doğrudan: 300 ve üstü 417.935 aday
    assert score_to_rank(300, tyt) == 417935


# --- veritabanı ----------------------------------------------------------

def test_rank_reference_loaded_into_db(tmp_path):
    conn = db.open_app_db(tmp_path / "t.db")
    tyt = db.rank_reference_frame(conn, "TYT")
    assert tyt["year"].unique().tolist() == [2026]
    assert tyt["score"].is_monotonic_increasing
    assert len(db.rank_reference_frame(conn, "Y-TYT")) == 23
    # Tekrar açınca çift kayıt oluşmaz
    conn.close()
    conn = db.open_app_db(tmp_path / "t.db")
    assert len(db.rank_reference_frame(conn, "TYT")) == 21
    conn.close()


def test_diploma_grade_roundtrip(conn):
    assert db.get_diploma_grade(conn) is None
    db.set_diploma_grade(conn, 82.5)
    assert db.get_diploma_grade(conn) == 82.5
    db.set_diploma_grade(conn, None)
    assert db.get_diploma_grade(conn) is None
    with pytest.raises(ValueError):
        db.set_diploma_grade(conn, 40)
