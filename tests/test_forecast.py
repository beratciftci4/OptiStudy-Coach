"""Net tahmini, backtest ve sınav günü kapısı testleri."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from optistudy.analysis import (
    MIN_COVERAGE_LOWER_BOUND, TOTAL_LABEL, BacktestResult,
    backtest, exam_day_ready, net_bounds, predict_exam_day, predict_net,
)
from optistudy.data import load_exam_dates, next_exam

START = date(2026, 9, 6)


def series(nets: list[float], step_days: int = 7) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [START + timedelta(days=step_days * i) for i in range(len(nets))],
        "net": nets,
    })


def week(i: int) -> date:
    return START + timedelta(days=7 * i)


# --- predict_net ---------------------------------------------------------

def test_forecast_needs_five_exams():
    assert predict_net(series([10, 12, 14, 16]), week(4)) is None
    assert predict_net(series([10, 12, 14, 16, 18]), week(5)) is not None


def test_prediction_interval_hand_computed():
    # x = 0..4 hafta, y = 10,13,14,15,18
    # x̄ = 2, Sxx = 10, ȳ = 14, Sxy = 18 → eğim 1.8, sabit 10.4
    # artıklar: -0.4, 0.8, 0, -0.8, 0.4 → SSR = 1.6, s² = 1.6/3
    # x₀ = 5 → ŷ = 19.4 ; se = s·√(1 + 1/5 + 9/10) = √(1.12)
    f = predict_net(series([10, 13, 14, 15, 18]), week(5))
    half = stats.t.ppf(0.90, df=3) * np.sqrt(1.12)
    assert f.point == pytest.approx(19.4)
    assert f.low == pytest.approx(19.4 - half)
    assert f.high == pytest.approx(19.4 + half)
    assert f.level == 0.80 and f.n_exams == 5


def test_prediction_interval_is_wider_than_confidence_interval():
    # Tahmin aralığı yeni bir gözlem içindir; ortalamanın güven aralığından geniş olmalı.
    # Ortalama için yarı genişlik: t·s·√(1/5 + 9/10) = t·√(0.5867)
    f = predict_net(series([10, 13, 14, 15, 18]), week(5))
    ci_half = stats.t.ppf(0.90, df=3) * np.sqrt(1.6 / 3 * (1 / 5 + 9 / 10))
    assert (f.high - f.low) / 2 > ci_half


def test_interval_widens_further_out():
    s = series([10, 13, 14, 15, 18])
    near, far = predict_net(s, week(5)), predict_net(s, week(30))
    assert (far.high - far.low) > (near.high - near.low)


def test_forecast_clipped_to_bounds():
    # Fen: en fazla 20 net. Trend 21'i gösterse de 20'yi aşamaz.
    f = predict_net(series([16, 17, 18, 19, 20]), week(5), bounds=(-5, 20))
    assert f.point == 20 and f.high == 20


def test_target_before_last_exam_rejected():
    with pytest.raises(ValueError):
        predict_net(series([10, 12, 14, 16, 18]), week(3))


def test_net_bounds():
    res = pd.DataFrame({
        "subject": ["Fen", "Türkçe", "Fen"], "question_count": [20, 40, 20],
    })
    assert net_bounds(res, "Fen") == (-5.0, 20.0)
    assert net_bounds(res, TOTAL_LABEL) == (-15.0, 60.0)


# --- backtest ------------------------------------------------------------

def test_backtest_needs_six_exams():
    assert backtest(series([10, 13, 14, 15, 18])) is None


def test_backtest_hit_and_miss():
    # Adım 1: ilk 5 denemeyle 6.'yı tahmin et → ŷ = 19.4; gerçek 19.4 → aralıkta
    # Adım 2: ilk 6 denemeyle 7.'yi tahmin et; gerçek 100 → kesinlikle aralık dışında
    bt = backtest(series([10, 13, 14, 15, 18, 19.4, 100]))
    assert bt.n == 2
    assert list(bt.steps["hit"]) == [True, False]
    assert bt.steps["point"].iloc[0] == pytest.approx(19.4)
    assert bt.coverage == 0.5


def test_backtest_uses_only_past_data():
    # Son denemeyi değiştirmek önceki adımların tahminini değiştirmemeli
    a = backtest(series([10, 13, 14, 15, 18, 19, 20, 21]))
    b = backtest(series([10, 13, 14, 15, 18, 19, 20, 90]))
    pd.testing.assert_series_equal(a.steps["point"].iloc[:-1], b.steps["point"].iloc[:-1])


# --- sınav günü kapısı ---------------------------------------------------

def result_with(hits: int, n: int) -> BacktestResult:
    return BacktestResult(pd.DataFrame({"hit": [True] * hits + [False] * (n - hits)}))


def test_wilson_interval_hand_computed():
    # 5/5 için Wilson alt sınırı: (1 + z²/2n − z·√(z²/4n²)) / (1 + z²/n), z = 1.96
    z, n = stats.norm.ppf(0.975), 5
    expected = (1 + z**2 / (2 * n) - z * np.sqrt(z**2 / (4 * n**2))) / (1 + z**2 / n)
    assert result_with(5, 5).coverage_ci[0] == pytest.approx(expected)


@pytest.mark.parametrize("hits, n, ready", [
    (5, 5, False),    # alt sınır ≈ 0.57
    (6, 6, True),     # alt sınır ≈ 0.61
    (8, 10, False),   # %80 tutuyor ama veri az → alt sınır ≈ 0.49
    (20, 25, True),   # %80 tutuyor, yeterli veri → alt sınır ≈ 0.61
    (14, 25, False),  # %56 → aralıklar fazla dar
])
def test_exam_day_gate(hits, n, ready):
    assert exam_day_ready(result_with(hits, n)) is ready


def test_gate_closed_without_backtest():
    assert exam_day_ready(None) is False
    assert MIN_COVERAGE_LOWER_BOUND == 0.60


def test_predict_exam_day_locked_then_open():
    # 6 deneme → tek backtest adımı; tutsa bile (alt sınır ≈ 0.21) kapı kapalı
    assert predict_exam_day(series([10, 13, 14, 15, 18, 19.4]), week(40)) is None
    # Haftada 1 net artış + ±0.5 dönüşümlü sapma: 12 denemede 7 adımın 7'si tutuyor
    s = series([10 + i + (0.5 if i % 2 else -0.5) for i in range(12)])
    bt = backtest(s)
    assert (bt.hits, bt.n) == (7, 7)
    assert exam_day_ready(bt)
    assert predict_exam_day(s, week(40)) == predict_net(s, week(40))


# --- sınav tarihleri -----------------------------------------------------

def test_exam_dates_reference_has_2027_tyt():
    df = load_exam_dates()
    row = df[(df["exam_type"] == "TYT") & (df["year"] == 2027)].iloc[0]
    assert row["date"] == date(2027, 6, 19)
    assert "ÖSYM" in row["source"]


def test_next_exam():
    df = load_exam_dates()
    assert next_exam(df, date(2026, 11, 5))["date"] == date(2027, 6, 19)
    assert next_exam(df, date(2027, 6, 19))["date"] == date(2027, 6, 19)
    assert next_exam(df, date(2027, 6, 20)) is None
