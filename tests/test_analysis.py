from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optistudy import analysis
from optistudy.analysis import (
    TOTAL_LABEL, VERDICT_DECREASE, VERDICT_INCREASE, VERDICT_NO_CLEAR_CHANGE,
    consistency, fit_trend, progress_summary, subject_trend,
)

START = date(2026, 9, 6)


def make_results(nets_by_subject: dict[str, list[float]], step_days: int = 7) -> pd.DataFrame:
    """Elle hesaplanabilir uzun tablo: net = doğru (yanlış = 0). Denemeler `step_days` arayla."""
    rows = []
    n = len(next(iter(nets_by_subject.values())))
    for i in range(n):
        for subject, nets in nets_by_subject.items():
            rows.append({
                "exam_id": i + 1, "date": START + timedelta(days=step_days * i),
                "publisher": f"Y{i}", "subject": subject, "question_count": 40,
                "correct": nets[i], "wrong": 0, "blank": 0,
            })
    return pd.DataFrame(rows, columns=[
        "exam_id", "date", "publisher", "subject", "question_count", "correct", "wrong", "blank",
    ])


def series(nets: list[float], step_days: int = 7) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [START + timedelta(days=step_days * i) for i in range(len(nets))],
        "net": nets,
    })


# --- trend ---------------------------------------------------------------

def test_trend_needs_three_exams():
    assert fit_trend(series([10, 12])) is None


def test_trend_needs_two_distinct_dates():
    assert fit_trend(series([10, 12, 14], step_days=0)) is None


def test_trend_perfect_line_weekly():
    # 10, 12, 14 haftada bir → eğim tam 2 net/hafta, artık yok → aralık sıfır genişlikte
    t = fit_trend(series([10, 12, 14]))
    assert t.slope_per_week == pytest.approx(2.0)
    assert t.intercept == pytest.approx(10.0)
    assert t.ci_low == pytest.approx(2.0) and t.ci_high == pytest.approx(2.0)
    assert t.verdict == VERDICT_INCREASE


def test_trend_slope_unit_is_per_week():
    # Aynı netler iki haftada bir → 1 net/hafta
    t = fit_trend(series([10, 12, 14], step_days=14))
    assert t.slope_per_week == pytest.approx(1.0)


def test_trend_ci_hand_computed_contains_zero():
    # x = 0,1,2 hafta; y = 10,13,14
    # eğim = Sxy/Sxx = 4/2 = 2 ; SSR = 2/3 ; s² = (2/3)/1
    # se = sqrt(s²/Sxx) = sqrt(1/3) ; t(0.975, sd=1) = 12.7062
    t = fit_trend(series([10, 13, 14]))
    se = np.sqrt(1 / 3)
    assert t.slope_per_week == pytest.approx(2.0)
    assert t.ci_low == pytest.approx(2 - 12.7062 * se, abs=1e-3)
    assert t.ci_high == pytest.approx(2 + 12.7062 * se, abs=1e-3)
    assert t.verdict == VERDICT_NO_CLEAR_CHANGE
    assert t.message == "Henüz net bir artış yok."


def test_trend_decrease():
    assert fit_trend(series([20, 18, 16, 14])).verdict == VERDICT_DECREASE


def test_fitted_values():
    t = fit_trend(series([10, 12, 14]))
    d = pd.Series(pd.to_datetime([START, START + timedelta(days=21)]))
    assert list(t.fitted(d)) == pytest.approx([10.0, 16.0])


def test_subject_trend_total_sums_subjects():
    res = make_results({"Türkçe": [10, 12, 14], "Fen": [5, 5, 5]})
    assert subject_trend(res, "Türkçe").slope_per_week == pytest.approx(2.0)
    assert subject_trend(res, "Fen").slope_per_week == pytest.approx(0.0)
    total = subject_trend(res, TOTAL_LABEL)
    assert total.intercept == pytest.approx(15.0)
    assert total.slope_per_week == pytest.approx(2.0)


# --- tutarlılık ----------------------------------------------------------

def test_consistency_uses_last_five_sample_std():
    # son 5: 12,14,16,18,20 → ortalama 16, kareler toplamı 40, /4 = 10 → sd = √10
    res = make_results({"Türkçe": [10, 12, 14, 16, 18, 20]})
    assert consistency(res, "Türkçe") == pytest.approx(np.sqrt(10))


def test_consistency_constant_is_zero():
    res = make_results({"Türkçe": [15, 15, 15]})
    assert consistency(res, "Türkçe") == 0.0


def test_consistency_two_exams_ddof1():
    # 10 ve 14 → sd = √((4 + 4) / 1) = √8
    res = make_results({"Türkçe": [10, 14]})
    assert consistency(res, "Türkçe") == pytest.approx(np.sqrt(8))


def test_consistency_needs_two_exams():
    assert consistency(make_results({"Türkçe": [10]}), "Türkçe") is None


def test_consistency_window_constant():
    assert analysis.CONSISTENCY_WINDOW == 5


# --- özet ----------------------------------------------------------------

def test_summary_empty_returns_none():
    assert progress_summary(make_results({"Türkçe": []})) is None


def test_summary_rows_and_values():
    res = make_results({"Türkçe": [10, 12, 14], "Fen": [4, 6, 5]})
    s = progress_summary(res).set_index("subject")
    assert list(s.index) == ["Türkçe", "Fen", TOTAL_LABEL]
    assert s.loc["Türkçe", "last_net"] == 14
    assert s.loc["Türkçe", "mean_net"] == 12
    assert s.loc["Türkçe", "slope_per_week"] == pytest.approx(2.0)
    assert s.loc[TOTAL_LABEL, "last_net"] == 19
    assert s.loc["Türkçe", "n_exams"] == 3


def test_summary_insufficient_data_leaves_blanks():
    s = progress_summary(make_results({"Türkçe": [10, 12]})).set_index("subject")
    assert np.isnan(s.loc["Türkçe", "slope_per_week"])
    assert s.loc["Türkçe", "verdict"] is None
    assert s.loc["Türkçe", "std_last5"] == pytest.approx(np.sqrt(2))


def test_latest_with_change():
    from optistudy.analysis import latest_with_change
    assert latest_with_change(series([10, 12.5])) == (12.5, 2.5)
    assert latest_with_change(series([10])) == (10.0, None)
    assert latest_with_change(series([])) is None
