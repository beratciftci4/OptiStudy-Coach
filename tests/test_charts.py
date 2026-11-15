from datetime import date, timedelta

import pandas as pd
import pytest

from optistudy.analysis import TOTAL_LABEL, progress_summary
from optistudy.charts import backtest_figure, consistency_figure, forecast_figure, net_trend_figure

COLUMNS = ["exam_id", "date", "publisher", "subject", "question_count", "correct", "wrong", "blank"]


def make_results(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for subject, c in (("Türkçe", 20 + i), ("Fen", 5 + i % 2)):
            rows.append({
                "exam_id": i + 1, "date": date(2026, 9, 6) + timedelta(days=7 * i),
                "publisher": "A", "subject": subject, "question_count": 40,
                "correct": c, "wrong": 0, "blank": 0,
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def test_trend_figure_has_trend_line_with_three_exams():
    fig = net_trend_figure(make_results(3), "Türkçe")
    assert [t.name for t in fig.data] == ["Net", "Trend"]
    assert list(fig.data[0].y) == [20, 21, 22]


def test_trend_figure_points_only_below_three():
    fig = net_trend_figure(make_results(2), TOTAL_LABEL)
    assert [t.name for t in fig.data] == ["Net"]


def test_trend_figure_none_when_empty():
    assert net_trend_figure(make_results(0), "Fen") is None


def test_consistency_figure_excludes_total():
    fig = consistency_figure(progress_summary(make_results(3)))
    assert list(fig.data[0].x) == ["Türkçe", "Fen"]


def test_consistency_figure_none_without_data():
    assert consistency_figure(progress_summary(make_results(1))) is None


def test_figures_have_zoom_disabled():
    for fig in (net_trend_figure(make_results(3), "Türkçe"), consistency_figure(progress_summary(make_results(3)))):
        assert fig.layout.dragmode is False
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_forecast_figure_extends_trend_and_adds_interval():
    from optistudy.analysis import predict_net, net_series
    res = make_results(6)
    target = date(2026, 9, 6) + timedelta(days=7 * 10)
    f = predict_net(net_series(res, "Türkçe"), target)
    fig = forecast_figure(res, "Türkçe", [f])
    assert [t.name for t in fig.data] == ["Net", "Trend", "Tahmin (%80 aralık)"]
    assert pd.Timestamp(fig.data[1].x[-1]).date() == target
    assert fig.data[2].error_y.array[0] == pytest.approx(f.high - f.point)


def test_backtest_figure_marks_hits_and_misses():
    from optistudy.analysis import BacktestResult
    steps = pd.DataFrame({
        "date": [date(2026, 10, 1), date(2026, 10, 8)], "actual": [10.0, 30.0],
        "point": [11.0, 12.0], "low": [8.0, 9.0], "high": [14.0, 15.0], "hit": [True, False],
    })
    fig = backtest_figure(BacktestResult(steps))
    names = [t.name for t in fig.data]
    assert names == ["Tahmin (%80 aralık)", "Gerçek net (aralıkta)", "Gerçek net (aralık dışı)"]
    assert fig.data[2].marker.symbol == "x"


def test_dark_colors_applied_to_trend_figure():
    from optistudy.charts import DARK_COLORS, chart_colors

    assert chart_colors(True) is DARK_COLORS
    fig = net_trend_figure(make_results(3), "Türkçe", chart_colors(True))
    assert fig.data[0].line.color == DARK_COLORS.series
    assert fig.data[1].line.color == DARK_COLORS.trend


def test_forecast_figure_draws_goal_line():
    fig = forecast_figure(make_results(3), TOTAL_LABEL, [], target_net=30)
    assert [s.y0 for s in fig.layout.shapes] == [30]
    assert fig.layout.annotations[0].text == "Hedef 30"
