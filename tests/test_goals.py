"""Hedef takibi: elle hesaplanabilir küçük örnekler."""

from contextlib import closing
from datetime import date, timedelta

import pandas as pd
import pytest

from optistudy import db
from optistudy.analysis import (
    GOAL_BEHIND, GOAL_ON_TRACK, GOAL_REACHED, GOAL_UNCLEAR, focus_expected_loss, goal_progress,
)

START = date(2026, 9, 6)


def weekly(nets: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "exam_id": range(1, len(nets) + 1),
        "date": [START + timedelta(weeks=i) for i in range(len(nets))],
        "publisher": "A",
        "net": nets,
    })


def ten_weeks_after_last(nets: list[float]) -> date:
    return START + timedelta(weeks=len(nets) - 1 + 10)


# 50, 51, 52: tam doğrusal, eğim 1 net/hafta, güven aralığı [1, 1]; mevcut seviye 52.
PERFECT = [50, 51, 52]


def test_on_track_when_required_pace_below_trend():
    g = goal_progress(weekly(PERFECT), 60, ten_weeks_after_last(PERFECT))
    assert g.current_level == pytest.approx(52)
    assert g.weeks_left == pytest.approx(10)
    assert g.gap == pytest.approx(8)
    assert g.required_slope == pytest.approx(0.8)  # 8 net / 10 hafta
    assert g.share_done == pytest.approx(52 / 60)
    assert g.status == GOAL_ON_TRACK


def test_behind_when_required_pace_above_trend():
    g = goal_progress(weekly(PERFECT), 65, ten_weeks_after_last(PERFECT))
    assert g.required_slope == pytest.approx(1.3)  # 13 / 10 > 1
    assert g.status == GOAL_BEHIND


def test_reached_when_trend_already_above_target():
    g = goal_progress(weekly(PERFECT), 50, ten_weeks_after_last(PERFECT))
    assert g.gap == pytest.approx(-2)
    assert g.share_done == 1.0
    assert g.status == GOAL_REACHED


def test_unclear_when_required_pace_inside_slope_interval():
    # 50, 53, 52, 55 → eğim 7/5 = 1,4; son tarihte trend 50,4 + 3·1,4 = 54,6.
    # s² = 3,2/2 = 1,6; se = √(1,6/5) ≈ 0,566; t(0,975; 2) ≈ 4,303 → aralık ≈ [−1,03; 3,83]
    nets = [50, 53, 52, 55]
    g = goal_progress(weekly(nets), 54.6 + 14, ten_weeks_after_last(nets))
    assert g.current_level == pytest.approx(54.6)
    assert g.required_slope == pytest.approx(1.4)
    assert g.trend.ci_low == pytest.approx(-1.034, abs=1e-2)
    assert g.trend.ci_high == pytest.approx(3.834, abs=1e-2)
    assert g.status == GOAL_UNCLEAR


def test_none_below_three_exams():
    assert goal_progress(weekly([50, 51]), 60, date(2027, 6, 19)) is None


def test_none_when_exam_not_after_last_exam():
    last = START + timedelta(weeks=2)
    assert goal_progress(weekly(PERFECT), 60, last) is None


def test_focus_expected_loss_sums_priorities():
    assert focus_expected_loss(pd.DataFrame({"priority": [1.5, 0.75, 0.25]})) == pytest.approx(2.5)


# --- Veritabanı --------------------------------------------------------------

def test_set_get_and_delete_goal(conn):
    assert db.get_goal(conn, "TYT") is None
    db.set_goal(conn, "TYT", 85, max_net=120)
    db.set_goal(conn, "TYT", 90, max_net=120)  # günceller
    db.set_goal(conn, "AYT", 50)
    assert db.goals_dict(conn) == {"TYT": 90.0, "AYT": 50.0}
    db.set_goal(conn, "TYT", None)
    assert db.get_goal(conn, "TYT") is None


@pytest.mark.parametrize("target", [0, -5, 121])
def test_set_goal_rejects_invalid(conn, target):
    with pytest.raises(ValueError):
        db.set_goal(conn, "TYT", target, max_net=120)
    assert db.get_goal(conn, "TYT") is None


def test_start_fresh_keeps_goals(tmp_path):
    path = tmp_path / "app.db"
    with closing(db.open_app_db(path)) as c:
        db.set_goal(c, "TYT", 90)
    db.start_fresh(path, "x")
    with closing(db.open_app_db(path)) as c:
        assert db.get_goal(c, "TYT") == 90.0


def test_v2_database_upgrades_without_remigrating(tmp_path):
    path = tmp_path / "v2.db"
    with closing(db.open_app_db(path)) as c:
        c.execute("DROP TABLE goals")
        c.execute("PRAGMA user_version = 2")
    with closing(db.open_app_db(path)) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        db.set_goal(c, "TYT", 80)
    assert not list(tmp_path.glob("*-v1-yedek*"))  # v1 geçişi tekrar çalışmadı


@pytest.mark.parametrize("exam_type, field, expected", [
    ("TYT", None, 120), ("AYT", "SAY", 80), ("AYT", "EA", 80), ("AYT", "SÖZ", 80), ("YDT", "DİL", 80),
])
def test_session_max_net(exam_type, field, expected):
    from optistudy.data import load_subjects, session_max_net

    assert session_max_net(load_subjects(), exam_type, field) == expected


@pytest.mark.parametrize("target, expected", [(59, "below"), (60, "inside"), (70, "inside"), (71, "above")])
def test_goal_vs_forecast(target, expected):
    from optistudy.analysis import NetForecast, goal_vs_forecast

    forecast = NetForecast(date(2027, 6, 19), point=65, low=60, high=70, n_exams=8)
    assert goal_vs_forecast(target, forecast) == expected
