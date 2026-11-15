import pandas as pd
import pytest

from optistudy.scoring import add_net_column, exam_net_table, exam_totals, net


@pytest.mark.parametrize(
    "correct, wrong, expected",
    [
        (0, 0, 0.0),
        (40, 0, 40.0),
        (30, 8, 28.0),
        (20, 12, 17.0),
        (8, 6, 6.5),
        (0, 4, -1.0),   # net negatif olabilir
        (10, 3, 9.25),
    ],
)
def test_net_formula(correct, wrong, expected):
    assert net(correct, wrong) == pytest.approx(expected)


def test_net_rejects_negative():
    with pytest.raises(ValueError):
        net(-1, 0)


def test_add_net_column_does_not_mutate():
    df = pd.DataFrame({"correct": [30, 8], "wrong": [8, 6]})
    out = add_net_column(df)
    assert "net" not in df.columns
    assert list(out["net"]) == [28.0, 6.5]


def test_add_net_column_missing_column():
    with pytest.raises(KeyError):
        add_net_column(pd.DataFrame({"correct": [1]}))


def test_exam_totals_hand_computed():
    df = pd.DataFrame(
        {
            "exam_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "correct": [30, 12, 20, 8, 32, 14, 24, 10],
            "wrong":   [8, 4, 12, 6, 4, 2, 8, 4],
        }
    )
    totals = exam_totals(df)
    assert list(totals["exam_id"]) == [1, 2]
    # Deneme 1: 28 + 11 + 17 + 6.5 = 62.5 ; Deneme 2: 31 + 13.5 + 22 + 9 = 75.5
    assert list(totals["total_net"]) == [62.5, 75.5]
    assert list(totals["total_correct"]) == [70, 80]


def test_exam_net_table_wide_format():
    df = pd.DataFrame({
        "exam_id": [1, 1, 2, 2],
        "date": ["2026-09-06", "2026-09-06", "2026-09-13", "2026-09-13"],
        "publisher": ["A", "A", "B", "B"],
        "subject": ["Türkçe", "Fen", "Türkçe", "Fen"],
        "correct": [30, 8, 32, 10],
        "wrong": [8, 6, 4, 4],
    })
    t = exam_net_table(df)
    assert list(t.columns) == ["exam_id", "date", "publisher", "Türkçe", "Fen", "Toplam"]
    assert list(t["Türkçe"]) == [28.0, 31.0]
    assert list(t["Toplam"]) == [34.5, 40.0]  # 28 + 6.5 ; 31 + 9


def test_exam_net_table_without_exam_id_and_empty():
    df = pd.DataFrame({"date": ["d"], "publisher": ["A"], "subject": ["Fen"], "correct": [4], "wrong": [4]})
    assert list(exam_net_table(df).columns) == ["date", "publisher", "Fen", "Toplam"]
    assert exam_net_table(df.iloc[0:0]).empty
