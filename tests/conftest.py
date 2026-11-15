import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optistudy import db  # noqa: E402
from optistudy.data import load_subjects, load_topics  # noqa: E402


@pytest.fixture
def conn():
    """Ders ve konu listesi yüklü, boş, bellek içi veritabanı."""
    c = db.connect(":memory:")
    db.seed_subjects(c, load_subjects())
    db.seed_topics(c, load_topics())
    yield c
    c.close()


@pytest.fixture
def full_exam():
    """Geçerli, elle hesaplanabilir bir deneme."""
    return {
        "Türkçe": (30, 8, 2),     # net 28
        "Sosyal": (12, 4, 4),     # net 11
        "Matematik": (20, 12, 8), # net 17
        "Fen": (8, 6, 6),         # net 6.5
    }
