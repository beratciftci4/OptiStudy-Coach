"""Tamamen sahte, tekrarlanabilir örnek deneme verisi üretir.

Kullanım: python scripts/make_sample.py
Çıktı:
  data/sample/exams.csv          12 deneme, haftalık, yavaşça artan netler + gürültü
  data/sample/topic_results.csv  ilk 2 deneme hariç konu bazlı sonuçlar (ders toplamlarıyla birebir uyumlu)
  data/sample/ayt_exams.csv      sahte sayısal (SAY) öğrencinin 10 AYT denemesi (TYT'den bir gün sonra)
  data/sample/ayt_topic_results.csv  AYT denemelerinin konu bazlı sonuçları
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "exams.csv"
TOPIC_OUT = ROOT / "data" / "sample" / "topic_results.csv"
AYT_OUT = ROOT / "data" / "sample" / "ayt_exams.csv"
AYT_TOPIC_OUT = ROOT / "data" / "sample" / "ayt_topic_results.csv"

# Sahte SAY öğrencisi: AYT testleri için başlangıç doğru oranı ve haftalık gelişim (uydurma)
AYT_PROFILE = {
    "AYT Matematik": (40, 0.35, 0.012),
    "Fizik":         (14, 0.35, 0.010),
    "Kimya":         (13, 0.40, 0.010),
    "Biyoloji":      (13, 0.50, 0.008),
}

# Sahte denemelerin konu dağılımı (soru sayısı) ve sahte öğrencinin konu bazlı başarısı.
# Gerçek TYT ağırlığı DEĞİLDİR; sadece örnek veri üretmek içindir.
TOPIC_PROFILE = {
    "Türkçe": {"Sözcükte anlam": (4, 1.1), "Cümlede anlam": (6, 1.0), "Paragraf": (22, 1.1),
               "Dil bilgisi": (5, 0.6), "Yazım ve noktalama": (3, 0.8)},
    "Sosyal": {"Tarih": (5, 0.7), "Coğrafya": (5, 1.2), "Felsefe": (5, 1.0),
               "Din Kültürü ve Ahlak Bilgisi": (5, 1.3)},
    "Matematik": {"Temel kavramlar ve sayılar": (8, 1.3), "Problemler": (12, 0.6),
                  "Cebir ve fonksiyonlar": (7, 0.9), "Kümeler ve mantık": (2, 1.2),
                  "Olasılık ve veri": (3, 0.8), "Geometri": (8, 0.5)},
    "Fen": {"Fizik": (7, 0.6), "Kimya": (7, 1.0), "Biyoloji": (6, 1.4)},
    "AYT Matematik": {"Temel kavramlar ve sayılar": (4, 1.3), "Fonksiyonlar ve polinomlar": (5, 1.0),
                      "Denklemler ve eşitsizlikler": (3, 1.1), "Trigonometri": (4, 0.6),
                      "Logaritma, üstel fonksiyon ve diziler": (4, 0.8), "Limit ve süreklilik": (2, 1.0),
                      "Türev": (5, 0.7), "İntegral": (4, 0.5), "Olasılık ve kombinatorik": (3, 0.9),
                      "Geometri": (6, 0.6)},
    "Fizik": {"Kuvvet, hareket ve enerji": (5, 0.8), "Elektrik ve manyetizma": (4, 0.6),
              "Dalgalar ve optik": (3, 1.0), "Modern fizik": (2, 1.2)},
    "Kimya": {"Atom, periyodik sistem ve bağlar": (2, 1.3), "Kimyasal hesaplamalar ve tepkimeler": (2, 0.7),
              "Gazlar ve çözeltiler": (3, 0.9), "Tepkime hızı ve denge": (2, 0.8),
              "Elektrokimya": (2, 0.6), "Organik kimya": (2, 1.0)},
    "Biyoloji": {"İnsan fizyolojisi": (5, 1.1), "Hücre, kalıtım ve genetik": (4, 0.8),
                 "Bitki biyolojisi": (2, 0.7), "Ekoloji ve canlıların sınıflandırılması": (2, 1.2)},
}

# Sahte öğrenci: başlangıç doğru oranı ve haftalık gelişim (tamamen uydurma)
PROFILE = {
    "Türkçe":    (40, 0.70, 0.006),
    "Sosyal":    (20, 0.60, 0.008),
    "Matematik": (40, 0.45, 0.012),
    "Fen":       (20, 0.40, 0.010),
}
PUBLISHERS = ["Yayınevi A", "Yayınevi B", "Yayınevi C"]


def main(n_exams: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2026, 9, 6)
    rows = []
    for i in range(n_exams):
        d = start + timedelta(weeks=i)
        pub = PUBLISHERS[i % len(PUBLISHERS)]
        for subject, (q, p0, slope) in PROFILE.items():
            p = float(np.clip(p0 + slope * i + rng.normal(0, 0.05), 0.05, 0.95))
            correct = int(rng.binomial(q, p))
            remaining = q - correct
            wrong = int(rng.binomial(remaining, 0.6))
            blank = remaining - wrong
            rows.append((d.isoformat(), pub, subject, correct, wrong, blank))
    df = pd.DataFrame(rows, columns=["date", "publisher", "subject", "correct", "wrong", "blank"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


def make_ayt(n_exams: int = 10, seed: int = 11) -> pd.DataFrame:
    """Sahte SAY öğrencisinin AYT denemeleri; TYT denemelerinden bir gün sonra çözülmüş gibi."""
    rng = np.random.default_rng(seed)
    start = date(2026, 9, 7)
    rows = []
    for i in range(n_exams):
        d = start + timedelta(weeks=i)
        pub = PUBLISHERS[i % len(PUBLISHERS)]
        for subject, (q, p0, slope) in AYT_PROFILE.items():
            p = float(np.clip(p0 + slope * i + rng.normal(0, 0.04), 0.05, 0.95))
            correct = int(rng.binomial(q, p))
            wrong = int(rng.binomial(q - correct, 0.5))
            rows.append((d.isoformat(), pub, "AYT", subject, correct, wrong, q - correct - wrong))
    df = pd.DataFrame(rows, columns=["date", "publisher", "exam_type", "subject", "correct", "wrong", "blank"])
    df.to_csv(AYT_OUT, index=False)
    return df


def make_topic_results(
    exams: pd.DataFrame, skip_first: int = 2, seed: int = 7, out: Path = TOPIC_OUT,
) -> pd.DataFrame:
    """Her ders sonucunu konulara dağıtır; konu toplamları ders sonucuna birebir eşittir.

    Soruların hangi konudan geldiği `TOPIC_PROFILE`'daki sayılarla sabit; doğru cevaplar
    konulara sahte başarı katsayısıyla ağırlıklı, yanlış/boş ise kalan sorulara rastgele dağıtılır.
    """
    rng = np.random.default_rng(seed)
    rows = []
    dates = sorted(exams["date"].unique())[skip_first:]
    for r in exams[exams["date"].isin(dates)].itertuples(index=False):
        names = list(TOPIC_PROFILE[r.subject])
        labels = np.repeat(np.arange(len(names)), [TOPIC_PROFILE[r.subject][n][0] for n in names])
        skill = np.array([TOPIC_PROFILE[r.subject][names[i]][1] for i in labels])
        order = rng.choice(len(labels), size=len(labels), replace=False, p=skill / skill.sum())
        correct_idx = order[: r.correct]
        rest = rng.permutation(order[r.correct:])
        wrong_idx, blank_idx = rest[: r.wrong], rest[r.wrong: r.wrong + r.blank]
        for i, name in enumerate(names):
            c = int((labels[correct_idx] == i).sum())
            w = int((labels[wrong_idx] == i).sum())
            b = int((labels[blank_idx] == i).sum())
            rows.append((r.date, r.publisher, r.subject, name, c, w, b))
    df = pd.DataFrame(rows, columns=["date", "publisher", "subject", "topic", "correct", "wrong", "blank"])
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    out = main()
    print(f"{out['date'].nunique()} sahte deneme yazıldı: {OUT.relative_to(ROOT)}")
    topics = make_topic_results(out)
    print(f"{len(topics)} sahte konu sonucu yazıldı: {TOPIC_OUT.relative_to(ROOT)}")
    ayt = make_ayt()
    ayt_topics = make_topic_results(ayt, skip_first=1, seed=13, out=AYT_TOPIC_OUT)
    print(f"{ayt['date'].nunique()} sahte AYT denemesi ve {len(ayt_topics)} konu sonucu yazıldı")
