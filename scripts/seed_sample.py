"""Örnek veriyi yerel SQLite veritabanına yükler.

Kullanım:
    python scripts/seed_sample.py            # optistudy.db yoksa oluşturur
    python scripts/seed_sample.py --reset    # siler ve yeniden oluşturur
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optistudy import db  # noqa: E402
from optistudy.data import load_subjects, read_exam_csv, read_topic_csv  # noqa: E402
from optistudy.scoring import exam_totals  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "optistudy.db"))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.reset and db_path.exists():
        db_path.unlink()

    conn = db.open_app_db(db_path)  # ders, konu ve sıralama tablolarını da yükler
    subjects = load_subjects()

    already = conn.execute("SELECT COUNT(*) FROM exams").fetchone()[0]
    if already:
        print(f"Veritabanında zaten {already} deneme var. Sıfırlamak için --reset kullanın.")
        return

    frame = read_exam_csv(ROOT / "data" / "sample" / "exams.csv", subjects)
    ids = db.import_exam_frame(conn, frame)
    sample = ROOT / "data" / "sample"
    if (sample / "ayt_exams.csv").exists():
        db.set_field(conn, "SAY")  # örnek öğrenci sayısalcı
        ayt_ids = db.import_exam_frame(conn, read_exam_csv(sample / "ayt_exams.csv", subjects, field="SAY"))
        print(f"{len(ayt_ids)} AYT denemesi yüklendi (alan: SAY)")
    for name in ("topic_results.csv", "ayt_topic_results.csv"):
        if (sample / name).exists():
            n_topics = db.import_topic_frame(conn, read_topic_csv(sample / name))
            print(f"{n_topics} konu sonucu yüklendi ({name})")

    totals = exam_totals(db.subject_results_frame(conn, "TYT"))
    print(f"{len(ids)} deneme yüklendi: {db_path.name}")
    print(totals[["date", "total_net"]].to_string(index=False))


if __name__ == "__main__":
    main()
