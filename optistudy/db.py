"""SQLite bağlantısı, şema ve sorgular.

v2'de çoklu kullanıcıya geçerken tüm tablolara `student_id` eklenecek;
bu yüzden sorgular tablo ve sütun adlarını tek bir yerde (bu modülde) tutar.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from optistudy.data import (
    FIELDS, load_rank_reference, load_subjects, load_topics, required_subjects,
    validate_counts, validate_topic_totals,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id             INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL UNIQUE,
    question_count INTEGER NOT NULL CHECK (question_count > 0),
    exam_type      TEXT    NOT NULL DEFAULT 'TYT' CHECK (exam_type IN ('TYT', 'AYT', 'YDT'))
);

CREATE TABLE IF NOT EXISTS topics (
    id         INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    name       TEXT    NOT NULL,
    weight     REAL    CHECK (weight IS NULL OR weight >= 0),
    UNIQUE (subject_id, name)
);

CREATE TABLE IF NOT EXISTS exams (
    id        INTEGER PRIMARY KEY,
    date      TEXT NOT NULL,                       -- YYYY-AA-GG
    publisher TEXT NOT NULL,
    exam_type TEXT NOT NULL DEFAULT 'TYT' CHECK (exam_type IN ('TYT', 'AYT', 'YDT')),
    notes     TEXT,
    UNIQUE (date, publisher, exam_type)
);

CREATE TABLE IF NOT EXISTS subject_results (
    exam_id    INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    correct    INTEGER NOT NULL CHECK (correct >= 0),
    wrong      INTEGER NOT NULL CHECK (wrong >= 0),
    blank      INTEGER NOT NULL CHECK (blank >= 0),
    PRIMARY KEY (exam_id, subject_id)
);

CREATE TABLE IF NOT EXISTS topic_results (
    exam_id  INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    correct  INTEGER NOT NULL DEFAULT 0 CHECK (correct >= 0),
    wrong    INTEGER NOT NULL DEFAULT 0 CHECK (wrong >= 0),
    blank    INTEGER NOT NULL DEFAULT 0 CHECK (blank >= 0),
    PRIMARY KEY (exam_id, topic_id)
);

CREATE TABLE IF NOT EXISTS rank_reference (
    year       INTEGER NOT NULL,
    score_type TEXT    NOT NULL,
    score      REAL    NOT NULL,
    rank       INTEGER NOT NULL CHECK (rank > 0),
    PRIMARY KEY (year, score_type, score)
);

-- Tek öğrenci: id her zaman 1. v2'de id = student_id olacak.
CREATE TABLE IF NOT EXISTS student_profile (
    id            INTEGER PRIMARY KEY,
    diploma_grade REAL CHECK (diploma_grade IS NULL OR diploma_grade BETWEEN 50 AND 100),
    field         TEXT CHECK (field IS NULL OR field IN ('SAY', 'EA', 'SÖZ', 'DİL'))
);

-- Oturum başına hedef toplam net (TYT ve öğrencinin alan oturumu).
CREATE TABLE IF NOT EXISTS goals (
    student_id INTEGER NOT NULL DEFAULT 1,
    exam_type  TEXT    NOT NULL CHECK (exam_type IN ('TYT', 'AYT', 'YDT')),
    target_net REAL    NOT NULL CHECK (target_net > 0),
    PRIMARY KEY (student_id, exam_type)
);
"""

STUDENT_ID = 1

EXAMS_V2_SQL = """CREATE TABLE exams (
    id        INTEGER PRIMARY KEY,
    date      TEXT NOT NULL,                       -- YYYY-AA-GG
    publisher TEXT NOT NULL,
    exam_type TEXT NOT NULL DEFAULT 'TYT' CHECK (exam_type IN ('TYT', 'AYT', 'YDT')),
    notes     TEXT,
    UNIQUE (date, publisher, exam_type)
);"""

# Şema sürümü (PRAGMA user_version). 0/1: sadece TYT; 2: AYT/YDT ve öğrenci alanı;
# 3: hedefler (`goals`). 2 → 3 için sadece yeni tablo eklenir, `SCHEMA` bunu kendisi yapar.
SCHEMA_VERSION = 3
BACKUP_SUFFIX = "-v1-yedek"


# Uygulamanın varsayılan veritabanı; testlerde OPTISTUDY_DB ortam değişkeniyle değiştirilir.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "optistudy.db"


class DuplicateExamError(ValueError):
    """Aynı tarih, yayınevi ve sınav türüne ait deneme zaten kayıtlıysa fırlatılır."""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate_to_v2(conn: sqlite3.Connection, path: Path | str) -> None:
    """Sadece TYT'li eski şemayı AYT/YDT destekli şemaya taşır; veriler korunur.

    Dosya veritabanıysa önce yanına `<ad>-v1-yedek.db` kopyası alınır. `exams` tablosu
    SQLite'ın önerdiği yolla (yeni tablo → kopyala → eskiyi sil → yeniden adlandır)
    yeniden kurulur, çünkü CHECK ve UNIQUE kısıtları ALTER ile değiştirilemez.
    """
    if str(path) != ":memory:":
        src = Path(path)
        backup = src.with_name(src.stem + BACKUP_SUFFIX + src.suffix)
        if not backup.exists():
            with closing(sqlite3.connect(str(backup))) as dest:
                conn.backup(dest)

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        with conn:
            conn.execute(EXAMS_V2_SQL.replace("exams (", "exams_v2 (", 1))
            conn.execute(
                "INSERT INTO exams_v2 (id, date, publisher, exam_type, notes) "
                "SELECT id, date, publisher, exam_type, notes FROM exams"
            )
            conn.execute("DROP TABLE exams")
            conn.execute("ALTER TABLE exams_v2 RENAME TO exams")
            if "exam_type" not in _columns(conn, "subjects"):
                conn.execute(
                    "ALTER TABLE subjects ADD COLUMN exam_type TEXT NOT NULL DEFAULT 'TYT' "
                    "CHECK (exam_type IN ('TYT', 'AYT', 'YDT'))"
                )
            profile = _columns(conn, "student_profile")
            if profile and "field" not in profile:
                conn.execute(
                    "ALTER TABLE student_profile ADD COLUMN field TEXT "
                    "CHECK (field IS NULL OR field IN ('SAY', 'EA', 'SÖZ', 'DİL'))"
                )
        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"Geçiş sonrası yabancı anahtar hatası: {problems[:3]}")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def connect(path: Path | str = ":memory:") -> sqlite3.Connection:
    """Bağlantı açar, yabancı anahtarları etkinleştirir, gerekirse eski şemayı taşır ve şemayı kurar."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    has_exams = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='exams'").fetchone()
    if has_exams and version < 2:
        _migrate_to_v2(conn, path)
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def open_app_db(path: Path | str | None = None) -> sqlite3.Connection:
    """Uygulama için bağlantı açar ve ders listesinin yüklü olduğunu garanti eder.

    Yol verilmezse önce `OPTISTUDY_DB` ortam değişkenine, yoksa `DEFAULT_DB_PATH`'e bakar.
    """
    conn = connect(path or app_db_path())
    seed_subjects(conn, load_subjects())
    seed_topics(conn, load_topics())
    seed_rank_reference(conn, load_rank_reference())
    return conn


def app_db_path() -> Path:
    """Uygulamanın kullandığı veritabanı dosyası: `OPTISTUDY_DB` varsa o, yoksa `DEFAULT_DB_PATH`."""
    return Path(os.environ.get("OPTISTUDY_DB") or DEFAULT_DB_PATH)


def start_fresh(path: Path | str, stamp: str) -> Path | None:
    """Veritabanını yedek adıyla kenara alır ve boş bir veritabanıyla yeniden başlar.

    Örnek veriden gerçek veriye geçerken kullanılır; hiçbir şey silinmez. Yedek
    `<ad>-yedek-<stamp>.db` olarak aynı klasöre taşınır. Öğrenci profili (alan ve
    diploma notu) ve hedefler yeni veritabanına aktarılır. Dosya yoksa hiçbir şey yapmaz ve `None` döner.
    """
    src = Path(path)
    if not src.exists():
        return None
    backup = src.with_name(f"{src.stem}-yedek-{stamp}{src.suffix}")
    if backup.exists():
        raise FileExistsError(f"Yedek dosyası zaten var: {backup.name}")

    with closing(open_app_db(src)) as conn:
        field, grade, goals = get_field(conn), get_diploma_grade(conn), goals_dict(conn)
    src.rename(backup)
    with closing(open_app_db(src)) as conn:
        set_field(conn, field)
        set_diploma_grade(conn, grade)
        for exam_type, target in goals.items():
            set_goal(conn, exam_type, target)
    return backup


def seed_subjects(conn: sqlite3.Connection, subjects: pd.DataFrame) -> None:
    """Ders listesini yükler; varsa soru sayısını ve oturumunu günceller."""
    if "exam_type" not in subjects.columns:
        subjects = subjects.assign(exam_type="TYT")
    conn.executemany(
        """INSERT INTO subjects (name, question_count, exam_type) VALUES (?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET question_count = excluded.question_count,
                                           exam_type = excluded.exam_type""",
        list(subjects[["name", "question_count", "exam_type"]].itertuples(index=False, name=None)),
    )
    conn.commit()


# {(ders, konu): (doğru, yanlış, boş)}
TopicResults = dict[tuple[str, str], tuple[int, int, int]]


def seed_topics(conn: sqlite3.Connection, topics: pd.DataFrame) -> None:
    """Konu listesini yükler; varsa ağırlığı günceller. Bilinmeyen ağırlık NULL yazılır."""
    subject_ids = {name: sid for sid, name in conn.execute("SELECT id, name FROM subjects")}
    rows = [
        (subject_ids[r.subject], r.name, None if pd.isna(r.weight) else float(r.weight))
        for r in topics.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT INTO topics (subject_id, name, weight) VALUES (?, ?, ?)
           ON CONFLICT(subject_id, name) DO UPDATE SET weight = excluded.weight""",
        rows,
    )
    conn.commit()


def _topic_map(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    """{(ders adı, konu adı): konu id}"""
    return {
        (subject, name): tid
        for tid, subject, name in conn.execute(
            "SELECT t.id, s.name, t.name FROM topics t JOIN subjects s ON s.id = t.subject_id"
        )
    }


def _validate_topics(
    date: str,
    results: dict[str, tuple[int, int, int]],
    topic_results: TopicResults,
    subjects: dict[str, tuple[int, int]],
    topics: dict[tuple[str, str], int],
) -> None:
    """Konu sonuçlarını doğrular: konu bilinmeli, her satır geçerli olmalı ve bir dersin
    konu toplamı (D, Y, B ayrı ayrı) ders sonucunu aşmamalı."""
    unknown = set(topic_results) - set(topics)
    if unknown:
        names = sorted(f"{s} / {t}" for s, t in unknown)
        raise ValueError(f"Bilinmeyen konu(lar): {names}")
    by_subject: dict[str, list[dict[str, int]]] = {}
    for (subject, topic), (c, w, b) in topic_results.items():
        validate_counts(c, w, b, subjects[subject][1], label=f"{date} {subject} / {topic}")
        by_subject.setdefault(subject, []).append({"correct": c, "wrong": w, "blank": b})
    for subject, rows in by_subject.items():
        c, w, b = results[subject]
        validate_topic_totals({"correct": c, "wrong": w, "blank": b}, rows, label=f"{date} {subject}")


def _insert_topics(
    conn: sqlite3.Connection, exam_id: int, topic_results: TopicResults, topics: dict[tuple[str, str], int],
) -> None:
    """Doğrulanmış konu sonuçlarını yazar; hepsi sıfır olan satırlar atlanır."""
    conn.executemany(
        "INSERT INTO topic_results (exam_id, topic_id, correct, wrong, blank) VALUES (?, ?, ?, ?, ?)",
        [(exam_id, topics[key], c, w, b) for key, (c, w, b) in topic_results.items() if c + w + b > 0],
    )


def _subject_map(conn: sqlite3.Connection) -> dict[str, tuple[int, int, str]]:
    """{ders adı: (id, soru sayısı, oturum)}"""
    return {
        name: (sid, qc, et)
        for sid, name, qc, et in conn.execute("SELECT id, name, question_count, exam_type FROM subjects")
    }


def subjects_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Ders listesi: `name`, `question_count`, `exam_type`, kayıt sırasıyla."""
    return pd.read_sql_query("SELECT name, question_count, exam_type FROM subjects ORDER BY id", conn)


def _validate_exam(
    date: str,
    results: dict[str, tuple[int, int, int]],
    subjects: dict[str, tuple[int, int, str]],
    required: list[str],
) -> None:
    """Zorunlu derslerin hepsinin (ve sadece onların) girildiğini ve her sonucun geçerli olduğunu kontrol eder."""
    if set(results) != set(required):
        raise ValueError(
            f"Bu denemenin dersleri girilmeli. Eksik: {sorted(set(required) - set(results))}, "
            f"fazla/bilinmeyen: {sorted(set(results) - set(required))}"
        )
    for name, (c, w, b) in results.items():
        validate_counts(c, w, b, subjects[name][1], label=f"{date} {name}")


def _required(conn: sqlite3.Connection, exam_type: str) -> list[str]:
    return required_subjects(subjects_frame(conn), exam_type, get_field(conn))


def _insert_exam(
    conn: sqlite3.Connection,
    date: str,
    publisher: str,
    results: dict[str, tuple[int, int, int]],
    subjects: dict[str, tuple[int, int, str]],
    notes: str | None,
    exam_type: str = "TYT",
) -> int:
    """Doğrulanmış bir denemeyi yazar. Sadece açık bir işlem (`with conn`) içinde çağrılır."""
    try:
        cur = conn.execute(
            "INSERT INTO exams (date, publisher, exam_type, notes) VALUES (?, ?, ?, ?)",
            (str(date), publisher, exam_type, notes),
        )
    except sqlite3.IntegrityError as exc:
        raise DuplicateExamError(f"{date} {publisher} ({exam_type}): bu deneme zaten kayıtlı.") from exc
    exam_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT INTO subject_results (exam_id, subject_id, correct, wrong, blank) VALUES (?, ?, ?, ?, ?)",
        [(exam_id, subjects[n][0], c, w, b) for n, (c, w, b) in results.items()],
    )
    return exam_id


def add_exam(
    conn: sqlite3.Connection,
    date: str,
    publisher: str,
    results: dict[str, tuple[int, int, int]],
    notes: str | None = None,
    topic_results: TopicResults | None = None,
    exam_type: str = "TYT",
) -> int:
    """Bir denemeyi ders (ve varsa konu) sonuçlarıyla birlikte tek işlemde kaydeder.

    `results`: {ders adı: (doğru, yanlış, boş)}. TYT'de 4 ders; AYT/YDT'de öğrencinin
    alanının testleri zorunlu (bkz. `data.required_subjects`).
    `topic_results`: {(ders, konu): (doğru, yanlış, boş)}; opsiyonel ve kısmi olabilir.
    Doğrulama başarısızsa veya deneme zaten kayıtlıysa hiçbir şey yazılmaz.
    Yeni denemenin id'sini döndürür.
    """
    subjects = _subject_map(conn)
    _validate_exam(str(date), results, subjects, _required(conn, exam_type))
    topics = _topic_map(conn)
    if topic_results:
        _validate_topics(str(date), results, topic_results, subjects, topics)
    with conn:  # hata olursa geri alınır
        exam_id = _insert_exam(conn, str(date), publisher, results, subjects, notes, exam_type)
        if topic_results:
            _insert_topics(conn, exam_id, topic_results, topics)
        return exam_id


def import_exam_frame(conn: sqlite3.Connection, df: pd.DataFrame) -> list[int]:
    """`data.read_exam_csv` çıktısını (doğrulanmış uzun format) veritabanına yazar.

    Ya hep ya hiç: herhangi bir deneme geçersizse veya zaten kayıtlıysa hiçbiri yazılmaz.
    """
    subjects = _subject_map(conn)
    if "exam_type" not in df.columns:
        df = df.assign(exam_type="TYT")
    exams: list[tuple[str, str, str, dict[str, tuple[int, int, int]]]] = []
    for (date, publisher, exam_type), group in df.groupby(["date", "publisher", "exam_type"], sort=True):
        results = {
            r.subject: (int(r.correct), int(r.wrong), int(r.blank))
            for r in group.itertuples(index=False)
        }
        _validate_exam(str(date), results, subjects, _required(conn, exam_type))
        exams.append((str(date), str(publisher), str(exam_type), results))

    with conn:
        return [_insert_exam(conn, d, p, r, subjects, None, t) for d, p, t, r in exams]


def import_topic_frame(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """`data.read_topic_csv` çıktısını kayıtlı denemelere ekler. Yazılan satır sayısını döndürür.

    Denemenin oturumu (TYT/AYT/YDT) dersten anlaşılır. Her (tarih, yayınevi, oturum) için
    deneme önceden kayıtlı olmalı ve henüz konu sonucu olmamalı.
    Ya hep ya hiç: bir hata varsa hiçbir şey yazılmaz.
    """
    subjects = _subject_map(conn)
    topics = _topic_map(conn)
    df = df.assign(exam_type=df["subject"].map(lambda name: subjects[name][2] if name in subjects else None))
    batches: list[tuple[int, TopicResults]] = []
    for (date, publisher, exam_type), group in df.groupby(["date", "publisher", "exam_type"], sort=True):
        label = f"{date} {publisher} ({exam_type})"
        row = conn.execute(
            "SELECT id FROM exams WHERE date = ? AND publisher = ? AND exam_type = ?",
            (str(date), str(publisher), exam_type),
        ).fetchone()
        if row is None:
            raise ValueError(f"{label}: bu deneme kayıtlı değil, önce ders sonuçlarını gir.")
        exam_id = int(row[0])
        if conn.execute("SELECT 1 FROM topic_results WHERE exam_id = ?", (exam_id,)).fetchone():
            raise DuplicateExamError(f"{label}: bu denemenin konu sonuçları zaten kayıtlı.")
        results = {
            name: (c, w, b)
            for name, c, w, b in conn.execute(
                """SELECT s.name, r.correct, r.wrong, r.blank FROM subject_results r
                   JOIN subjects s ON s.id = r.subject_id WHERE r.exam_id = ?""",
                (exam_id,),
            )
        }
        topic_results = {
            (r.subject, r.topic): (int(r.correct), int(r.wrong), int(r.blank))
            for r in group.itertuples(index=False)
        }
        missing = {s for s, _ in topic_results} - set(results)
        if missing:
            raise ValueError(f"{label}: bu denemede {sorted(missing)} dersi yok.")
        _validate_topics(str(date), results, topic_results, subjects, topics)
        batches.append((exam_id, topic_results))

    with conn:
        for exam_id, topic_results in batches:
            _insert_topics(conn, exam_id, topic_results, topics)
    return sum(1 for _, tr in batches for v in tr.values() if sum(v) > 0)


def delete_exam(conn: sqlite3.Connection, exam_id: int) -> bool:
    """Denemeyi ve ona bağlı tüm sonuçları siler. Silindiyse True döndürür."""
    with conn:
        cur = conn.execute("DELETE FROM exams WHERE id = ?", (int(exam_id),))
    return cur.rowcount > 0


def subject_results_frame(conn: sqlite3.Connection, exam_type: str | None = None) -> pd.DataFrame:
    """Tüm ders sonuçlarını analiz için tek bir DataFrame olarak döndürür.

    Sütunlar: exam_id, date, publisher, exam_type, subject, question_count, correct, wrong, blank.
    `exam_type` verilirse sadece o oturumun denemeleri döner.
    """
    query = """
        SELECT e.id AS exam_id, e.date, e.publisher, e.exam_type, s.name AS subject, s.question_count,
               r.correct, r.wrong, r.blank
        FROM subject_results r
        JOIN exams e    ON e.id = r.exam_id
        JOIN subjects s ON s.id = r.subject_id
        WHERE (? IS NULL OR e.exam_type = ?)
        ORDER BY e.date, e.id, s.id
    """
    df = pd.read_sql_query(query, conn, params=(exam_type, exam_type))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def seed_rank_reference(conn: sqlite3.Connection, ref: pd.DataFrame) -> None:
    """ÖSYM puan–sıralama tablosunu yükler; aynı (yıl, puan türü, puan) varsa günceller."""
    conn.executemany(
        """INSERT INTO rank_reference (year, score_type, score, rank) VALUES (?, ?, ?, ?)
           ON CONFLICT(year, score_type, score) DO UPDATE SET rank = excluded.rank""",
        [(int(y), str(t), float(sc), int(r))
         for y, t, sc, r in ref[["year", "score_type", "score", "rank"]].itertuples(index=False)],
    )
    conn.commit()


def rank_reference_frame(conn: sqlite3.Connection, score_type: str, year: int | None = None) -> pd.DataFrame:
    """Bir puan türünün sıralama tablosu (`year`, `score`, `rank`), puana göre artan.

    Yıl verilmezse en güncel yıl kullanılır. Tablo boşsa boş DataFrame döner.
    """
    if year is None:
        row = conn.execute(
            "SELECT MAX(year) FROM rank_reference WHERE score_type = ?", (score_type,)
        ).fetchone()
        year = row[0]
    return pd.read_sql_query(
        "SELECT year, score, rank FROM rank_reference WHERE score_type = ? AND year = ? ORDER BY score",
        conn, params=(score_type, year),
    )


def get_diploma_grade(conn: sqlite3.Connection) -> float | None:
    """Kayıtlı diploma notu (50–100); girilmemişse `None`."""
    row = conn.execute(
        "SELECT diploma_grade FROM student_profile WHERE id = ?", (STUDENT_ID,)
    ).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def set_diploma_grade(conn: sqlite3.Connection, grade: float | None) -> None:
    """Diploma notunu kaydeder (`None` siler). 50–100 dışı değerde `ValueError`."""
    if grade is not None and not 50 <= grade <= 100:
        raise ValueError("Diploma notu 50 ile 100 arasında olmalı.")
    with conn:
        conn.execute(
            """INSERT INTO student_profile (id, diploma_grade) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET diploma_grade = excluded.diploma_grade""",
            (STUDENT_ID, grade),
        )


def topics_frame(conn: sqlite3.Connection, exam_type: str | None = None) -> pd.DataFrame:
    """Konu listesi: `subject`, `topic`, `weight` (bilinmiyorsa NaN), ders sırasıyla."""
    return pd.read_sql_query(
        """SELECT s.name AS subject, t.name AS topic, t.weight
           FROM topics t JOIN subjects s ON s.id = t.subject_id
           WHERE (? IS NULL OR s.exam_type = ?)
           ORDER BY s.id, t.id""",
        conn, params=(exam_type, exam_type),
    )


def topic_results_frame(conn: sqlite3.Connection, exam_type: str | None = None) -> pd.DataFrame:
    """Konu sonuçları: exam_id, date, exam_type, subject, topic, correct, wrong, blank."""
    df = pd.read_sql_query(
        """SELECT e.id AS exam_id, e.date, e.exam_type, s.name AS subject, t.name AS topic,
                  r.correct, r.wrong, r.blank
           FROM topic_results r
           JOIN exams e    ON e.id = r.exam_id
           JOIN topics t   ON t.id = r.topic_id
           JOIN subjects s ON s.id = t.subject_id
           WHERE (? IS NULL OR e.exam_type = ?)
           ORDER BY e.date, e.id, s.id, t.id""",
        conn, params=(exam_type, exam_type),
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def get_field(conn: sqlite3.Connection) -> str | None:
    """Öğrencinin seçtiği alan (SAY/EA/SÖZ/DİL); seçilmemişse `None`."""
    row = conn.execute("SELECT field FROM student_profile WHERE id = ?", (STUDENT_ID,)).fetchone()
    return None if row is None else row[0]


def set_field(conn: sqlite3.Connection, field: str | None) -> None:
    """Öğrencinin alanını kaydeder (`None` siler). Bilinmeyen alanda `ValueError`."""
    if field is not None and field not in FIELDS:
        raise ValueError(f"Alan {', '.join(FIELDS)} değerlerinden biri olmalı.")
    with conn:
        conn.execute(
            """INSERT INTO student_profile (id, field) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET field = excluded.field""",
            (STUDENT_ID, field),
        )


def export_exam_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Tüm ders sonuçlarını içe aktarma biçiminde döndürür (yedek / taşıma için).

    Sütunlar: date, publisher, exam_type, subject, correct, wrong, blank. Bu tablo
    `data.read_exam_csv` + `import_exam_frame` ile aynen geri yüklenebilir.
    """
    return pd.read_sql_query(
        """SELECT e.date, e.publisher, e.exam_type, s.name AS subject, r.correct, r.wrong, r.blank
           FROM subject_results r
           JOIN exams e    ON e.id = r.exam_id
           JOIN subjects s ON s.id = r.subject_id
           ORDER BY e.date, e.exam_type, e.publisher, s.id""",
        conn,
    )


def export_topic_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Tüm konu sonuçlarını konu CSV'si biçiminde döndürür (date, publisher, subject, topic, correct, wrong, blank)."""
    return pd.read_sql_query(
        """SELECT e.date, e.publisher, s.name AS subject, t.name AS topic, r.correct, r.wrong, r.blank
           FROM topic_results r
           JOIN exams e    ON e.id = r.exam_id
           JOIN topics t   ON t.id = r.topic_id
           JOIN subjects s ON s.id = t.subject_id
           ORDER BY e.date, e.exam_type, e.publisher, s.id, t.id""",
        conn,
    )


def goals_dict(conn: sqlite3.Connection) -> dict[str, float]:
    """Kayıtlı hedefler: {oturum: hedef toplam net}."""
    rows = conn.execute(
        "SELECT exam_type, target_net FROM goals WHERE student_id = ?", (STUDENT_ID,)
    ).fetchall()
    return {exam_type: float(target) for exam_type, target in rows}


def get_goal(conn: sqlite3.Connection, exam_type: str) -> float | None:
    """Bir oturumun hedef toplam neti; girilmemişse `None`."""
    return goals_dict(conn).get(exam_type)


def set_goal(
    conn: sqlite3.Connection, exam_type: str, target_net: float | None, max_net: float | None = None,
) -> None:
    """Hedef neti kaydeder (`None` siler).

    Hedef pozitif olmalı; `max_net` verilirse (oturumun soru sayısı) onu aşamaz. Aksi halde `ValueError`.
    """
    if target_net is None:
        with conn:
            conn.execute(
                "DELETE FROM goals WHERE student_id = ? AND exam_type = ?", (STUDENT_ID, exam_type)
            )
        return
    if target_net <= 0:
        raise ValueError("Hedef net 0'dan büyük olmalı.")
    if max_net is not None and target_net > max_net:
        raise ValueError(f"Hedef net en fazla {max_net:g} olabilir.")
    with conn:
        conn.execute(
            """INSERT INTO goals (student_id, exam_type, target_net) VALUES (?, ?, ?)
               ON CONFLICT(student_id, exam_type) DO UPDATE SET target_net = excluded.target_net""",
            (STUDENT_ID, exam_type, float(target_net)),
        )
