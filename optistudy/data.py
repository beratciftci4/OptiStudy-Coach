"""Veri okuma, doğrulama ve CSV içe aktarma."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import IO

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"

# Deneme CSV'sinde beklenen sütunlar (uzun format: deneme × ders başına bir satır).
# `exam_type` opsiyoneldir; yoksa tüm satırlar TYT sayılır.
EXAM_CSV_COLUMNS: list[str] = ["date", "publisher", "subject", "correct", "wrong", "blank"]
EXAM_TYPES: tuple[str, ...] = ("TYT", "AYT", "YDT")

# Alanlar (puan türleri) ve her alanın ikinci oturumda çözdüğü testler (ÖSYM Kılavuzu Tablo 1E)
FIELDS: tuple[str, ...] = ("SAY", "EA", "SÖZ", "DİL")
FIELD_NAMES: dict[str, str] = {"SAY": "Sayısal", "EA": "Eşit ağırlık", "SÖZ": "Sözel", "DİL": "Dil"}
FIELD_SESSION: dict[str, str] = {"SAY": "AYT", "EA": "AYT", "SÖZ": "AYT", "DİL": "YDT"}
FIELD_SUBJECTS: dict[str, tuple[str, ...]] = {
    "SAY": ("AYT Matematik", "Fizik", "Kimya", "Biyoloji"),
    "EA": ("Edebiyat", "Tarih-1", "Coğrafya-1", "AYT Matematik"),
    "SÖZ": ("Edebiyat", "Tarih-1", "Coğrafya-1", "Tarih-2", "Coğrafya-2", "Felsefe Grubu", "Din Kültürü"),
    "DİL": ("Yabancı Dil",),
}


class ValidationError(ValueError):
    """Girilen deneme verisi kurallara uymadığında fırlatılır."""


def load_subjects(path: Path | None = None, exam_type: str | None = None) -> pd.DataFrame:
    """Ders listesini okur (`name`, `question_count`, `exam_type`). İstenirse tek oturum."""
    df = pd.read_csv(path or REFERENCE_DIR / "subjects.csv")
    if exam_type is not None:
        df = df[df["exam_type"] == exam_type].reset_index(drop=True)
    return df


def required_subjects(subjects: pd.DataFrame, exam_type: str, field: str | None) -> list[str]:
    """Bir denemede girilmesi zorunlu dersler.

    TYT: TYT'nin 4 dersi. AYT/YDT: öğrencinin alanının o oturumdaki testleri; alan seçilmemişse
    veya alanın oturumu farklıysa (ör. DİL öğrencisi AYT girmez) `ValidationError`.
    """
    if exam_type not in EXAM_TYPES:
        raise ValidationError(f"Bilinmeyen sınav türü: {exam_type}")
    if exam_type == "TYT":
        return list(subjects.loc[subjects["exam_type"] == "TYT", "name"])
    if field is None:
        raise ValidationError(f"{exam_type} denemesi girmek için önce alanını seç.")
    if FIELD_SESSION[field] != exam_type:
        raise ValidationError(
            f"{FIELD_NAMES[field]} alanı {FIELD_SESSION[field]} çözer, {exam_type} değil."
        )
    return list(FIELD_SUBJECTS[field])


def session_max_net(subjects: pd.DataFrame, exam_type: str, field: str | None) -> float:
    """Bir oturumda alınabilecek en yüksek toplam net (zorunlu derslerin soru sayısı toplamı)."""
    names = required_subjects(subjects, exam_type, field)
    return float(subjects.loc[subjects["name"].isin(names), "question_count"].sum())


def load_topics(path: Path | None = None) -> pd.DataFrame:
    """Konu listesini okur (`subject`, `name`, `weight`, `source`). Bilinmeyen ağırlık NaN."""
    return pd.read_csv(path or REFERENCE_DIR / "topics.csv")


TOPIC_CSV_COLUMNS: list[str] = ["date", "publisher", "subject", "topic", "correct", "wrong", "blank"]


def read_topic_csv(path: Path | str | IO[bytes] | IO[str], topics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Konu bazlı sonuç CSV'sini okur ve satır düzeyinde doğrular.

    Beklenen sütunlar: date, publisher, subject, topic, correct, wrong, blank.
    Ders sonucuyla karşılaştırma (konu toplamı ≤ ders sonucu) veritabanına yazarken yapılır.
    """
    topics = topics if topics is not None else load_topics()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValidationError("CSV dosyası okunamadı. UTF-8 ve virgülle ayrılmış olmalı.") from exc
    missing = set(TOPIC_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValidationError(f"CSV'de eksik sütun(lar): {sorted(missing)}")
    df = df[TOPIC_CSV_COLUMNS].copy()
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    except (ValueError, TypeError) as exc:
        raise ValidationError("Tarihler YYYY-AA-GG biçiminde olmalı.") from exc
    for col in ("correct", "wrong", "blank"):
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ValidationError(f"'{col}' sütunu tam sayı olmalı.")
        if (df[col] < 0).any():
            raise ValidationError(f"'{col}' negatif olamaz.")
    known = set(zip(topics["subject"], topics["name"]))
    unknown = {(s, t) for s, t in zip(df["subject"], df["topic"])} - known
    if unknown:
        raise ValidationError(f"Bilinmeyen konu(lar): {sorted(f'{s} / {t}' for s, t in unknown)}")
    if df.duplicated(["date", "publisher", "subject", "topic"]).any():
        raise ValidationError("Aynı deneme ve konu birden fazla kez girilmiş.")
    return df.sort_values(["date", "publisher", "subject", "topic"]).reset_index(drop=True)


def load_exam_dates(path: Path | None = None) -> pd.DataFrame:
    """Sınav tarihlerini okur (`exam_type`, `year`, `date`, `source`)."""
    df = pd.read_csv(path or REFERENCE_DIR / "exam_dates.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    return df


def next_exam(exam_dates: pd.DataFrame, today: date, exam_type: str = "TYT") -> dict | None:
    """Bugünden sonraki (bugün dahil) ilk sınavı döndürür; yoksa `None`.

    Dönen sözlük: `date`, `year`, `source`.
    """
    upcoming = exam_dates[(exam_dates["exam_type"] == exam_type) & (exam_dates["date"] >= today)]
    if upcoming.empty:
        return None
    row = upcoming.sort_values("date").iloc[0]
    return {"date": row["date"], "year": int(row["year"]), "source": row["source"]}


def load_tyt_test_stats(path: Path | None = None, year: int | None = None) -> pd.DataFrame:
    """TYT test istatistiklerini ve ağırlıklarını okur. Yıl verilmezse en güncel yıl."""
    df = pd.read_csv(path or REFERENCE_DIR / "tyt_test_stats.csv")
    year = year if year is not None else int(df["year"].max())
    return df[df["year"] == year].reset_index(drop=True)


def load_ayt_test_stats(path: Path | None = None, year: int | None = None) -> pd.DataFrame:
    """AYT/YDT test istatistiklerini okur (geçerli soru sayısı, ortalama, standart sapma, aday sayısı)."""
    df = pd.read_csv(path or REFERENCE_DIR / "ayt_test_stats.csv")
    year = year if year is not None else int(df["year"].max())
    return df[df["year"] == year].reset_index(drop=True)


def load_score_weights(path: Path | None = None, year: int | None = None) -> pd.DataFrame:
    """Alan puan türlerinin test ağırlıklarını okur (`score_type`, `subject`, `weight`)."""
    df = pd.read_csv(path or REFERENCE_DIR / "score_weights.csv")
    year = year if year is not None else int(df["year"].max())
    return df[df["year"] == year].reset_index(drop=True)


def load_rank_reference(path: Path | None = None) -> pd.DataFrame:
    """ÖSYM puan–sıralama tablosunu okur (`year`, `score_type`, `score`, `rank`, `source`)."""
    return pd.read_csv(path or REFERENCE_DIR / "rank_reference.csv")


def validate_counts(correct: int, wrong: int, blank: int, question_count: int, label: str = "") -> None:
    """Tek bir sonuç satırını doğrular.

    Kurallar: değerler negatif olamaz, D + Y + B soru sayısını aşamaz.
    """
    prefix = f"{label}: " if label else ""
    if min(correct, wrong, blank) < 0:
        raise ValidationError(f"{prefix}doğru/yanlış/boş negatif olamaz.")
    total = correct + wrong + blank
    if total > question_count:
        raise ValidationError(
            f"{prefix}D + Y + B = {total}, ama bu derste {question_count} soru var."
        )


def validate_topic_totals(subject_row: dict[str, int], topic_rows: list[dict[str, int]], label: str = "") -> None:
    """Bir dersin konu sonuçları toplamı ders sonucunu aşamaz (D, Y, B ayrı ayrı)."""
    prefix = f"{label}: " if label else ""
    for field, name in (("correct", "doğru"), ("wrong", "yanlış"), ("blank", "boş")):
        topic_sum = sum(r.get(field, 0) for r in topic_rows)
        if topic_sum > subject_row[field]:
            raise ValidationError(
                f"{prefix}konu bazlı {name} toplamı ({topic_sum}) ders sonucunu ({subject_row[field]}) aşıyor."
            )


def read_exam_csv(
    path: Path | str | IO[bytes] | IO[str],
    subjects: pd.DataFrame | None = None,
    field: str | None = None,
) -> pd.DataFrame:
    """Deneme CSV'sini okur ve doğrular.

    `path` bir dosya yolu ya da açık bir dosya nesnesi (ör. arayüzden yüklenen dosya) olabilir.
    Beklenen sütunlar: date, publisher, subject, correct, wrong, blank; opsiyonel exam_type
    (TYT/AYT/YDT, yoksa TYT). Her (date, publisher, exam_type) bir deneme sayılır; TYT'de 4
    dersin hepsi, AYT/YDT'de öğrencinin alanının (`field`) testleri olmalı.
    Hata varsa ilk hatayla `ValidationError` fırlatır.
    """
    subjects = subjects if subjects is not None else load_subjects()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")  # Excel'in eklediği BOM'u da kabul et
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValidationError("CSV dosyası okunamadı. UTF-8 ve virgülle ayrılmış olmalı.") from exc

    missing = set(EXAM_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValidationError(f"CSV'de eksik sütun(lar): {sorted(missing)}")

    if "exam_type" not in df.columns:
        df["exam_type"] = "TYT"
    df = df[[*EXAM_CSV_COLUMNS, "exam_type"]].copy()
    df["exam_type"] = df["exam_type"].fillna("TYT").astype(str).str.strip().str.upper()
    bad_types = set(df["exam_type"]) - set(EXAM_TYPES)
    if bad_types:
        raise ValidationError(f"Bilinmeyen sınav türü: {sorted(bad_types)} (TYT, AYT veya YDT olmalı)")
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    except (ValueError, TypeError) as exc:
        raise ValidationError("Tarihler YYYY-AA-GG biçiminde olmalı.") from exc

    for col in ("correct", "wrong", "blank"):
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ValidationError(f"'{col}' sütunu tam sayı olmalı.")

    question_counts = dict(zip(subjects["name"], subjects["question_count"]))
    unknown = set(df["subject"]) - set(question_counts)
    if unknown:
        raise ValidationError(f"Bilinmeyen ders(ler): {sorted(unknown)}")

    for row in df.itertuples(index=False):
        validate_counts(
            row.correct, row.wrong, row.blank, question_counts[row.subject],
            label=f"{row.date} {row.subject}",
        )

    for (date, publisher, exam_type), group in df.groupby(["date", "publisher", "exam_type"]):
        label = f"{date} {publisher} ({exam_type})"
        expected = set(required_subjects(subjects, exam_type, field))
        present = list(group["subject"])
        if len(present) != len(set(present)):
            raise ValidationError(f"{label}: aynı ders birden fazla kez girilmiş.")
        extra = set(present) - expected
        if extra:
            raise ValidationError(f"{label}: bu denemeye ait olmayan ders(ler) {sorted(extra)}.")
        if set(present) != expected:
            raise ValidationError(f"{label}: eksik ders(ler) {sorted(expected - set(present))}.")

    return df.sort_values(["date", "publisher", "exam_type", "subject"]).reset_index(drop=True)
