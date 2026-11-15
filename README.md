# OptiStudy Coach

YKS'ye hazırlanan öğrenci için istatistik tabanlı kişisel koç. Deneme sonuçlarını "bu hafta ne çalışmalıyım?" sorusunun cevabına çevirir.

**Özellikler:** TYT ve alan denemeleri (AYT/YDT) girişi, gelişim paneli, net tahmini, TYT ve alan (SAY/EA/SÖZ/DİL) sıralama tahmini, haftalık konu önceliği, hedef takibi, açık/koyu temalı arayüz.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Örnek veriyi yükle

```bash
python scripts/seed_sample.py            # optistudy.db oluşturur
python scripts/seed_sample.py --reset    # sıfırdan yeniden oluşturur
```

## CSV biçimleri

Deneme sonuçları. `exam_type` opsiyonel (TYT/AYT/YDT, yoksa TYT). TYT denemesinde 4 dersin hepsi,
AYT/YDT denemesinde öğrencinin alanının tüm testleri olmalı (SAY: AYT Matematik, Fizik, Kimya, Biyoloji;
EA: Edebiyat, Tarih-1, Coğrafya-1, AYT Matematik; SÖZ: Edebiyat, Tarih-1, Coğrafya-1, Tarih-2, Coğrafya-2,
Felsefe Grubu, Din Kültürü; DİL: Yabancı Dil):

```csv
date,publisher,exam_type,subject,correct,wrong,blank
2026-09-06,Yayınevi A,TYT,Türkçe,29,5,6
2026-09-07,Yayınevi A,AYT,AYT Matematik,18,8,14
```

Konu sonuçları (opsiyonel, deneme önceden kayıtlı olmalı; konu adları `data/reference/topics.csv` ile aynı):

```csv
date,publisher,subject,topic,correct,wrong,blank
2026-09-06,Yayınevi A,Fen,Fizik,3,2,1
```

## Gerçek veriye geçiş

Örnek veriyle denedikten sonra **Deneme girişi → Veri yönetimi → Sıfırdan başla** ile boş bir başlangıç yap.
Hiçbir şey silinmez: mevcut veritabanı `optistudy-yedek-<tarih-saat>.db` adıyla proje klasöründe kalır; alanın ve
diploma notun yeni veritabanına aktarılır. `.db` dosyaları `.gitignore`'da olduğu için gerçek verin repoya girmez.

## Uygulama

```bash
streamlit run app.py
```

## Testler

```bash
pytest -q
```

## Klasörler

| Klasör | İçerik |
| --- | --- |
| `optistudy/` | Çekirdek paket: `db.py`, `data.py`, `scoring.py`, `analysis.py`, `charts.py` |
| `app.py`, `pages/` | Streamlit arayüzü |
| `data/reference/` | Ders listesi (TYT soru sayıları), konu listesi şablonu |
| `data/sample/` | Tamamen sahte deneme verisi |
| `scripts/` | Yardımcı scriptler |
| `tests/` | pytest testleri |
| `legacy/` | Eski R Shiny prototipi (sadece referans) |
