# Referans verisi

- `subjects.csv`: TYT dersleri ve soru sayıları.
- `topics.csv`: TYT konu listesi ve ağırlıklar (`weight` = TYT'de o konudan gelen soru sayısı).
  - Sosyal ve Fen ağırlıkları resmidir: ÖSYM 2026-YKS Kılavuzu Tablo 1A (Tarih/Coğrafya/Felsefe/Din 5'er; Fizik 7, Kimya 7, Biyoloji 6).
  - Türkçe ve Matematik için ÖSYM konu bazlı soru sayısı yayımlamaz; ikincil kaynaklar da birbiriyle çelişir. Bu yüzden `weight` **bilerek boş** bırakıldı. Uygulama bu konularda ağırlığı öğrencinin kendi denemelerindeki ortalama soru sayısından hesaplar. Uydurma ağırlık girmeyin; güvenilir bir kaynak bulunursa buraya kaynağıyla girilebilir.
- `exam_dates.csv`: Sınav tarihleri. Kaynak: ÖSYM'nin resmi yıllık sınav takvimi (PDF). Her yıl yeni takvim açıklanınca satır ekleyin.
- `tyt_test_stats.csv`: TYT testlerinin ortalama ve standart sapmaları (son sınıf ve tüm adaylar) ile puan ağırlıkları. Kaynak: ÖSYM 2026-YKS Sayısal Bilgiler, s. 3; ağırlıklar 2026-YKS Kılavuzu Tablo 1C.
- `rank_reference.csv`: "X puan ve üstü aday sayısı" tablosu (yığılımlı dağılım). `rank` = o puan ve üstünü alan aday sayısı, yani o puanın yaklaşık başarı sırası. `TYT` = sınav puanı, `Y-TYT` = yerleştirme puanı (OBP eklenmiş). Kaynak: ÖSYM 2026-YKS Sayısal Bilgiler, s. 12–13. Tablo 20 puanlık aralıklarla verildiği için sıralama ara değerlerde doğrusal interpolasyonla yaklaşık bulunur.

Her yıl ÖSYM sayısal bilgileri açıklanınca yeni yılın satırları eklenir; uygulama en güncel yılı kullanır.

## AYT / YDT

- `subjects.csv`: `exam_type` sütunu dersin oturumunu (TYT/AYT/YDT) gösterir. AYT alt testleri ayrı ders olarak tutulur (Edebiyat 24, Tarih-1 10, Coğrafya-1 6, Tarih-2 11, Coğrafya-2 11, Felsefe Grubu 12, Din Kültürü 6, AYT Matematik 40, Fizik 14, Kimya 13, Biyoloji 13; YDT Yabancı Dil 80). Kaynak: 2026-YKS Kılavuzu Tablo 1B.
- `ayt_test_stats.csv`: AYT/YDT testlerinin geçerli soru sayısı, son sınıf ortalama/standart sapması, tüm adaylar ortalaması ve aday sayısı. Kaynak: 2026-YKS Sayısal Bilgiler, s. 3. Yabancı Dil için İngilizce değerleri kullanılır.
- `score_weights.csv`: SAY/EA/SÖZ/DİL puan türlerinin test ağırlıkları (Kılavuz Tablo 1E). TYT'nin %40'lık payı, Tablo 1C oranlarıyla (33/17/33/17) dört TYT testine dağıtılmıştır.
- `rank_reference.csv`: SAY/SÖZ/EA/DİL sınav puanı ve `Y-` önekli yerleştirme puanı dağılımları (Sayısal Bilgiler s. 12–13).
- `exam_dates.csv`: AYT ve YDT 20.06.2027 (ÖSYM 2027 takvimi).
- `topics.csv`: AYT konuları. Tek konulu alt testlerin (Tarih-1, Coğrafya-1, Tarih-2, Coğrafya-2, Felsefe Grubu, Din Kültürü) ağırlığı alt testin resmi soru sayısıdır; diğer AYT konularının ağırlığı ÖSYM yayımlamadığı için bilerek boştur.
