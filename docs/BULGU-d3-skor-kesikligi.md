# Bulgu — Eşik Platolarının Skor Kesikliği (Deney D3)

> **Birleşik karar: KISMİ (genişlik).** Geniş düz bölgeler gerçek ve
> açıklandı; büyük metrik sıçramaları **açıklanmadı**.
> **Ön-kayıt:** `docs/deneyler/DENEY-D3-Esik-Platolari-Skor-Kesikligi.md` (dondurulmuş, revize edilmedi)
> **Kaynak:** `attack/dataset.csv` · **Hedef:** `macyste-20260726T113328Z`
> **İlgili:** `docs/bulgular/BULGU-lodo-v04-ERRATA-01.md` §5 ve §9, `docs/bulgular/BULGU-d2-gorulmemis-kategori.md`
> **Tarih:** 27 Temmuz 2026

D2 görülmemiş kategoriyi eledikten sonra kalan aday buydu: skor dağılımının
kendi kesikliği. D3 bunu iki ayrı olguya bölerek sınadı — **plato genişliği**
ve **sıçrama yüksekliği**. Biri desteklendi, öbürü desteklenmedi.

---

## 1. Deney

| | |
|---|---|
| Şema | yalnız `physical_proxy` |
| Seed evreni | 42–61, 20 seed'in tamamı |
| Karar aralığı | `I = [0.1238, 0.8795]` (Errata §9'daki tam eşik aralığı) |
| Ağaç paneli | iç içe `T ∈ {25, 50, 100, 200}` |
| Skor eşitliği | IEEE-754 `float64` **bit düzeyinde**; `±ε` komşuluğu yok |
| Değerlendirilebilirlik | `K_I >= 10`, sonlu plato sınırları, `τ ∈ I` |

Dokuz seed'in tarihsel üç bandı yalnız betimsel; karara girmedi. Bütün
durdurma koşulları (kanonik evren, bit düzeyinde tekrar, `T=200` iç içe
eşitliği, `K_hard <= T+1`, `n_jobs`) **tetiklenmedi**.

D3 hiçbir performans metriği (FPR, recall, alarm yükü) üretmedi; mevcut
guard'ları yalnız ham skor ve ağaç-yaprak çıktısına erişmek için aştı.

---

## 2. Skor desteği ne kadar seyrek?

`T = 200`, gerçek yaprak olasılıkları, 20 seed, ortanca [min, max]:

| büyüklük | değer |
|---|---|
| ayrı skor sayısı `K` | 17 [15, 21] |
| `I` içindeki ayrı skor `K_I` | 11 [7, 12] |
| **etkin** destek `K_eff` | **7.37 [6.15, 7.93]** |
| `I` içindeki satır `N_I` | 16 956 [15 035, 21 715] |
| saf yaprak ziyaret payı | 0.7209 [0.7049, 0.7325] |
| ortalama yaprak saflığı | 0.9085 [0.8938, 0.9482] |

**34 949 hedef satır, etkin olarak ~7 ayrı skor değerine düşüyor.** Bu, eşik
kararsızlığının yapısal zeminidir: bu kadar kaba bir destekte eşiğin nereye
düştüğü kaçınılmaz olarak kabadır.

---

## 3. H1 — plato genişliği: **DESTEK**

> Seçilen eşikler, skor desteğinin ortalama boşluğundan daha geniş platolara
> düşer.

| | |
|---|---|
| değerlendirilebilir seed | **14 / 20** (asgari 10) |
| ortanca `gap_enrichment` | **5.58** (destek eşiği ≥ 2.0) |
| `> 1` payı | **0.86** (destek eşiği ≥ 0.75) |

OpenPLC validation'da %1 FPR'ye kalibre edilen eşik, MaCySTe skor
desteğinin ortalama boşluğundan **~5.6 kat geniş** düz bölgelere düşüyor.
Plato genişliği olgusu gerçektir ve skor desteğinin seyrekliğiyle açıklanır.

**Dışarıda kalan altı seed'in tamamı `K_I < 10` nedeniyle dışarıdadır**
(`K_I` 7–9); hepsinde sınırlar sonlu ve `τ ∈ I`. Yani kapı yalnız çözünürlüğü
yetersiz seed'leri ayıkladı, sonuca göre seçim yapmadı.

> **Kapı seçiminin sonucu.** İlk taslaktaki `K_I >= 20` kapısıyla 20 seed'in
> **tamamı** elenir ve D3 hiçbir şey ölçmeden `underpowered` dönerdi. Kapı
> koşudan önce, bilinen `K` büyüklüğüne göre düzeltilmişti.

---

## 4. H2 — sıçrama yüksekliği: **KARIŞIK** (destek sayılmaz)

> Plato sınırları, eşit-kütle tabanından daha ağır nokta kütleleri taşır.

| | |
|---|---|
| değerlendirilebilir seed | 13 / 20 |
| ortanca `mass_enrichment` | 2.85 (eşiği geçiyor) |
| destek payı (`>1` **ve** `n(R) ≥ 50`) | **0.62** (gerekli ≥ 0.75) |

Ortanca eşiği geçiyor fakat **pay geçmiyor**; ön-kayıt gereği bu `karışık`tır
ve **destek sayılmaz**.

Dağılım çift tepeli: sekiz seed'de sınır gerçekten ağır, beş seed'de neredeyse
boş.

| seed | `mass_enrichment` | `n(R)` |
|---|---:|---:|
| 55 | 0.0019 | 3 |
| 57 | 0.0091 | 18 |
| 49 | 0.0897 | 177 |
| 46 | 0.0978 | 177 |
| 61 | 0.1253 | 177 |

Bir seed'de plato sınırında yalnız **3 satır** var. Böyle bir sınırın geçilmesi
metrikte kayda değer bir sıçrama üretemez.

> **Payda uyarısı.** §2'deki genel `mass_enrichment` yayılımı (ortanca 0.52)
> ile buradaki 2.85 **farklı topluluklardır**: ilki sınırı `I` içinde kalan tüm
> seed'ler, ikincisi H2'nin değerlendirilebilirlik koşullarını sağlayan 13
> seed. Biri diğerinin yerine kullanılamaz.

---

## 5. H3 — ağaç sayısı: **KARIŞIK**

> Ağaç sayısı arttıkça etkin destek artar ve plato genişliği azalır.

| yön | ortanca Spearman | ölçüt |
|---|---:|---|
| `rho_support` (T ↔ `K_eff`) | **1.00** | ≥ 0.80 ✓ |
| `rho_width` (T ↔ `w(τ)`) | **0.00** | ≤ −0.80 ✗ |

İki yönü birden sağlayan seed payı 0.25. `rho_width` seed'ler arasında
`[−1.00, +1.00]` aralığında dağılıyor — tutarlı bir yön yok.

Ağaç paneli (ortanca):

| `T` | `K` (gerçek) | `K_eff` (gerçek) | `K` (sert-oy) | `K_eff` (sert-oy) | `w(τ)` |
|---:|---:|---:|---:|---:|---:|
| 25 | 13 | 6.19 | 10.5 | 5.02 | 0.3241 |
| 50 | 14 | 7.13 | 12.0 | 5.98 | 0.3135 |
| 100 | 16 | 7.18 | 14.0 | 6.01 | 0.3488 |
| 200 | 17 | 7.37 | 15.5 | 6.18 | 0.3478 |

**Ağaç sayısını sekiz kat artırmak etkin desteği 6.19'dan 7.37'ye taşıyor —
ve seçili platoyu daraltmıyor.** Pratik sonuç: eşik kararsızlığı daha çok
ağaçla çözülmez.

---

## 6. H4 — yaprak saflığı: **YANLIŞLANDI**

> Gerçek yaprak olasılıkları, sert oy skorundan belirgin biçimde daha geniş
> etkin destek üretir.

| | |
|---|---|
| ortanca `K_eff_soft / K_eff_hard` | **1.19** |
| `> 1` payı | 1.00 |
| `> 1.25` payı | **0.20** (yanlışlama eşiği ≤ 0.50) |

Yirmi seed'in **tamamında** oran 1'in üzerinde — yani saf olmayan yapraklar
desteği gerçekten genişletiyor. Ama büyüklük küçük: ortanca %19, ön-kayıtlı
destek eşiği ise ≥ 2.0 idi. Ön-kayıt gereği H4 **yanlışlanmıştır**.

Yorum: skor kesikliğinin baskın kaynağı **ağaç oyu kafesidir**, yaprak
saflığı değil. Saf yaprak ziyaret payının 0.72 olması bununla tutarlıdır.

---

## 7. Birleşik karar

| H1 | H2 | sonuç |
|---|---|---|
| destek | karışık | **kısmi (genişlik)** |

Geniş düz bölgeler var ve skor desteğinin seyrekliğiyle açıklanıyor. Ancak
errata §5'teki **otuz kat** metrik sıçramaları için ağır sınır blokları
gerekir; bu tutarlı biçimde gözlenmedi. Mekanizmanın yarısı kapandı.

**Kanıtladığı:**

- Hedef skor desteği aşırı seyrek: 34 949 satır, etkin ~7 ayrı skor.
- Kaynak alandan gelen eşikler, ortalama boşluğun ~5.6 katı geniş platolara
  düşüyor (14 seed, pay 0.86).
- Kesikliğin baskın kaynağı ağaç oyu kafesi; yaprak saflığı katkısı ~%19.
- Ağaç sayısını artırmak etkin desteği büyütüyor ama platoyu daraltmıyor.

**Kanıtlamadığı:**

- Büyük metrik sıçramalarının nedeni. Sınır kütlesi tutarlı biçimde ağır
  değil (H2 karışık).
- Eşik kalibrasyonunun **neden** farklı OpenPLC koşularıyla değiştiği. D3 skor
  yapısını ölçer, kalibrasyonun kaynak-tarafı değişkenliğini değil.
- Model başarımı veya yanlış-alarm etkisi. D3 performans metriği üretmedi.
- "Ağaç sayısı neden oldu" / "yaprak saflığı neden oldu" gibi tekil nedensel
  cümleler. D3 kontrollü skor dönüşümleri ölçer, biçimsel aracılık analizi
  yapmaz.

---

## 8. Tez için sonuç

Errata §5'in "eşik kalibrasyonu da transfer etmiyor" sonucu artık **kısmen
yapısal bir zemine** oturuyor: hedef alandaki skor desteği o kadar seyrek ki
kaynaktan gelen bir eşik kaçınılmaz olarak geniş bir boşluğa düşüyor. Bu,
kararsızlığı gizemli bir gözlem olmaktan çıkarır.

Tezde kullanılacak iki cümle:

> Hedef alandaki karar skoru etkin olarak ~7 ayrı değere düşer; kaynak alanda
> kalibre edilen eşik, bu desteğin ortalama boşluğundan yaklaşık **5.6 kat**
> geniş düz bölgelere iner.

> Bu seyreklik plato **genişliğini** açıklar; plato sınırlarındaki metrik
> **sıçramalarının** büyüklüğünü açıklamaz ve ağaç sayısını artırmakla
> giderilmez.

Uygulamaya dönük not: sabit bir olasılık eşiğiyle kalibrasyon, bu kadar kaba
skor desteğinde alanlar arasında güvenilir değildir. Bu bir D3 sonucu değil,
D3'ün ölçtüğü destek yapısının doğrudan sonucudur; ayrıca sınanması gerekir.

---

## 9. Yeniden üretim

```bash
python -m ml.score_discreteness_diagnostic
```

Çıktı `ml/ml_out/d3_score_discreteness.json` (bütün nokta kütlesi envanteri,
skor boşlukları ve `float.hex()` değerleriyle). Testler:
`tests/test_d3_score_discreteness.py`.

Ön-kayıt koşudan sonra **revize edilmedi**. Koşu öncesi denetimde bulunan dört
madde (`K_I` kapısı, §1.1 körlük beyanı, §3.3 plato tanımı, `n_jobs`
invaryantı) dondurma **öncesinde** düzeltilmişti; bkz. commit `c03346b`.
