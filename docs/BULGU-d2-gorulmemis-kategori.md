# Bulgu — Görülmemiş Kategori Teşhisi (Deney D2)

> **Sonuç: encoding uyuşmazlığı gerçek; kütlesi ölçüldü.** **Tüm** MaCySTe hedef
> satırlarının **%0.58'i** görülmemiş kategori taşıyor. Bu bir toplam kütle
> ölçüsüdür; saldırı satırları veya tek tek senaryolar içindeki payı **ayrıca
> ölçülmemiştir** (bkz. §5).
> **Tabakalı dağılım (D2A, §5):** kütle saldırı satırlarında **seyreliyor**
> (zenginleşme 0.10); toplamın %92.5'i saldırı-dışı, %73'ü `normal` senaryoda.
> **Ön-kayıt:** `docs/deneyler/DENEY-D2-Gorulmemis-Kategori-Teshisi.md` (R1–R3 dâhil) ve
> `docs/deneyler/DENEY-D2A-Tabakali-Gorulmemis-Kutle.md`
> **Kaynak:** `attack/dataset.csv` · **Hedef:** `macyste-20260726T113328Z`
> **İlgili:** `docs/bulgular/BULGU-lodo-v04-cross-testbed.md`, `docs/bulgular/BULGU-lodo-v04-ERRATA-01.md`
> **Tarih:** 26 Temmuz 2026

Bu belge, v0.4 bulgusunun izleyen iş listesindeki 3. maddeyi kapatır: sessiz
encoding uyuşmazlığını **ölçülen bir büyüklüğe** çevirmek. Sonuç, mekanizma
anlatısının bir bölümünü zayıflatmaktadır ve bu hâliyle raporlanır.

---

## 1. Deney

| | |
|---|---|
| Kaynak alan | OpenPLC, `attack/dataset.csv` — 57 499 satır, 20 koşu |
| Hedef alan | MaCySTe, `macyste-20260726T113328Z/*/events-v0.4.csv` — 34 949 satır, 12 koşu |
| Feature yolu | `ml/lodo_generalization.py::build_feature_frame()`, değişiklik yok |
| Şema | `flow`, `protocol`, `physical_strict`, `physical_proxy` |
| Birincil panel | bölmeden **bağımsız**: seed, train/validation bölmesi, model skoru ve eşik **yok** |
| İkincil panel | 42–61 (20 seed), sözlük yalnız train parçasından |
| Uygulama | `ml/unseen_category_diagnostic.py`, `ml/unseen_category_mechanism.py` |

`handle_unknown="ignore"` **değiştirilmedi**; teşhis model hattına paralel,
salt-okunur bir gözlemcidir. Dondurulmuş v0.4 sonuçları yeniden hesaplanmadı.

---

## 2. Birincil sonuç (bölmeden bağımsız)

Sözlük tüm OpenPLC frame'inden, karşılaştırma tüm MaCySTe frame'ine karşı.
Errata'daki bölme kararsızlığı bu panele **taşınamaz**.

| şema | kolon | \|V_src\| | \|V_tgt\| | \|U\| | çeşit açığı | satır kütlesi | kütle (gözlenen) |
|---|---|---:|---:|---:|---:|---:|---:|
| `flow` | `flow_state` | 4 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `protocol` | `modbus_function` | 7 | 4 | 1 | **0.2500** | **0.0058** | **0.0080** |
| `protocol` | `modbus_access` | 7 | 4 | 1 | **0.2500** | **0.0058** | **0.0080** |
| `physical_strict` | `rudder_band` | 2 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `physical_proxy` | `propulsion_band` | 3 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 |

Görülmemiş değerlerin **tamamı** iki tanedir:

```text
modbus_function : 16.0
modbus_access   : WRITE | HOLDING | MULTIPLE | WRITE_MULTIPLE
```

Mutlak sayı: 34 949 hedef satırın **201'i**. Gözlenen payda (Modbus alanı dolu
olan satırlar) 25 006; oran orada %0.80. `protocol` şemasının birleşim kütlesi
de %0.58 — iki kolon **aynı** satırları işaretliyor.

**Çeşit ile frekans arasındaki uçurum bu bulgunun özüdür.** Hedefteki ayrı
değerlerin dörtte biri sözlük dışında (çeşit %25), ama bu değerler satırların
yalnız binde 6'sında geçiyor (kütle %0.58). Tek bir sayıyla raporlansaydı,
hangisinin seçildiğine göre iki farklı hikâye anlatılabilirdi.

---

## 3. Bölmeye duyarlılık: yok (H2 reddedildi)

42–61 seed evreninin **tamamında**, her şema × kolon için `max = min`. Sözlük
teşhisi bölmeden etkilenmiyor.

Bu, errata §5 ile doğrudan karşıtlık oluşturur: aynı seed evreninde **eşik**
0.29–0.88 arasında geziniyor ve eşiğe bağlı metrikler 30 kat oynuyordu. Sözlük
kapsaması ise hiç kıpırdamıyor. Yani bu iki büyüklük farklı türden: biri
kalibrasyona, öbürü alan çiftinin yapısına ait.

Guard sınıfları (betimsel) errata §3'ün tablosunu **birebir** yeniden üretti:

| sınıf | n | seed'ler |
|---|---:|---|
| validation'da `fault` yok (Kusur A) | 8 | 42, 45, 49, 52, 54, 56, 58, 60 |
| train'de `fault` yok (Kusur B) | 3 | 43, 46, 48 |
| iki tarafta da var | 9 | 44, 47, 50, 51, 53, 55, 57, 59, 61 |

---

## 4. H3 mekanizma testi: test edilemedi

**Soru:** errata §5'teki eşik platoları, görülmemiş-kategori katmanlarıyla
açıklanabilir mi?

Ön-kayıtlı ölçüt (§7.4) sağlandı ve karar `desteklendi` olarak kaydedildi.
**Bu sonuç dejeneredir ve mekanizma kanıtı sayılamaz.**

Ölçüt "bloktaki satırların en az %80'i tek bir katmana ait olmalı" der, fakat
**hangi** katman olduğunu sormaz. §2'ye göre hedef satırların %99.4'ü zaten
"görülmemiş kategori taşımıyor" katmanındadır (`k = 0`). Dolayısıyla herhangi
bir blok taban oran gereği ölçütü sağlar — mekanizmadan bağımsız olarak.

Sonuçlar görüldükten sonra eklenen betimsel kontrol (karara **girmez**):

| geçiş | aralık | blokta görülmemiş pay | taban oran | zenginleşme |
|---|---|---:|---:|---:|
| T1 | (0.49, 0.72) | 0.00598 | 0.00575 | 1.04 |
| T2 | (0.74, 0.88) | 0.00000 | 0.00575 | 0.00 |

T1'de yığılma yok; T2 bloğunda görülmemiş satır **hiç yok**. Plato sınırındaki
skor nokta kütlelerinde görülmemiş kategoriler yoğunlaşmıyor.

**Karar taşınmadı.** `desteklendi` sonucu ön-kayıtlı hâliyle kayıtlıdır; ölçüt
sonuç görüldükten sonra değiştirilip yeni bir manşet üretilmemiştir. Düzeltilmiş
ölçüt yalnız gelecek koşular için, ayrı ön-kayıtla yazılmıştır (belge §10 R3).

**Doğru okuma:** H3 bu veride **test edilememiştir** — ne destek ne ret. Ölçüt
yanlış kurulmuştur.

Bloklardaki doğrudan gözlem ise ayrı bir olgudur ve ölçüte bağlı değildir:
plato sınırı nokta kütlelerinde görülmemiş satırların payı taban orandan
farksızdır (T1) ya da sıfırdır (T2). Bu, **incelenen bloklar için** görülmemiş
kategorilerle bir ilişki bulunmadığını söyler. Skor dağılımının tamamı için
genel bir çıkarım değildir.

### Yan gözlem — errata'nın plato tarifinin kapsamı

H3 için 20 seed'in tamamında `physical_proxy` eşikleri hesaplandı. Doğru çerçeve
dört maddedir:

1. Errata §5'te dokuz seed üzerinden bildirilen üç plato **gerçektir**.
2. Bu üç bant 20-seed evreninin tamamını **kapsamıyor**: eşikler
   **0.1238 – 0.8795** aralığında ve 5 seed (48, 52, 54, 55, 56) bantların
   dışında kalıyor.
3. Bu, errata'nın sonucunu **bozmaz** — plato *tarifinin* kapsamını daraltır.
   Geçersiz sayılan metrikler, guard sınıflandırması ve kalıcı çözüm koşulları
   aynen geçerlidir.
4. Eşik aktarım kararsızlığı ise tam aralıkla **daha da belirginleşir**:
   dokuz seed'de 0.2900–0.8795 olarak görülen yayılım, 20 seed'de 0.1238'e
   kadar iniyor.

Yani bu gözlem errata'yı zayıflatmaz; ana iddiasını (eşik kalibrasyonu
transfer etmiyor) güçlendirir.

---

## 5. Tabakalı dağılım (D2A)

Ön-kayıt: `docs/deneyler/DENEY-D2A-Tabakali-Gorulmemis-Kutle.md`. §2'nin toplam sonucu
görüldükten sonra, tabakalı sayılar hesaplanmadan önce donduruldu. Aşağıdaki
**14 tabakanın tamamı** yayımlanır; sıra ön-kayıtlıdır, büyüklüğe göre
sıralanmamıştır. Boş hücreler `n = 0` ve oran `null` olarak durur.

Değerler `modbus_function` kolonundandır. `modbus_access` **birebir aynıdır**
(iki kolon aynı 201 satırı işaretliyor); şema birleşimi de aynı sayıları verir.

`mass` = tabaka içi yaygınlık (tüm satır paydası) · `obs` = gözlenen satır
paydası · `alloc` = toplam görülmemiş kütlenin bu tabakaya düşen payı ·
`zengin` = tabaka kütlesi ÷ hedefin tamamındaki kütle (%0.58).

### Panel 1 — senaryo marjinleri

| tabaka | n_all | n_obs | n_unseen | mass | obs | alloc | zengin |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal` | 3 238 | 3 220 | 147 | 0.0454 | 0.0457 | 0.7313 | 7.89 |
| `fault` | 4 317 | 4 299 | 24 | 0.0056 | 0.0056 | 0.1194 | 0.97 |
| `manipulation` | 4 325 | 4 307 | 30 | 0.0069 | 0.0070 | 0.1493 | 1.21 |
| `recon` | 23 069 | 13 180 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 |

### Panel 2 — etiket marjinleri

| tabaka | n_all | n_obs | n_unseen | mass | obs | alloc | zengin |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attack` | 27 262 | 17 370 | 15 | 0.0006 | 0.0009 | 0.0746 | **0.10** |
| `non_attack` | 7 687 | 7 636 | 186 | 0.0242 | 0.0244 | 0.9254 | **4.21** |

### Panel 3 — senaryo × etiket

| tabaka | n_all | n_obs | n_unseen | mass | obs | alloc | zengin |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal × attack` | 0 | 0 | 0 | null | null | 0.0000 | null |
| `normal × non_attack` | 3 238 | 3 220 | 147 | 0.0454 | 0.0457 | 0.7313 | 7.89 |
| `fault × attack` | 0 | 0 | 0 | null | null | 0.0000 | null |
| `fault × non_attack` | 4 317 | 4 299 | 24 | 0.0056 | 0.0056 | 0.1194 | 0.97 |
| `manipulation × attack` | 4 208 | 4 199 | 15 | 0.0036 | 0.0036 | 0.0746 | 0.62 |
| `manipulation × non_attack` | 117 | 108 | 15 | 0.1282 | 0.1389 | 0.0746 | 22.29 |
| `recon × attack` | 23 054 | 13 171 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| `recon × non_attack` | 15 | 9 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 |

### Koşu duyarlılığı

Çapraz hücrelerin **tamamında** koşular arası yayılım ihmal edilebilir; örneğin
`normal × non_attack` için üç koşuda `0.0454 [0.0453, 0.0455]`. Hiçbir tabaka
tek bir koşu tarafından sürüklenmiyor.

Marjinlerde görülen geniş yayılım (`non_attack`: `0.0254 [0.0000, 0.1282]`,
12 koşu) kararsızlık **değil**, senaryo karışımıdır: aynı marjin hem `recon`
(sıfır) hem `manipulation × non_attack` (0.1282) koşularını içerir.

### Okuma

**Görülmemiş kategoriler saldırı satırlarında yoğunlaşmıyor — tersine
seyreliyor.** `attack` tabakasında kütle %0.06 ve zenginleşme **0.10**; yani
hedefin genelinden on kat *daha seyrek*. Toplam görülmemiş kütlenin
**%92.5'i** saldırı-dışı satırlarda.

Kütlenin ana yatağı `normal` senaryosudur: 201 satırın 147'si (alloc 0.73),
tabaka içi yaygınlık %4.5, zenginleşme 7.9. Bu, v0.4'ün mekanizma tarifindeki
"yazma-yoğun **normal** operasyon" ifadesiyle tutarlıdır — FC16 /
`WRITE_MULTIPLE`, MaCySTe'nin olağan yazma trafiğidir.

`recon`, hedef satırların **%66'sı** olmasına rağmen görülmemiş kategori
**hiç** taşımıyor.

**Manşete çekilmemesi gereken hücre:** `manipulation × non_attack` en yüksek
zenginleşmeye sahip (22.29) fakat tabaka **117 satır** ve yalnız **15**
görülmemiş satır içeriyor. Mutlak sayı bu kadar küçükken oran tek başına
raporlanamaz; tablodaki yerinde, paydalarıyla birlikte durur.

---

## 6. Ne kanıtlıyor, ne kanıtlamıyor

**Kanıtladığı:**

- OpenPLC sözlüğü MaCySTe'yi tam kapsamıyor; FC16 / `WRITE_MULTIPLE` sözlük
  dışında ve `handle_unknown="ignore"` bunu sessizce yutuyor.
- Uyuşmazlığın büyüklüğü ölçüldü: çeşit %25, kütle %0.58 (gözlenen %0.80).
- Sözlük kapsaması bölmeye **dayanıklı**; eşik kalibrasyonundan farklı bir
  büyüklük.

**Kanıtlamadığı:**

- Görülmemiş kategorilerin aktarım başarısızlığına **sebep olduğu ya da
  olmadığı**. D2 nedensellik ölçmez; H3 testi de sonuç vermedi.
- Sıfırlanan one-hot bloğunun herhangi bir satırı yanlış sınıflandırdığı. Bir
  kolon sıfırlansa da sayısal ve diğer kategorik özellikler modelde kalır.
- Uyuşmazlığın AUC, recall veya eşik platolarını açıkladığı. Eşik platolarının
  nedeni açık kalmıştır.
- Görülmemiş değeri kaldırmanın veya yeniden kodlamanın başarımı iyileştireceği.

**D2A'nın kapattığı boşluk.** §5 öncesinde "kütle saldırı satırlarında
yoğunlaşıyor olabilir" itirazı açıktı; bu ölçüldü ve **tersi** çıktı
(zenginleşme 0.10). Artık toplam kütlenin küçüklüğü, saldırı tarafında gizli
bir yoğunlaşmayla açıklanamaz.

> **Yazım kuralı.** Bu bulgudan "FC6/FC16 sınıflandırma hatasına **sebep**
> oluyor/olmuyor" biçiminde bir cümle **kurulamaz**. D2 ve D2A birlikte yalnız
> **birlikte görülme** ölçer: kütlenin büyüklüğü ve tabakalar arası dağılımı.
> Nedensel iddia için kodlama müdahalesi ve karşı-olgusal başarım
> karşılaştırması içeren ayrı bir ön-kayıt gerekir.
>
> "En yüksek", "yalnızca", "yoğunlaşmış" gibi ifadeler kullanılırsa yanlarında
> mutlak pay/payda, iki kütle oranı, `allocation` ve zenginleşme birlikte
> verilir. Özellikle `manipulation × non_attack` (117 satır) tek başına
> alıntılanamaz.

---

## 7. Tez için sonuç

v0.4 bulgusu mekanizmayı "FC6/FC16 uyuşmazlığı + yazma-yoğun normal operasyon"
olarak tarif ediyordu. D2, bu tarifin **birinci bileşenini nitel olmaktan
çıkarıp ölçtü**: uyuşmazlık gerçektir, sessizdir ve tüm hedef satırların
%0.58'ini kapsar.

D2A bu sayının nereye düştüğünü de ölçtü: kütle **saldırı satırlarında değil,
normal trafikte**.

Tezde kullanılacak cümle şudur:

> Uyuşmazlık, saldırıların kaçırılmasından çok, **normal operasyonların yanlış
> alarm üretmesi riskine işaret eder.**

"İşaret eder" ifadesi bilerek seçilmiştir: bu bir risk yönü tespitidir, ölçülmüş
bir yanlış-alarm etkisi **değildir**. Görülmemiş kategori taşıyan normal
satırların fiilen daha çok alarm üretip üretmediği ölçülmemiştir; ölçmek için
karşı-olgusal bir kurulum gerekir.

Yine de "bileşen açıklayıcı değildir" cümlesi **kurulamaz**: D2 ve D2A birlikte
birlikte görülme ölçer, nedensellik ölçmez.

Tezde D2 şu biçimde kullanılmalıdır:

1. Metodolojik katkı olarak: `handle_unknown="ignore"` ile çalışan her
   cross-domain kurulumda bu iki sayı (çeşit açığı ve satır kütlesi, iki
   paydayla) raporlanmalıdır. Sessiz başarısızlık ölçülebilir hâle gelir.
2. Ölçüm sonucu olarak: nitel bir gözlem (FC6/FC16) sayısallaştırıldı. Bu, o
   gözlemin **büyüklüğünü** verir; açıklayıcı gücü hakkında hüküm vermez.
3. Kapı A = Hayır kararı **etkilenmez**; D2 onu ne güçlendirir ne zayıflatır.

**Sıradaki iş:** eşik platolarının nedeni açık kaldı. Sonraki aday skor
dağılımının kendi yapısıdır (Random Forest'ın ayrık olasılık üretmesi, hedef
alanda az sayıda ayrı skor değeri). D2'den **ayrı** bir ön-kayıt gerektirir;
"plato", "nokta kütlesi" ve yanlışlama ölçütleri sonuçlara yeniden bakılmadan
tanımlanmalıdır.

---

## 8. Yeniden üretim

```bash
python -m ml.unseen_category_diagnostic
```

```bash
python -m ml.unseen_category_mechanism
```

```bash
python -m ml.unseen_category_strata
```

Çıktılar: `ml/ml_out/d2_unseen_category.json`, `ml/ml_out/d2_h3_mechanism.json`,
`ml/ml_out/d2a_unseen_category_strata.json`. Testler:
`tests/test_d2_unseen_category.py`, `tests/test_d2_h3_mechanism.py`,
`tests/test_d2a_strata.py`.

D2 ön-kaydı üç kez revize edildi (§10): R1 yanlış hedef dosya seti, R2 eksiklik
göstergesinin yanlış literali, R3 dejenere hizalanma ölçütü. Üçü de sonuç
üretilmeden **önce** ya da sonucu değiştirmeden kaydedilmiştir. D2A ön-kaydı
(`docs/deneyler/DENEY-D2A-Tabakali-Gorulmemis-Kutle.md`) revize edilmedi.
