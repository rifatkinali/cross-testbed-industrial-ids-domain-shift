# Bulgu — OpenPLC → MaCySTe Cross-Testbed LODO (v0.4)

> **Kapı A kararı: HAYIR.** Semantik transfer iddiası bu veriyle desteklenmiyor.
> **Ham veri:** `macyste-20260726T113328Z` · acquisition `v0.3.2-macyste-campaign`
> **Türetme:** `v0.4-macyste-labeling` · **Analiz:** `v0.3.1-lodo-contract`
> **Tarih:** 26 Temmuz 2026

Bu belge olumsuz bir sonucu raporlar. Yol haritası (`ANA-REFERANS.md` §5, Kapı A)
bu durumu öngörmüştür: *"Sonuç zayıfsa gizlenmez."*

---

## 1. Deney

| | |
|---|---|
| Eğitim | OpenPLC (`attack/dataset.csv`), koşu-bazlı train/validation |
| Eşik | yalnız OpenPLC validation, hedef FPR %1 |
| Test | MaCySTe, tamamen held-out, 12 koşu (4 senaryo × 3 tekrar × 90 s) |
| Şema | `flow`, `protocol`, `physical_strict`, `physical_proxy` |
| Seed | 42, 43, 44 |
| Contract guard | `ready: true`, `mode: headline` — `--force-headline` KULLANILMADI |

Kabul kontrolü: 81/82 geçti, 1 gerekçeli muafiyet
(`ACCEPTED-LIMITATIONS.json`: `python_base` imaj kimliği; `--no-build`
zorunluluğu Docker Hub oran sınırından). 20/20 kaynak hash'i ve tüm artifact
hash'leri yeniden hesaplanıp doğrulandı.

---

## 2. Sonuç

Aşağıdaki değerler **seed başına hesaplanan AUC'lerin ortalamasıdır** (n = 3
seed). Üç seed'in skorları önce ortalanıp sonra AUC alınırsa topluluk etkisiyle
bir miktar yüksek çıkar; o değerler §5'te ayrıca verilmiştir.

| şema | OpenPLC val AUC | AUC (olay-ağırlıklı) | AUC (senaryo-dengeli) | manipulation | recon |
|---|---:|---:|---:|---:|---:|
| `flow` | 0.954 | 0.322 | 0.394 | 0.502 | 0.289 |
| `protocol` | 0.976 | 0.319 | 0.390 | 0.492 | 0.282 |
| `physical_strict` | 0.981 | 0.329 | 0.385 | 0.465 | 0.299 |
| `physical_proxy` | 0.984 | **0.400** | **0.460** | **0.544** | **0.368** |

Eşik %1 FPR için kalibre edilmişti; MaCySTe'de gerçekleşen FPR %26–50.
`recall ≈ FPR` her şemada — sınıflandırıcı hedef alanda ayrım yapmıyor.
Episode recall 1.00 olması yalnızca satırların ~%27'sinin işaretlenmesinden
kaynaklanıyor; tespit değil.

**Model iç alanda güçlü** (val AUC 0.95–0.98): hat, özellik çıkarımı ve
etiketleme çalışıyor. Başarısızlık transfer aşamasında.

### Ayrıntı: iki saldırı senaryosu farklı davranıyor

- `manipulation` AUC ≈ 0.49–0.54 → **şans düzeyi**. Model bu saldırıyı hiç
  ayırt edemiyor.
- `recon` AUC ≈ 0.28–0.37 → **şansın altında**. Model tarama trafiğini normalden
  *daha az* şüpheli sıralıyor.

Olay-ağırlıklı AUC'nin 0.32'ye kadar düşmesi `recon`'un test setinin %66'sı
olmasından. Senaryo-dengeli ölçüm bunu düzeltiyor (0.39–0.46) ama sonuç hâlâ
şansın altında.

---

## 3. Mekanizma — neden ters dönüyor

```
OpenPLC (eğitim):   WRITE|HOLDING|SINGLE|WRITE_SINGLE  →  attack satırlarının %70'i
                                                          normal satırların %25'i
MaCySTe (test):     WRITE|HOLDING|MULTIPLE|WRITE_MULTIPLE  →  %0.1–5
                    READ|HOLDING                            →  %75
```

Üç etken birleşiyor:

1. **Encoding uyuşmazlığı.** OpenPLC saldırıları FC6 (tek register yazma)
   kullanıyor, MaCySTe FC16 (çoklu yazma). Modelin en güçlü saldırı işareti olan
   `WRITE_SINGLE` token'ı hedef alanda **hiç yok**. `handle_unknown="ignore"` onu
   sessizce sıfırlıyor — hata vermiyor, özellik yok oluyor.
2. **Normal operasyonun karakteri tersine dönüyor.** MaCySTe'de operatör sürekli
   dümen komutu yazıyor; yani "yazma = şüpheli" kuralı hedef alanda *normali*
   işaretliyor.
3. **`recon` salt okuma.** OpenPLC'de okumalar ağırlıkla normaldi. Saldırı
   etiketli `recon`, öğrenilmiş "okuma = normal" kuralına tam oturuyor → şansın
   altında AUC.

Senaryo bazında ortalama skorlar bunu doğruluyor (`protocol`):

```
normal 0.276 · manipulation 0.271 · fault 0.270 · recon 0.155
```

Üç yazma-tabanlı senaryo birbirinden ayrılamıyor; saldırı etiketli `recon` en
düşük skoru alıyor.

---

## 4. Fiziksel özellikler neden kurtarmıyor

`physical_strict` sonuçları `protocol` ile pratikte aynı. Ölçülen sebep:
**fiziksel rol olayların yalnızca %8'inde gözlenebiliyor**
(`rudder_norm_present` = 1 olan satır: 2 896 / 34 949). Geri kalan %92'de değer
imputed 0 ve `present = 0`; ağaç toplulluğunun kararını protokol kategorileri
belirliyor.

`physical_proxy` her ölçümde en az kötü olan (dengeli 0.460, manipulation 0.544)
— semantik proxy bir miktar yardım ediyor, ama %8 kapsama ile domain kaymasını
telafi etmeye yetmiyor.

**Bu, fiziksel-semantik yaklaşımın çürütülmesi değildir.** Gösterdiği şey: bu
testbed çiftinde fiziksel rolün olay-başına gözlenebilirliği, olay-düzeyi bir
model için fazla seyrek. Daha yoğun proses örneklemesi ya da olay-düzeyi yerine
pencere-düzeyi bir gösterim ayrı bir deneyin konusudur.

---

## 5. Belirsizlik — dürüst ifade

Koşu-bazlı tabakalı bootstrap %95 aralıkları **çok dar** (genişlik < 0.01) ve
**genelleme aralığı olarak okunamaz.**

Tekrar bazında ölçülen gerçek yayılım (3 seed'in skor ortalaması üzerinden;
her saldırı koşusu tüm normal satırlara karşı):

| şema | manipulation | yayılım | recon | yayılım |
|---|---:|---:|---:|---:|
| `flow` | 0.5017 | 0.00000 | 0.2889 | 0.00000 |
| `protocol` | 0.4917 | 0.00020 | 0.2826 | 0.00020 |
| `physical_strict` | 0.4652 | 0.00020 | 0.2995 | 0.00040 |
| `physical_proxy` | 0.5837 | 0.00030 | 0.3675 | 0.00040 |

Üç tekrarın yayılımı her yerde **0.0004'ten küçük**.

`physical_proxy` manipulation değerinin §2'deki 0.544'ten yüksek çıkması
(0.584) topluluk etkisidir: üç seed'in skorları ortalanınca tek seed
ortalamasından daha iyi ayrım oluyor. İki değer farklı büyüklüklerdir ve
karıştırılmamalıdır — §2 tek-seed ortalamasıdır, headline sonuçla tutarlı olan
odur.

Sebep: tekrarlar betiklenmiş bir senaryonun neredeyse birebir kopyası. Contract
guard'ın "≥3 tekrar" şartı burada **yeniden üretilebilirliği** doğruluyor,
**genellemeyi** değil.

Asıl belirsizlik nitel ve bu aralıkların kapsamadığı yerde:

- **Tek testbed çifti** (OpenPLC → MaCySTe). Başka bir hedef alanda sonuç farklı
  olabilir.
- **Tek saldırı uygulaması.** `manipulation` tek bir setpoint manipülasyonu;
  `recon` tek bir tarama deseni.
- **Tek istemci betiği.** Trafik deterministik; gerçek operatör davranışının
  değişkenliğini içermiyor.
- **MaCySTe'nin çalışan alt kümesi** upstream'in tamamı değil (bkz.
  `docs/entegrasyon/MACYSTE-ENTEGRASYON.md`).

Bu yüzden "cross-testbed genelleme başarısız" ifadesi **bu testbed çifti ve bu
saldırı uygulaması için** geçerlidir; genel bir imkânsızlık iddiası değildir.

---

## 6. Ne söylenebilir, ne söylenemez

**Söylenebilir:**

- Testbed'e özgü protokol/encoding özellikleri bu testbed çiftinde transfer
  etmiyor ve **neden** transfer etmediği somut bir mekanizmayla gösterilmiştir
  (FC6/FC16 uyuşmazlığı + yazma-yoğun normal operasyon).
- Görülmemiş kategori değerlerinin sessizce yutulması (`handle_unknown="ignore"`)
  bu başarısızlığı **gözden kaçırılabilir** hâle getiriyor; encoding uyuşmazlığı
  bir uyarı üretmiyor.
- Fiziksel-semantik proxy, protokol özelliklerinden **tutarlı biçimde daha iyi**
  transfer ediyor (dengeli AUC 0.460 vs 0.390) — küçük ama dört şemada ve üç
  seed'de istikrarlı bir fark.
- Ürün konumlandırması: "genel model" değil, **gemi/testbed'e özel kalibrasyon
  ve transfer teşhis aracı**.

**Söylenemez:**

- "Fiziksel-semantik özellikler işe yaramaz." Kapsama %8; yöntem test edilmedi,
  veri yeterli değildi.
- "Cross-vendor genelleme başarısız." Tek testbed çifti; vendor iddiası zaten
  kapsam dışı (`ANA-REFERANS.md` §1).
- "Model kötü." İç alanda val AUC 0.98; sorun modelde değil, domain kaymasında.

---

## 7. Bundan sonra

1. **Bu bulguyu dondur.** Tez ana bulgusu budur; olumsuz sonuç yayınlanır.
2. **Deney D1** (senaryo-dengeli kampanya) — ön-tanımlı duyarlılık deneyi,
   bkz. `docs/deneyler/DENEY-D1-Senaryo-Dengeli-Kampanya.md`. Ana sonucun yerine geçmez.
3. **Görülmemiş kategori teşhisi.** `handle_unknown="ignore"` yerine, hedef
   alanda karşılığı olmayan kategori oranını ölçüp raporlayan bir tanı adımı
   eklenmeli. Bu, encoding uyuşmazlığını sessiz bir başarısızlıktan **ölçülen
   bir büyüklüğe** çevirir ve tezin katkılarından biri olabilir.
4. **Pencere-düzeyi gösterim.** Fiziksel rolün %8 kapsaması olay-düzeyi model
   için az; sabit zaman pencerelerinde toplanmış fiziksel özellikler ayrı bir
   deney olarak denenebilir (aynı zamanda flow sayaçları sorununu da çözer,
   bkz. `ml/label_phases.py` TODO).
