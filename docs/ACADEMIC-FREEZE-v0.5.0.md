# Akademik Freeze v0.5.0 — Tez Ana Sonuç Paketi

> **Durum:** Yerel reprodüksiyon kapısı
>
> **Ana karar:** Kapı A = HAYIR
>
> **Analiz:** OpenPLC → MaCySTe, LODO, seed 42/43/44, hedef FPR %1

Bu paket yeni bir model veya yeni bir dataset eklemez. Dondurulmuş ana LODO
sonucunu ve onu açıklayan D2/D3 teşhislerini tek çalışma noktasından yeniden
üretir.

## Tek komut

```powershell
python scripts/freeze_academic_results.py
```

Komut:

1. akademik sözleşme testlerini çalıştırır;
2. OpenPLC dataset snapshot hash'lerini doğrular;
3. MaCySTe kampanyasındaki ham artifact ve türetilmiş CSV hash'lerini doğrular;
4. 12 koşulu sabit sırada 34.949 satırlık hedef tabloya birleştirir;
5. dört feature şeması için ana LODO tablosunu yeniden hesaplar;
6. D2, D2-H3, D2A ve D3 teşhislerini yeniden çalıştırır;
7. sonuç JSON'larını, tez tablosunu, SVG şekli ve makine-okunur manifesti
   `academic_release/v0.5.0/` altında üretir.

## Tezde kullanılacak ana sonuç

| Şema | OpenPLC val AUC | MaCySTe olay AUC | MaCySTe dengeli AUC | Manipulation | Recon |
|---|---:|---:|---:|---:|---:|
| `flow` | 0.9543 | 0.3220 | 0.3940 | 0.5017 | 0.2889 |
| `protocol` | 0.9759 | 0.3186 | 0.3895 | 0.4917 | 0.2824 |
| `physical_strict` | 0.9813 | 0.3289 | 0.3845 | 0.4649 | 0.2994 |
| `physical_proxy` | 0.9842 | 0.4001 | 0.4595 | 0.5440 | 0.3675 |

Kaynak alanda AUC 0,95–0,98 iken hedef alanda olay-ağırlıklı AUC
0,32–0,40'tır. `physical_proxy` göreli olarak en iyi aktarımı sağlasa da
hedef alanda güvenilir ayrım kurmaz. Bu nedenle semantik transfer iddiası bu
testbed çifti için desteklenmez.

## Mekanizma zinciri

- D2: hedefte 201 satırda görülmemiş FC16/`WRITE_MULTIPLE` kategorisi vardır
  (tüm hedefin %0,5751'i; ilgili gözlenen Modbus satırlarının %0,8038'i).
- D2-H3: ön-kayıtlı ölçüt biçimsel olarak sağlansa da sonuç dejeneredir ve
  mekanizma kanıtı sayılmaz.
- D2A: görülmemiş kütle 14 ön-kayıtlı tabakanın tamamında yayımlanır.
- D3: hedef skor desteği seyrektir; geniş plato bulgusu desteklenir, büyük
  sıçrama açıklaması karışıktır. Birleşik karar `kismi_genislik`tir.

## İddia sınırı

Söylenebilir:

> Bu OpenPLC–MaCySTe testbed çiftinde protokol/encoding ve operasyon dağılımı
> kayması aktarımı bozmuştur; fiziksel proxy göreli iyileşme sağlamış fakat
> güvenilir hedef-alan ayrımı üretmemiştir.

Söylenemez:

- fiziksel-semantik özellikler genel olarak işe yaramaz;
- cross-vendor genelleme sınanmıştır;
- sonuç gerçek gemiye veya filoya genellenir;
- dar koşu-bootstrap aralıkları genel belirsizliği temsil eder.

## Taşınabilirlik durumu

Kaynak kodu, sonuç JSON'ları, tablo, şekil ve hash manifesti Git içinde
dondurulur. Ham MaCySTe PCAP/EVE/CSV kampanyası büyüklük ve veri politikası
nedeniyle `captures/` altında Git dışındadır. Harici, değişmez bir arşiv ve
kalıcı DOI yayımlanana kadar paket **yerelde tam**, başka bir klonda ise
**veri-girdisi eksik** kabul edilir. Bu sınır manifestte
`pending_external_campaign_archive` olarak makine-okunur biçimde kayıtlıdır.
