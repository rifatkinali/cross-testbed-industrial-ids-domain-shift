# Veri Seti Kartı — Maritime-Lab OT IDS Dataset (v0.3)

> Bu belge veri setinin nasıl üretildiğini, ne içerdiğini ve **neyi
> temsil etmediğini** açıkça tanımlar. Yayın/atıf ve tez savunması için
> zorunlu referans belgedir.

---

## 1. Künye

| Alan | Değer |
|---|---|
| Ad | Maritime-Lab OT IDS Dataset |
| Sürüm | v0.3 |
| Üretim tarihi | 2026-07-23 → 2026-07-25 |
| Dosyalar | `attack/dataset.csv` (olay-seviyesi öznitelikler), `attack/labels.csv` (koşu/pencere ground-truth), `DATASET-SNAPSHOT-v0.3.json` (hash ve istatistik manifesti), `suricata-ot/logs/ot.pcap.*` (ham paket, repoda tutulmaz) |
| Satır sayısı | 57 499 |
| Kolon sayısı | 34 |
| Koşu sayısı | 20 (`run_id` = 3…22) |
| Sensör | `ot` 57 397 satır (OpenPLC/Modbus, Suricata `suricata-ot`) · `signalk` 102 satır (AIS/NMEA, Suricata `suricata`) |
| Lisans | Veri: CC BY 4.0 · Kod: MIT (bkz. `LICENSE`) |
| Kaynak attestation | `synthetic-lab-only-no-real-vessel-traffic` |
| Attestation kapsamı | Yalnız `primary_dataset` ve `ground_truth`; dış karşılaştırma profilleri ayrı kaynaklardır |
| Ana veri SHA-256 | `24d3380a9d237e3b90f4ff942823edd94252d336f59ed12203b4ec6a00a30a46` |
| Ground-truth SHA-256 | `72f28e4d1fe40e3b55fd92d4cf3c96162b01523a060e1eb208a80ddba3240a75` |

---

## 2. Nasıl üretildi

```
docker compose up            (SignalK + OpenPLC + Suricata IT + Suricata OT)
python attack/run_scenarios.py   → izole saldırı pencereleri + labels.csv (ground-truth)
python attack/build_dataset.py   → eve.json + labels.csv  →  dataset.csv
```

- Tüm trafik **izole Docker ağında** üretilir. Gerçek gemi, liman veya
  canlı AIS sistemine hiçbir şekilde dokunulmaz.
- Açık veri sürümünün kaynak beyanı
  **`synthetic-lab-only-no-real-vessel-traffic`** değeridir. Bu beyan
  `attack/dataset.csv` ve `attack/labels.csv` için geçerlidir; manifestte
  ayrıca listelenen dış karşılaştırma profillerinin kendi kaynak ve lisans
  koşullarının yerine geçmez.
- Gelecekte fiziksel bir laboratuvar, üretici testbed'i veya gerçek gemi
  üzerinde izinli doğrulama yapılırsa, o trafik bu veri setine eklenmez.
  Ayrı erişim denetimi, ayrı provenance manifesti ve ayrı lisans incelemesi
  gerektirir. Gerçek operasyon verisi CC BY 4.0 sürüm hattına girmez.
- `run_scenarios.py` her senaryonun **başlangıç/bitiş zaman damgasını**
  yazar; etiketleme buradan otomatik türetilir (elle etiketleme yok).
- Ek olarak `build_dataset.py`, pencere içi olayları **içerik temelli**
  ayıklar (`is_attacker_action`): pencere içinde kalan meşru poll
  okumaları `background` senaryosu altında **normal** etiketlenir. Bu,
  "saldırı penceresindeki her paket saldırıdır" hatasını önler.

---

## 3. Sınıf ve senaryo dağılımı

| Etiket | Satır | Oran |
|---|---:|---:|
| normal | 41 558 | %72.3 |
| attack | 12 378 | %21.5 |
| fault | 3 563 | %6.2 |

| Senaryo | Satır | Saldırı payı | Etiket | MITRE ATT&CK for ICS |
|---|---:|---:|---|---|
| `baseline` | 40 944 | — | normal | — |
| `background` | 614 | — | normal | — (pencere içi meşru trafik) |
| `dos_flood` | 8 553 | %69.1 | attack | T0814 Denial of Service |
| `recon_scan` | 905 | %7.3 | attack | T0842 / T0846 keşif |
| `steering_hijack` | 697 | %5.6 | attack | T0836 / T0831 |
| `engine_rpm_ramping` | 685 | %5.5 | attack | T0836 / T0831 |
| `power_blackout_trip` | 576 | %4.7 | attack | T0828 / T0831 |
| `power_overload` | 481 | %3.9 | attack | T0836 / T0828 |
| `modbus_write` | 379 | %3.1 | attack | T0836 / T0855 / T0831 |
| `ais_spoof` | 102 | %0.8 | attack | T0865 / T0856 |
| `fault_poll_storm` | 3 272 | — | fault | İyi huylu aşırı polling |
| `fault_plc_restart` | 221 | — | fault | İyi huylu PLC yeniden başlatma |
| `fault_comm_dropout` | 70 | — | fault | İyi huylu iletişim kesintisi |

**Olay tipi dağılımı:** alert 17 739 · modbus 14 108 · flow 8 275 ·
anomaly 6 821 · http 4 470 · fileinfo 4 457 · stats 1 629.

**v0.2 → v0.3 değişimi:** dört yeni koşu (19–22), üç iyi huylu arıza sınıfı,
daha uzun AIS/GNSS enjeksiyonları ve güç/dümen/RPM saldırılarının yeni
tekrarları eklendi. Ham satır sayısı 40 771'den 57 499'a çıktı. Ana veri,
ground-truth, değerlendirme raporları ve dış profil çıktıları
`DATASET-SNAPSHOT-v0.3.json` içinde SHA-256 ile donduruldu.

Snapshot bütünlüğünü sonradan doğrulamak için:

```powershell
python ml/snapshot_dataset.py --verify DATASET-SNAPSHOT-v0.3.json --data-only
```

Bu komut yayımlanan ana veri ile ground-truth bütünlüğünü doğrular. Tam
manifest doğrulaması kaynak kodu ve raporları da kapsar; bunun için snapshot
içindeki `provenance.git.head` commit'i checkout edilmelidir. Daha sonraki
güvenlik düzeltmelerinin v0.3 üretim koduymuş gibi hash'e yeniden yazılması
provenance tahrifi olur.

### 3.1 Katman ayrımı: ham dağılım ≠ model girdisi (v0.3)

Yukarıdaki tablo **ham** satır sayısıdır. ML modeli yalnızca `flow` ve `modbus`
olaylarını görür (`anomaly` v0.3'te çıkarıldı — bkz. `docs/bulgular/BULGU-ariza-anomaly.md`).
Bu ayrım, denge tartışması için kritik olan tabloyu değiştirir:

| Senaryo | Model-girdisi (`flow`+`modbus`) | Saldırı içi pay | Katman |
|---|---:|---:|---|
| `dos_flood` | **0** | %0 | **imza** (hacim; ML görmez) |
| `recon_scan` | 551 | %38.9 | ML |
| `steering_hijack` | 213 | %15.1 | ML |
| `engine_rpm_ramping` | 210 | %14.8 | ML |
| `power_blackout_trip` | 175 | %12.4 | ML |
| `power_overload` | 139 | %9.8 | ML |
| `modbus_write` | 96 | %6.8 | ML |
| `ais_spoof` | 31 | %2.2 | ML + seyir dedektörü |

Model-girdisi toplamı 22 383 olaydır: 19 277 normal, 1 415 saldırı ve
1 691 arıza. İkili saldırı modelinin eğitim/değerlendirmesinde arızalar
eğitimden çıkarılır; ayrı arıza testi bunları görülmemiş iyi huylu durum
olarak kullanır.

**Tasarım kararı — hibrit katman ayrımı:** `dos_flood` bir **hacim** saldırısıdır
ve `flow`/`modbus` içerik özniteliklerinde ayırt edici iz bırakmaz; tespiti
imza/anomali katmanının işidir (kapsama matrisinde `dos_flood` imza tarafından
17/17 yakalanır). Bu bilinçli bir mimari seçimdir: **hacim → imza, içerik → ML.**
Dolayısıyla "DoS payı çok yüksek" eleştirisi ML değerlendirmesi için geçersizdir
— ML metriklerinin paydası model-girdisi satırlardır, ki orada DoS yoktur.

**Ölçüm kuralı:** senaryo dengesi ve azınlık-sınıf yeterliliği **daima
model-girdisi** tablosuna göre değerlendirilir; ham tablo yalnızca üretim
hacmini gösterir.

---

## 4. Kolon sözlüğü

**Kimlik / etiket**

| Kolon | Açıklama |
|---|---|
| `timestamp` | Olayın UTC zaman damgası (ISO 8601) |
| `run_id` | Koşu numarası — zaman-temelli bölme ve LOSO için gruplama anahtarı |
| `label` | `normal` / `attack` — ground-truth |
| `scenario` | Senaryo adı (yukarıdaki tablo) |
| `sensor` | `ot` (Modbus segmenti) / `signalk` (seyir segmenti) |

**Akış / ağ öznitelikleri (Suricata eve.json)**

`event_type`, `proto`, `app_proto`, `src_ip`, `src_port`, `dest_ip`,
`dest_port`, `pkts_toserver`, `pkts_toclient`, `bytes_toserver`,
`bytes_toclient`, `flow_state`

**Protokol öznitelikleri (Modbus)**

| Kolon | Açıklama |
|---|---|
| `modbus_function` | Function code (FC1/3 okuma, FC5/6/15/16 yazma) |
| `modbus_access` | READ / WRITE |
| `modbus_address` | Hedef register/coil adresi |
| `modbus_quantity` | İstenen register sayısı |
| `modbus_value` | Yazılan değer |

**Süreç (process-aware) öznitelikleri — OpenPLC holding register 0–9**

| Register | Kolon | Fiziksel anlam |
|---:|---|---|
| 0 | `engine_rpm` | Ana makine devri |
| 1 | `ballast_level` | Balast tankı seviyesi |
| 2 | `ballast_setpoint` | Balast set-point |
| 3 | `rpm_command` | RPM komutu |
| 4 | `heading_cmd` | Rota komutu |
| 5 | `heading` | Gerçek rota |
| 6 | `rudder_angle` | Dümen açısı |
| 7 | `gen_load` | Jeneratör yükü |
| 8 | `bus_freq` | Bara frekansı |
| 9 | `load_cmd` | Yük komutu |

**İmza katmanı:** `alert_signature`, `alert_sid` (Suricata kural eşleşmesi).

### Etiket sınıfları

| `label` | Anlam | İkili y (`attack`?) |
|---|---|---|
| `normal` | Meşru trafik | 0 |
| `attack` | Saldırgan eylemi | 1 |
| `fault` | **İyi huylu ekipman arızası** (v0.3+): link kopması, PLC restart, aşırı poll | 0 |

`fault` sınıfı, "saldırı mı arıza mı" ayrımını ölçmek için eklenmiştir.
İkili sınıflandırmada saldırı **değildir** (y=0); `ml/evaluate_faults.py`
arızaları eğitimden tamamen çıkarıp görülmemiş arıza altında yanlış-alarm
oranını ölçer.

---

## 5. Önerilen kullanım

- **Bölme:** `run_id` bazlı gruplama ile zaman-temelli bölme. Rastgele
  satır bölmesi **kullanmayın** — aynı akışın olayları sızar.
- **Referans holdout:** son iki koşu (v0.3'te 21 ve 22) test olarak ayrılır
  (`ml/ml_out/holdout_metrics.json`, `ml/ml_out/detection_report.json`).
- **Zero-day testi:** `ml/loso_generalization.py` — bir senaryoyu tamamen
  eğitimden çıkarıp genelleme ölçer.
- **Değerlendirme metriği:** OT bağlamında **yanlış pozitif oranı (FPR)**
  F1'den daha belirleyicidir; her sonuç FPR ile birlikte raporlanmalıdır.

---

## 6. Bilinen sınırlılıklar (dürüst konumlandırma)

1. **Senaryo dengesizliği yapaydır.** Saldırı satırlarının %69.1'i tek bir
   senaryodan (`dos_flood`) gelir; çünkü flood doğası gereği çok olay
   üretir. Bu nedenle **makro-ortalama ve senaryo-bazlı metrikler**
   raporlanmalıdır; genel accuracy ve mikro-recall yanıltıcıdır
   (bkz. `ml/ml_out/benchmark_table.txt`, bölüm B ve C).
2. **AIS/GPS spoofing yapısal olarak seyrek (102 ham, 31 model-girdisi
   satırı).** Bir spoofing penceresinde NMEA cümleleri tek bir TCP bağlantısı üzerinden gider;
   Suricata bunu yalnızca bir-iki akış olayı olarak görür. Yani koşu
   eklemek AIS satırlarını ancak doğrusal ve yavaş artırır. AIS'in
   içerik-seviyesi tespiti eve.json'da değil, SignalK akışını tüketen
   `ml/nav_ids.py` tarafındadır; veri seti bu saldırıyı **ağ izi**
   düzeyinde temsil eder, sema/konum tutarlılığı düzeyinde değil.
3. **Ham pcap üretiliyor ama repoda yayınlanmıyor.** `suricata-ot` artık
   `ot.pcap.<unixtime>` dosyaları yazar (paket-seviyesi analiz mümkündür);
   dosyalar boyut nedeniyle `.gitignore`'dadır ve ayrı bir veri deposuna
   (Zenodo/IEEE DataPort) yüklenmelidir. `dataset.csv` ise eve.json türevi
   olay seviyesindedir.
4. **Simülasyondur.** OpenPLC ST mantığı gerçek bir gemi otomasyon
   sistemini basitleştirir; gerçek gemi ağındaki gürültü, cihaz çeşitliliği
   ve satıcıya özgü Modbus davranışları temsil edilmez.
5. **Tek topoloji, tek saldırgan.** Saldırgan tek konak, sabit IP. Yanal
   hareket, çok aşamalı kampanya ve içeriden tehdit senaryoları yoktur.
6. **Kısa süre.** Toplam ~2 günlük (kesintili) pencere; mevsimsel/sefer-boyu
   davranış değişimi (concept drift) gözlenemez.
7. **Etiket türetimi kısmen kural tabanlıdır.** `is_attacker_action`
   içerik kuralları ground-truth'u iyileştirir ama aynı zamanda bir
   *önyargı* kaynağıdır: aynı kuralları kullanan bir modelin başarısı
   şişebilir. Model karşılaştırmalarında bu bağımlılık tartışılmalıdır.

---

## 7. Etik ve yasal

Tüm veriler izole laboratuvarda sentetik olarak üretilmiştir. Gerçek kişi
verisi, gerçek gemi/MMSI kimliği veya canlı AIS yayını içermez. Saldırı
scriptleri yalnızca bu lab içindeki hedeflere yöneliktir.

---

## 8. Lisans ve atıf

Birincil OpenPLC veri seti **CC BY 4.0** ile paylaşılır. OpenPLC–MaCySTe
transfer artefaktıyla ayrıca dağıtılan 12 MaCySTe `events-v0.4.csv` dosyası da
proje tarafından üretilmiş deney gözlemleri olarak **CC BY 4.0** kapsamındadır.
Bu CSV'ler MaCySTe kaynak kodunu içermez ve MaCySTe kaynak kodunu yeniden
lisanslamaz. MaCySTe atfı ile upstream AGPL-3.0 lisans/README kopyaları
artefaktın `THIRD-PARTY-NOTICE.md` ve `THIRD_PARTY_LICENSES/` yollarında
korunur. Kullanımda `CITATION.cff` içindeki atıf bilgisini kullanın.

## 9. Sürüm geçmişi

| Sürüm | Tarih | Değişiklik |
|---|---|---|
| v0.1 | 2026-07-25 | İlk kart; 10 koşu, 8 senaryo, 20 249 satır. AIS yok, pcap yok. |
| v0.2 | 2026-07-25 | 16 koşu, 9 senaryo, 40 771 satır. AIS/NMEA ilk kez dahil (67 satır); ham pcap üretimi açıldı; içerik saldırıları 5–12 kat arttı. |
| v0.3 | 2026-07-26 | 20 koşu, 13 senaryo, 57 499 satır. Üç arıza sınıfı eklendi; azınlık saldırıları büyütüldü; model-girdisi/ham olay ayrımı kesinleştirildi; hash snapshot'ı oluşturuldu. |
