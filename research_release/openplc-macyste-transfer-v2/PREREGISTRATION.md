# OpenPLC–MaCySTe RC2 Ön-Kayıtlı Deney Sözleşmesi

**Sürüm:** `v1.0.0-rc2-preregistration`  
**Dondurma tarihi:** 17 Ağustos 2026  
**Amaç:** RC1 veri-kapsamı, eşik ve belirsizlik kusurlarını sonuç seçmeden
önce kapatan RC2 analiz sözleşmesi

Bu belge bir iş planı değildir. RC2 bilimsel sonucunun hangi veri, bölme ve
ölçülerle üretileceğini sonuçlar yeniden hesaplanmadan önce donduran deney
sözleşmesidir. Makine-okunur eşlik kaydı `preregistration.json` dosyasındadır.

## 1. Dondurulmuş girdiler

### OpenPLC kaynak + arıza aday tablosu

- Yol: `.codex-validation/ds02-independent-faults-20260815/dataset.csv`
- SHA-256: `616bf8f05a8a52f9ba7679d54229a5dcab7c384b8f7db9616fadb78f12a37118`
- Ham türetilmiş satır: 62.630
- Köken kanıtı:
  `docs/sonuclar/OPENPLC-BAGIMSIZ-ARIZA-KOSULARI-2026-08-15.json`
- Köken kanıtı SHA-256:
  `d6ae0bb5baf94f80cbf42c0adfb54cac9d0bd3dd41bad361a1b278b57c54df12`

RC1 yalnız `run_id=19,20` arıza koşullarını içerir. RC2 adayı bunlara sonuç
görülmeden önce üretilmiş `run_id=23–28` koşullarını ekler. Bağımsızlık birimi
olay veya arıza etiketi değil, `run_id`'dir; 24 run×senaryo hücresi sekiz
bağımsız koşu olarak yorumlanır.

### MaCySTe hedef tablosu

- Kampanya: `captures/macyste/macyste-20260726T113328Z`
- Türetilmiş manifest: `derived-manifest-v0.4.json`
- Manifest SHA-256:
  `aa03a6fa04c279be88e7e4784129886f93a2f561ce6c730c7d336bdfc36ee8c8`
- Beklenen kapsam: 12 koşu, 34.949 olay; normal, fault, manipulation ve recon
  ailelerinin her birinde üç betiklenmiş tekrar.

## 2. Ana model veri sözleşmesi

1. Yalnız `event_type ∈ {flow, modbus}` kabul edilir.
2. Ana modelde yalnız `label ∈ {normal, attack}` kabul edilir.
3. Herhangi bir `fault` satırı taşıyan `run_id`, bütün satırlarıyla ana model
   eğitiminden ve eşik seçiminden çıkarılır. Böylece arıza koşulunun normal
   başlangıcı veya aynı çalışma bağlamı ana modele sızmaz.
4. Beklenen ana havuz tam olarak 18.941 satır, 18 run, 17.646 normal ve 1.295
   attack olayıdır. Run kimlikleri `3–18,21,22` olmalıdır.
5. Ana model run-bazlı %70 train / %30 validation bölmesi kullanır. Satır bazlı
   rastgele bölme yasaktır; train ve validation run kümeleri ayrık olmalıdır.
6. Dondurulmuş seed evreni `42–61` arası 20 tam sayıdır. Sonuca göre seed
   çıkarılmaz veya eklenmez.
7. Her seed bölmesinde train ve validation taraflarının her birinde en az iki
   bağımsız normal ve iki bağımsız attack run bulunmalıdır. Validation'da en az
   100 normal olay bulunmalıdır; aksi halde %1 eşik çözünemez ve koşu reddedilir.
8. Dört şema değişmeden kullanılır: `flow`, `protocol`, `physical_strict`,
   `physical_proxy`. Model ailesi 200 ağaçlı mevcut Random Forest'tır.
9. Eşik yalnız kaynak validation normal olaylarından, `pred = score >= threshold`
   altında ampirik FPR en fazla 0,01 olacak şekilde seçilir. Hedef veri veya
   fault stres verisi eşik/model/özellik seçimine giremez.

## 3. Ayrı OpenPLC fault stres testi

1. Fault havuzu yalnız `event_type ∈ {flow, modbus}`, `label=fault` ve
   `run_id ∈ {19,20,23,24,25,26,27,28}` satırlarından oluşur.
2. Beklenen kapsam 3.337 olay, 8 bağımsız run ve her birinde bulunan üç sınıftır:
   `fault_comm_dropout`, `fault_plc_restart`, `fault_poll_storm`.
3. Fault verisi model eğitimine, özellik seçimine veya eşik seçimine hiçbir
   şekilde katılmaz; yalnız dondurulmuş modeller/eşiklerle sonradan skorlanır.
4. Birincil betimsel stres ölçüsü, her run×fault-senaryo hücresine eşit ağırlık
   veren 24 hücreli yanlış-alarm oranıdır. Olay-ağırlıklı oran ikincil olarak
   ayrıca sunulur; çok satırlı poll-storm koşusu sekiz bağımsız deney sayılmaz.
5. Ölçüler 20 seed için ayrı hesaplanır ve ortanca `[min,max]` olarak verilir.
   Bunlar gerçek-saha güven aralığı veya genel performans iddiası değildir.

## 4. MaCySTe hedef ölçüleri

### Birincil, eşikten bağımsız

- Ensemble skor: 20 seed model skorunun olay bazında aritmetik ortalaması.
- Yalnız `event_type ∈ {flow, modbus}` olayları kabul edilir.
- Olay-ağırlıklı AUC.
- Dört senaryo ailesine eşit toplam ağırlık veren senaryo-dengeli AUC.
- Manipulation-vs-normal ve recon-vs-normal AUC.

Nokta tahmini ensemble skorundan hesaplanır. Run-stratified bootstrap aynı
ensemble skoru ve aynı ölçüyü kullanır. Üç yakın betik tekrar nedeniyle sonuç
`scripted-run sensitivity interval` olarak adlandırılır; popülasyon güven
aralığı veya testbed'ler arası genelleme aralığı denmez.

Bootstrap dört senaryo ailesinin her birinde koşu bazında tabakalıdır; 2.000
tekrar, `20260817` tohumu ve `[0,025; 0,975]` yüzdelikleri kullanılır.

### İkincil, eşik-bağımlı ve yalnız betimsel

- MaCySTe normal FPR.
- MaCySTe fault FPR.
- Manipulation recall.
- Recon recall.

Bu dört ölçü seed başına hesaplanıp ortanca `[min,max]` olarak raporlanır.
README/release başlığına, özetin ilk paragrafına veya ürün iddiasına taşınmaz.

## 5. Ön-kayıtlı red ve iddia sınırları

Analiz aşağıdakilerden biri olursa fail-closed reddedilir:

- girdi hash'i, beklenen satır/run/sınıf sayısı veya run kümesi değişirse;
- ana modelde `fault`, fault-run bağlamı veya flow/modbus dışı olay kalırsa;
- train/validation run'ları çakışır veya minimum run/eşik çözünürlüğü bozulursa;
- target/fault verisi model, özellik veya eşik seçimine girerse;
- nokta tahmini ve belirsizlik aralığı farklı skor/ölçülerden üretilirse;
- 20 seed'in herhangi biri sonuç nedeniyle dışlanırsa;
- üç MaCySTe tekrarından genel testbed/saha güven aralığı iddia edilirse.

RC2 şu iddiaları üretmez:

- bütün endüstriyel veya denizcilik IDS'lerinin aktarılamadığı;
- gerçek gemi performansı, IMO/IACS/klas uygunluğu veya ürün doğrulaması;
- fault veya MaCySTe FPR değerlerinin genel/popülasyon tahmini olduğu;
- D2/D3 gözlemlerinin nedensellik kanıtladığı;
- GÖZCÜ EDGE'in bu deneyle doğrulandığı veya üstün olduğu.

## 6. Yayın kapısı

RC2 ancak şu koşullarda public adaya dönüşebilir:

- sözleşmenin fail-closed testleri geçer;
- ana sonuç, bütün tablo/JSON/şekil/README/release/tez ifadelerinde aynıdır;
- RC1 erratası ve RC2 düzeltme farkı görünürdür;
- CFF tarihi fiilî yayın tarihine kadar `date-released` taşımaz;
- ZIP kök adı sürümle aynıdır ve staging adı içermez;
- yeniden üretim temiz ortamda aynı makine-okunur sonuçları üretir;
- repo/tag/release asset readback hashleri yerelle eşleşir.
