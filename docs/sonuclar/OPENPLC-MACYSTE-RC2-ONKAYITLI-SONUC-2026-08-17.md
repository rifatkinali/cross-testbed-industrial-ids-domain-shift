# OpenPLC–MaCySTe RC2 Ön-Kayıtlı Analiz Sonucu

**Tarih:** 17 Ağustos 2026  
**Durum:** Yerel RC2 bilimsel sonuç üretildi; kamu yayını yapılmadı  
**Makine-okunur kanıt:**
[`OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json`](OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json)  
**Kanıt SHA-256:**
`472df978e118f84cde1d3c9b7263f3ab1b21dfeed0f5b8be817723ab0055384b`

## Ne yaptık

RC1 denetiminde bulunan veri-kapsamı, eşik ve belirsizlik kusurlarını sonuç
görmeden önce dondurulan RC2 sözleşmesiyle kapattık.

- Ana model yalnız `flow/modbus` türündeki `normal/attack` olaylarıyla eğitildi.
- Herhangi bir arıza olayı taşıyan sekiz koşunun bütün bağlamı ana modelden ve
  eşik seçiminden çıkarıldı.
- 18 bağımsız ana koşu, her seed için koşu bazında 12 eğitim ve 6 doğrulama
  koşusuna ayrıldı. `seed=42–61` aralığındaki 20 sonucun hiçbiri seçilmedi veya
  çıkarılmadı.
- Eşik yalnız kaynak doğrulama normal olaylarında en fazla %1 yanlış alarm
  verecek şekilde seçildi.
- `run_id=19,20,23–28` içindeki 3.337 arıza olayı, eğitimden sonra sekiz
  görülmemiş iyi huylu stres koşusu olarak ölçüldü.
- MaCySTe için nokta tahmini ile duyarlılık aralığı aynı 20-model ensemble
  skorundan hesaplandı. Aralık 12 yakın betik koşusuna duyarlılığı gösterir;
  popülasyon güven aralığı değildir.

## Neden önemli

Bu düzeltme iki farklı soruyu birbirine karıştırmayı önlüyor:

1. Model kaynak laboratuvarda normal ve saldırıyı ayırabiliyor mu?
2. Kaynakta öğrenilen sıralama başka bir testbed'e taşınıyor mu?

Arıza verisini saldırı eğitiminin veya eşik seçiminin içine koymak, iyi huylu
stresi modelin önceden gördüğü bir sınıfa dönüştürürdü. RC2'de arıza koşulları
tamamen ayrı tutulduğu için stres sonucu daha dürüst ve denetlenebilirdir.

## Sonuç ne anlama geliyor

| Özellik şeması | Kaynak doğrulama AUC, 20 seed ortancası | MaCySTe olay AUC, ensemble | MaCySTe senaryo-dengeli AUC | OpenPLC arıza hücre-dengeli yanlış alarm, betimsel |
|---|---:|---:|---:|---:|
| `flow` | 0,723309 | 0,321893 | 0,393842 | %0,0000 |
| `protocol` | 0,904289 | 0,318365 | 0,389290 | %0,0000 |
| `physical_strict` | 0,936170 | 0,328890 | 0,384612 | %0,0000 |
| `physical_proxy` | 0,968221 | 0,400285 | 0,454040 | %0,0568 |

Ana ve savunulabilir gözlem şudur: kaynak doğrulamada güçlü görünen üç şema,
MaCySTe'de 0,5'in altında kalıyor; `flow` şeması da her iki alanda zayıf. Bu,
yalnız bu OpenPLC → MaCySTe yönü, bu veri üretimleri ve bu Random Forest model
ailesi için aktarım başarısızlığına işaret eder.

Sekiz OpenPLC arıza koşusundaki düşük yanlış alarm ayrıca olumlu bir betimsel
stres bulgusudur; fakat sekiz koşudan gerçek saha, ürün veya popülasyon
performansı çıkarılamaz. MaCySTe eşik-bağımlı yanlış alarm ve recall değerleri
de üç yakın betik tekrar nedeniyle manşet veya genel performans iddiası değildir.

## Sıradaki adım ne

RC2 README, şekil, release açıklaması ve yeniden üretim paketinin bütün
ifadeleri bu sonuçla eşleştirilecek. Paket temiz ortamda tekrar üretilecek ve
hash doğrulaması yapılacak. Ancak bundan sonra özel GitHub deposunda yeni bir
RC2 taslağı hazırlanabilir; kamu yayını ayrıca açık bir son kontrol gerektirir.

Bağımsız R1 yeniden üretimi, ikinci hedef testbed veya ters yön deneyi hâlâ en
değerli dış destek alanıdır. R2 ham veri arşivi ve R3 bağımsız tekrar bu sonuçla
tamamlanmış sayılmaz.
