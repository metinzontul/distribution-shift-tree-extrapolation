# TabReD doğal temporal shift karakterizasyonu, yorum notu

> Statü: POST-HOC / EXPLORATORY. Confirmatory bir test DEĞİL. Amaç: WP3 temporal-arm'ın önceden
> kaydedilmiş NULL bulgusunun (4 TabReD sette aileler eşdeğer) ekstrapolasyon-mekanizmasıyla
> TUTARLI olup olmadığını, doğal temporal shift'in TÜRÜnü ölçerek göstermek. Manuscript'te açıkça
> post-hoc/exploratory etiketlenecek (methodology-review §6). Ölçüm sonuç görülmeden tasarlandı,
> sonuç dürüstçe raporlanıyor (beklentiyi destekleyip desteklemediğine bakılmaksızın).

## Soru ve mantık

Mekanizma (WP2/sentetik + WP3 covariate-arm): ağaç-tabanlı modeller (GBDT) eğitim feature-aralığının
DIŞINDA kötü ekstrapole eder, NN/FM daha iyi. Bu mekanizmanın ön-koşulu: test noktalarının, tahmini
sürükleyen (Y-ile-ilişkili) feature'larda train-desteğinin (train [min,max]) DIŞINA taşması. WP3
covariate-arm bunu construct ile DAYATIR (split feature = max |Spearman(X,Y)|, train = alt-%30,
test sev50 = üst pencere), mekanizma REPLİKE olur (%83 dataset). WP3 temporal-arm (4 gerçek TabReD,
doğal zaman-split) NULL. Hipotez: doğal temporal shift "aralık-dışına taşma" türü DEĞİL, dolayısıyla
mekanizmanın ön-koşulu yok, temporal-null BEKLENEN ve mekanizmayla TUTARLI.

## Ölçüm (kısa)

Her TabReD setinin authored `default` temporal split'i (fit=train dönemi, OOD=test/gelecek dönemi),
yalnız sayısal (x_num) feature'lar üzerinde. `x_bin` binary (aralık-ekstrapolasyonu tanımsız),
`x_cat`/`x_meta` feature değil (x_meta col0 = timestamp). Train desteği = TAM train-dönem bloğu,
train-medyan impute (leakage-safe, load_tabred ile uyumlu). Out-of-range (OOR) = OOD-test
hücrelerinin train [min,max] (ve [p1,p99]) dışına düşme oranı. Construct kıyası: aynı OOR tanımı,
covariate extrapolation-arm sev50, leakage-safe 27 covariate dataset (air_quality [predictive
feature'lar %30-90 NaN] + housing [ctr23 california_housing dedup] roster'dan hariç -> N=27).

## Ana bulgu (headline)

**Doğal temporal shift, tahmini sürükleyen feature'larda feature-range extrapolation ÜRETMİYOR.**

Mekanizmanın tasarlanmış ekseninde, yani en Y-ilişkili (max |Spearman|) feature'da, out-of-range:

| ölçüm (OOR, train [min,max]) | sberbank | cooking | weather | delivery | construct sev50 (n=27) |
| --- | --- | --- | --- | --- | --- |
| **Y-ilişkili feature'da OOR** | **0.000** | **0.000** | **0.000** | **0.000** | **1.000** (medyan; 27/27 = 1.00) |
| ortalama hücre OOR (tüm feature) | 0.092 | 0.000 | 0.000 | 0.000 | 0.156 (medyan), 0.228 (ort) |
| max-feature OOR | 1.000* | 0.000 | 0.001 | 0.001 | 1.000 |
| OOR [p1,p99] ortalama | 0.102 | 0.013 | 0.031 | 0.018 | 0.298 (ort; medyan 0.226) |

Kontrast X vs Y: **pozitif testimiz Y-ilişkili feature'a ~%100 out-of-range DAYATIR (27/27 set);
doğal temporal shift'te bu oran ~%0 (4/4 set).** Robustluk: modellerin gerçekte gördüğü train
alt-örneği (max_fit_train=5000, hatta 3000) ile de Y-feature OOR ≈ 0 (≤0.0006) ve ortalama OOR
construct seviyesine SIÇRAMIYOR (`robustness_traincap.json`), yani sonuç train-desteği boyutuna
dayanıklı.

*sberbank max-feature OOR = 1.0 bir NÜANS içeriyor, aşağıda dürüstçe ele alınıyor.

## Kayma GERÇEK ama farklı-tür (aralık-dışı değil, destek-içi yeniden-dağılım)

"OOR ≈ 0" yorumu "kayma yok" DEĞİLDİR. Kovaryat kayma ölçülebilir düzeyde var, ama destek-içi
kalıyor (density yeniden-dağılıyor, aralık genişlemiyor):

- Per-feature PSI: median küçük (0.005-0.067) ama **PSI_max büyük** (weather 12.79, sberbank 12.83,
  delivery 5.70); featurelerin %17-26'sı "large" eşiğini (PSI>0.25) aşıyor (cooking hariç, ~stabil).
  Yani bazı feature'lar dağılımca ağır kayıyor, ama test değerleri train [min,max] içinde kalıyor
  (weather: PSI_max 12.79 iken max-feature OOR 0.001). Bu klasik covariate-shift-as-redistribution
  (ör. weather'da mevsimsel kayma), feature-range extrapolation DEĞİL.
- Target/prior shift modest ama var: en belirgin sberbank (Y_SMD +0.227, Y üst-kantil kayması
  +0.266 std, KS 0.107) ve weather (Y üst-kantil +0.257). Y varyansı stabil (std-ratio ~1.0).
- Crude concept-drift probesi (ridge OOD-RMSE / train-RMSE): sberbank 3.34 (lineer P(Y|X) ilişkisi
  gelecek döneminde bozuluyor, concept/prior drift işareti), cooking/weather/delivery ~1.0
  (lineer ilişki stabil). Bu bir PROXY, kesin P(Y|X)-drift testi değil (gelecek-iş).

## NET HÜKÜM

**Doğal temporal shift (4 TabReD) feature-range extrapolation TÜRÜNDE DEĞİL; mekanizmanın ön-koşulu
(tahmin-sürükleyen feature'larda train-desteğinin dışına taşma) 4 sette de YOK. Dolayısıyla
temporal-null BEKLENEN sonuçtur ve ekstrapolasyon-mekanizmasıyla TUTARLIDIR.** Ölçülen shift türü
(destek-içi kovaryat yeniden-dağılım + modest prior/concept drift) tam da tüm aileleri EŞİT
cezalandıran türdür, ki verdict.md'de gözlenen de budur (nOOD tüm ailelerde ~0.01 içinde eşdeğer).
Mekanizma tam öngördüğü yerde (construct feature-range, Y-feature %100 OOR) çıkıyor, öngörmediği
yerde (doğal temporal, Y-feature %0 OOR) kayboluyor. Triangülasyon: mekanizma "gerçek veride hiç
tutmuyor" değil, construct-arm'da ateşlenebiliyor; ön-koşul karşılandığında var, karşılanmadığında yok.

## Dürüst caveat'lar (overclaim önleme)

1. **Post-hoc + N=4 + descriptive.** Bu ölçüm önceden-kaydedilmiş bir null'ı AÇIKLAMAK için sonradan
   tasarlandı. "Tutarlı" bir uyum iddiasıdır, nedensel ispat değil. 4 temporal set küçük; hüküm
   descriptive. Manuscript dili "consistent-with", "mechanism-consistent reading" olmalı.
2. **sberbank nüansı (kritik, dürüstçe).** sberbank'ta ortalama OOR %9.2 ve bazı feature'lar %100
   out-of-range (max-feature OOR = 1.0). AMA bu OOR tamamen NUISANCE / zayıf-tahminci feature'larda:
   en-çok-out-of-range 5 feature'ın |Spearman(Y)| değerleri 0.02-0.17 (hepsi düşük; muhtemelen
   monoton artan tarih/indeks tipi kolonlar), buna karşın gerçek tahminci feature (|Spearman|=0.71)
   %0 out-of-range. Yorum: ağaçlar yalnız tahmini SÜRÜKLEYEN feature'larda ekstrapole ederken
   cezalanır; tahmine katkısı olmayan feature'larda aralık-dışına çıkmak bedava. Bu, mekanizma-
   tutarlı bir okuma, ama YORUM (nedensel olarak kanıtlanmadı). Ayrıca sberbank tam da concept/prior
   drift'in en güçlü olduğu set (ridge 3.34), ve mekanizma orada da null, ki bu da tutarlı.
3. **Ters yönü açıkça beyan (güçlendirme tersine dönmez).** Eğer TabReD Y-ilişkili feature'larda
   YÜKSEK out-of-range gösterip mekanizma yine null olsaydı, güçlendirme tersine döner ve mekanizma
   doğal-durumda zayıflardı. Ölçüm bunu GÖSTERMEDİ: Y-feature OOR 4/4 sette ~0. O yüzden tutarlılık-
   argümanı ayakta.
4. **Concept-drift ölçümü zayıf.** Lineer ridge proxy; kesin P(Y|X)-drift ayrıştırması gelecek-iş.

## Go / no-go (mekanizma-tutarlılık kontrolü)

| Soru | Cevap |
| --- | --- |
| Doğal temporal shift feature-range extrapolation mı (Y-feature'da)? | HAYIR (4/4 set OOR ≈ 0) |
| Mekanizmanın ön-koşulu karşılanıyor mu? | HAYIR (Y-feature train-desteği içinde) |
| Temporal-null mekanizmayla tutarlı mı? | **EVET (tutarlı; null beklenen)** |
| Kayma var mı (yanlışlıkla "shift yok" denmesin diye)? | EVET (destek-içi yeniden-dağılım + modest prior/concept drift) |
| Güçlendirme tersine döner mi (yüksek-OOR-ama-null)? | HAYIR (OOR düşük, risk yok) |

## Methodology-reviewer'a bayraklar

1. **Post-hoc consistency framing.** Bu, önceden-kaydedilmiş null'ı açıklayan sonradan-ölçüm; §6
   uyarınca exploratory/consistency olarak sunulmalı, confirmatory olarak DEĞİL. Manuscript "mechanism-
   consistent" der, "mechanism confirmed" demez.
2. **"Nuisance-feature ekstrapolasyonu sayılmaz" argümanı bir YORUM.** sberbank'ta gerçek feature-range
   extrapolation VAR (nuisance feature'larda %100), mekanizma yine null. Bunu "tahmini sürükleyen
   feature'da değil, o yüzden mekanizma-tutarlı" diye okuyoruz; makul ama nedensel ispat değil. Öneri:
   yorumu "plausible mechanism-consistent reading" olarak hedge et.
3. **N=4 temporal, güç sınırlı.** Genelleme yapılmamalı; TabReD 4 set endüstriyel-temporal örneklem.

## Dosya haritası (results/wp3_analysis/tabred_shift_characterization/)

- `shift_characterization.{csv,md}` -- ana tablo (4 TabReD × OOR [min,max]+[p1,p99] + construct kıyası
  + PSI/SMD + Y-shift + ridge probe) + covariate/Y-shift detay tablosu + construct per-dataset ek.
- `construct_per_dataset.csv` -- 27 covariate dataset sev50 out-of-range (kalibrasyon; leakage-safe roster).
- `fig_oor_tabred_vs_construct.png` -- 3 panel: (A) Y-ilişkili feature OOR (mekanizma ekseni),
  (B) ortalama-hücre OOR, (C) kovaryat-shift büyüklüğü (median PSI); 4 TabReD vs construct sev50.
- `tabred_metrics.json`, `construct_metrics.json` -- ham makine-okunur metrikler.
- `robustness_traincap.json` -- OOR'un train-desteği boyutuna (full/5000/3000) dayanıklılığı.
