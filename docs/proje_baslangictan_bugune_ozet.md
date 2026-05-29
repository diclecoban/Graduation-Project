# Projenin Başlangıcından Bugüne Kadar Yapılanlar

Bu dosya, batarya ömrü tahmini bitirme projesinde başlangıçtan bugüne kadar
yapılan çalışmaları tek bir yerde açıklamak için hazırlanmıştır. Amaç, projeyi
sonradan açan birinin hangi veri setleriyle çalışıldığını, hangi hataların
düzeltildiğini, hangi deneylerin yapıldığını, hangi sonuçların elde edildiğini
ve son danışman geri bildirimi sonrası nelerin eklendiğini hızlıca
anlayabilmesidir.

## 1. Projenin Amacı

Projenin ana hedefi, lityum-iyon batarya hücrelerinin erken çevrim verilerinden
nihai çevrim ömrünü tahmin etmektir. Burada "erken çevrim" ifadesi, bataryanın
tüm ömrünü beklemeden ilk belirli sayıda çevrimdeki kapasite davranışına bakarak
ömür tahmini yapmayı ifade eder.

Bu çalışma iki halka açık LFP batarya veri setine odaklanır:

- **MATR / Severson veri seti**: Literatürde sık kullanılan batarya ömrü veri
  setidir.
- **HUST veri seti**: Farklı deney koşulları ve farklı ömür dağılımı olan ikinci
  veri setidir.

Projenin temel araştırma soruları şunlardır:

1. Aynı veri seti içinde erken çevrim kapasite özellikleriyle batarya ömrü ne
   kadar iyi tahmin edilebilir?
2. Bir veri setinde eğitilen model diğer veri setine aktarılabilir mi?
3. Aktarım başarısız oluyorsa bunun nedeni sadece özellik dağılımı farkı mı,
   yoksa aynı özelliklerin ömürle ilişkisinin değişmesi mi?
4. Küçük sayıda hedef veri seti etiketi kullanılarak model veya belirsizlik
   tahmini iyileştirilebilir mi?

## 2. Başlangıçtaki Durum ve Düzeltilen Problemler

Projenin ilk halinde bazı yöntemsel problemler vardı. Bunlar daha sonra
danışman SOPv2 protokolüne göre düzeltildi.

Başlıca düzeltilen noktalar:

- **Q0 tanımı düzeltildi**: Başlangıçta Q0, ilk pozitif discharge kapasitesi
  gibi hatalı bir yaklaşımla alınabiliyordu. SOPv2'de Q0, çevrim 2-5 arasındaki
  discharge kapasitesinin medyanı olarak tanımlandı.
- **EOL tanımı düzeltildi**: Batarya ömrü, kapasitenin `0.85 * Q0` eşiğinin
  altına ilk düştüğü çevrim olarak tanımlandı.
- **Censored hücreler ayrıldı**: EOL eşiğine veri süresi içinde ulaşmayan
  hücreler modellenirken dışarıda bırakıldı, ancak sayıları ve etkileri ayrıca
  raporlandı.
- **Özellik seti temizlendi**: IR, sıcaklık veya farklı türevsel sinyallerle
  karışık eski özellikler yerine kapasite tabanlı, SOP uyumlu özellikler
  kullanıldı.
- **Veri bölme protokolü sabitlendi**: Hücre seviyesinde 70/15/15
  train/calibration/test bölmeleri oluşturuldu ve 5 farklı seed ile tekrarlandı.
- **Standardizasyon doğru yapıldı**: Z-score standardizasyon sadece train
  verisine fit edildi, calibration/test aynı scaler ile dönüştürüldü.

Bu düzeltmelerden sonra proje, tekrar üretilebilir bir SOPv2 pipeline haline
getirildi.

## 3. Repo ve Pipeline Yapısı

Aktif proje akışı şu klasörler üzerine kuruludur:

```text
0_data/       Raw veri indirme ve veri denetimi
1_features/   Kapasite tabanlı özellik çıkarımı
2_models/     Split üretimi, VIF analizi, modelleme deneyleri
3_analysis/   Shift, SHAP, survival, calibration, domain adaptation ve CP analizleri
docs/         Açıklama ve rapor notları
outputs/      Deney sonuçları, CSV/JSON/figür çıktıları
splits/       5 seed için train/calibration/test bölmeleri
legacy/       Eski kodlar ve arşiv materyalleri
```

Pipeline iki fazlı tasarlandı:

- **Faz A - Veri çıkarımı**: Büyük `.pkl` dosyalarından audit ve feature CSV
  dosyaları üretildi. Bu faz daha çok Colab/Drive üzerinden çalıştırılmak üzere
  düzenlendi.
- **Faz B - Modelleme ve analiz**: Yerel bilgisayarda küçük CSV dosyalarıyla
  modelleme, cross-dataset transfer, shift analizi, SHAP, calibration ve
  conformal prediction deneyleri yürütüldü.

Ana orkestrasyon dosyası:

```bash
python3 run_pipeline.py --status
python3 run_pipeline.py --phase model
python3 run_pipeline.py --phase analysis
```

## 4. Veri Hazırlama ve Audit Çalışmaları

İlk aşamada MATR ve HUST veri setleri için ayrı audit scriptleri hazırlandı.

MATR tarafında:

- Batch dosyaları okundu.
- Hücre bazında Q0 ve EOL hesaplandı.
- Censored hücreler tespit edildi.
- Kapasite retention özetleri çıkarıldı.

HUST tarafında:

- Hücrelerin discharge kapasite eğrileri düzenlendi.
- HUST'a özel bazı çevrim temizleme kontrolleri yapıldı.
- Q0 ve EOL değerleri aynı SOP mantığıyla hesaplandı.

İlgili ana dosyalar:

```text
0_data/build_matr_audit.py
0_data/build_hust_audit.py
data/intermediate/matr_cell_audit_strict.csv
data/intermediate/hust_threshold_audit.csv
data/intermediate/matr_cycles_tidy.csv
data/intermediate/hust_cycles_tidy.csv
```

Önemli sonuç:

- MATR veri setinde 6 hücre censored olarak tespit edildi.
- HUST veri setinde censored hücre bulunmadı.

## 5. Özellik Çıkarımı

Özellik çıkarımı kapasite tabanlı olacak şekilde yeniden kuruldu. Ana feature
tablosu:

```text
data/intermediate/features_sop12_combined.csv
```

Bu tabloda MATR ve HUST birlikte yer alır. Özellikler üç ana grupta
toparlanabilir:

1. **SOP12 kapasite özellikleri**:
   `Qdis_N`, `delta_Qdis`, `retention_ratio`, `slope_linear`,
   `variance_Qdis`, `range_Qdis`, `max_drop`, `std_diff`, `skewness_Qdis`,
   `slope_ratio`, `Qdis_cycle10`, `mean_diff`.

2. **Şekil ve bozunma özellikleri**:
   Polinom katsayıları, exponential decay, erken/geç eğim, knee cycle gibi
   kapasite eğrisinin şekline odaklanan ek özellikler.

3. **Entropy / FFT / ikinci türev özellikleri**:
   Kapasite eğrisinin pürüzlülüğü, değişkenliği ve frekans benzeri davranışını
   özetleyen ek özellikler.

Toplamda 34 kapasite tabanlı özellik kullanıldı.

Ayrıca Q0-normalized feature varyantları da denendi. Bu, özellikle iki veri
seti arasındaki kapasite ölçek farkını azaltmak için kullanıldı.

## 6. Split Protokolü

Modelleme için hücre seviyesinde split kullanıldı. Aynı hücrenin farklı
çevrimlerinden gelen bilgilerin train ve test tarafına karışmaması için split
hücre bazında yapıldı.

Kullanılan protokol:

- Train: %70
- Calibration: %15
- Test: %15
- 5 farklı seed: `42`, `123`, `456`, `789`, `1011`
- Lifetime quartile stratification

Split dosyaları:

```text
splits/sop_v2/matr_42.json
splits/sop_v2/hust_42.json
...
```

Bu yapı hem normal modelleme hem de conformal prediction için temel oluşturdu.

## 7. Within-Dataset Modelleme

Önce her veri seti kendi içinde değerlendirildi:

- MATR train -> MATR test
- HUST train -> HUST test

Kullanılan modeller:

- Elastic Net
- PLS
- Random Forest
- XGBoost
- CatBoost
- Gaussian Process
- Stacking ensemble

Varsayılan ana deneyde hedef değişken log-transform ile kullanıldı:

```bash
python3 2_models/run_experiments.py --log-target
```

Ana within-dataset sonuçları:

| Veri seti | En iyi model | MAE | R2 |
|---|---|---:|---:|
| MATR | CatBoost | 171.7 | 0.575 |
| HUST | Random Forest | 178.0 | 0.340 |

Yorum:

- MATR tarafında 34 özellik + log-target ile iyi bir kapasite-only baseline
  elde edildi.
- HUST tarafında mutlak hata benzer olsa da R2 daha düşük kaldı. Bunun önemli
  nedeni HUST ömür dağılımının daha dar olmasıdır; varyans az olduğunda R2 daha
  zor yükselir.

## 8. VIF, PCA ve Feature Ablation Denemeleri

Özellikler arasında çoklu bağlantı olup olmadığını görmek için VIF analizi
yapıldı. VIF sonuçları ana protokolü değiştirmek için değil, raporlayıcı bir
kontrol olarak kullanıldı.

Ayrıca şu ablation denemeleri yapıldı:

- 12 temel SOP özelliği
- 24 özellik
- 34 özellik
- VIF-pruned özellik setleri
- PCA varyantları
- Log-target açık/kapalı denemeleri

Genel sonuç:

- 34 özellik MATR within-dataset performansını artırdı.
- HUST için bazı daha kompakt feature setleri benzer veya daha iyi sonuçlar
  verebildi.
- Ancak within-dataset iyi çalışan feature setleri cross-dataset aktarımda
  otomatik olarak iyi çalışmadı.

## 9. Cross-Dataset Transfer Deneyleri

Projenin kritik kısmı, bir veri setinde eğitilen modelin diğer veri setinde
çalışıp çalışmadığını test etmekti:

- MATR -> HUST
- HUST -> MATR

Sonuçlar oldukça netti:

- Naive cross-dataset transfer her iki yönde de başarısız oldu.
- R2 değerleri negatif kaldı.
- Negatif R2, modelin hedef veri setinin ortalamasını tahmin etmekten bile kötü
  olduğunu gösterir.

Örnek sonuçlar:

| Yön | En iyi raw transfer R2 | Yorum |
|---|---:|---|
| HUST -> MATR | yaklaşık -2.05 ile -1.5 arası | Daha az kötü ama hala başarısız |
| MATR -> HUST | yaklaşık -8 civarı | Çok ciddi transfer çöküşü |

Bu bulgu projenin ana hikayesini belirledi: sorun sadece iyi model seçmek veya
daha fazla özellik eklemek değildi.

## 10. Distribution Shift Analizleri

İki veri seti arasındaki özellik dağılımlarının ne kadar farklı olduğunu ölçmek
için MMD ve Mahalanobis mesafesi hesaplandı.

Önemli bulgular:

- Ham özelliklerde iki veri seti arasında büyük fark vardı.
- `Qdis_cycle10`, `poly2_a`, `Qdis_N` gibi kapasite ölçekli özellikler çok
  yüksek dataset shift gösterdi.
- Q0-normalization bu farkın önemli bir kısmını azalttı.

Ancak önemli nokta şuydu:

Özellik dağılımlarını yakınlaştırmak cross-dataset transfer problemini tamamen
çözmedi. Bu nedenle problem yalnızca covariate shift olarak açıklanamadı.

## 11. Feature Transfer Stability Analizi

Her özelliğin sadece dağılımı değil, ömürle ilişkisi de incelendi.

Analiz edilen noktalar:

- Özelliğin MATR ve HUST dağılımları ne kadar farklı?
- Özelliğin ömürle korelasyonu veri setleri arasında korunuyor mu?
- Özellik tek başına within-dataset tahmin için işe yarıyor mu?
- Residual target calibration sonrası cross-dataset durumda faydalı oluyor mu?

Bu analiz sonucunda bazı özelliklerin within-dataset için önemli ama
cross-dataset için kırılgan olduğu görüldü.

Örnek:

- `Qdis_cycle10`, `Qdis_N`, `poly2_a` within-domain modellerde önemli olabilir.
- Ancak bu özellikler dataset ölçek farkına çok duyarlıdır.

Bu bulgu SHAP analiziyle de birleştirildi.

## 12. SHAP / XAI Analizi

Within-dataset modellerin hangi özelliklere dayandığını anlamak için SHAP
analizi yapıldı.

Ana modeller:

- MATR: CatBoost
- HUST: Random Forest

Önemli yorum:

Modeller within-dataset içinde anlamlı sinyaller öğreniyor. Ancak modellerin
önem verdiği bazı özellikler cross-dataset transfer açısından stabil değil.

Bu nedenle "model kötü öğrendi" demek yerine daha doğru açıklama şudur:

> Model kendi veri setinde gerçek sinyal öğreniyor, fakat bu sinyal diğer veri
> setinde aynı ömür ilişkisine karşılık gelmiyor.

Bu, concept shift anlatısını destekler.

## 13. Survival ve Censoring Sensitivity

MATR veri setinde EOL eşiğine ulaşmayan 6 censored hücre olduğu için bunun
sonuçları bozup bozmadığı kontrol edildi.

Yapılanlar:

- Kaplan-Meier survival eğrileri çıkarıldı.
- Log-rank testi yapıldı.
- Censored hücreler lower-bound olarak dahil edildiğinde dağılım farkının
  değişip değişmediği incelendi.

Sonuç:

- Censoring, MATR-HUST ömür farkını açıklamıyor.
- İki veri setinin ömür dağılımı belirgin biçimde farklı kalıyor.

## 14. Concept Shift ve Conditional Shift Analizleri

Danışman geri bildirimiyle de netleşen en önemli kavram budur:

Bu projedeki cross-dataset problem sadece "target shift" veya "2.09 kat ömür
farkı" değildir. Daha doğru ifade **concept shift** veya conditional shift'tir.

Yani:

> Aynı özellikler, MATR ve HUST veri setlerinde batarya ömrüyle aynı ilişkiye
> sahip değildir.

Yapılan analizlerde:

- HUST ve MATR arasında yaklaşık 2.09 kat merkezi ömür farkı bulundu.
- Bu fark sadece merkezi bir ölçek farkıdır.
- Asıl problem, bazı özelliklerin ömürle ilişkisinin veri setleri arasında
  değişmesidir.
- 34 özellikten 16'sında slope-shift görüldü.
- HUST -> MATR yönünde zayıf da olsa rank signal korunurken, MATR -> HUST
  yönünde transfer sinyali neredeyse kayboldu.

Bu nedenle rapor anlatısında "concept shift" terimi kullanılmalıdır.

## 15. Target Calibration / Target Rescaling

Naive cross transfer kötü olduğu için küçük sayıda hedef veri seti etiketiyle
kalibrasyon denendi.

Temel fikir:

1. Model source veri setinde eğitilir.
2. Target veri setinde az sayıda etiketli hücre seçilir.
3. Modelin target tahminleri bu küçük kalibrasyon setiyle düzeltilir.
4. Geri kalan target hücrelerde performans ölçülür.

Denemeler:

- k=5 target hücre
- k=10 target hücre
- k=20 target hücre

Sonuç:

- Target calibration, raw transfer'a göre çok büyük iyileşme sağladı.
- R2 değerleri ciddi negatiflerden sıfıra yakın değerlere geldi.
- Bu, concept shift altında az sayıda target etiketi kullanmanın faydalı
  olduğunu gösterdi.

Örnek:

| Yön | Raw transfer | k=20 target calibration |
|---|---:|---:|
| HUST -> MATR | negatif R2 | yaklaşık -0.05 / -0.11 aralığı |
| MATR -> HUST | çok negatif R2 | yaklaşık -0.02 / -0.13 aralığı |

## 16. Domain Adaptation ve CORAL Denemeleri

Covariate alignment'ın tek başına yeterli olup olmadığını test etmek için
domain adaptation deneyleri yapıldı.

Denemeler:

- Source-only MLP
- CORAL MLP
- MMD MLP
- CORAL + residual target calibration

Burada CORAL ana çözüm olarak değil, bir baseline olarak kullanıldı:

> CORAL, source ve target feature dağılımlarını hizalamanın tek başına yeterli
> olup olmadığını test eden bir kontroldür.

Sonuç:

- CORAL-only bazı yönlerde raw MLP'ye göre iyileşse de R2 negatif kaldı.
- Yani sadece feature distribution alignment yeterli olmadı.
- CORAL + target calibration daha iyi çalıştı.
- Bu da ana iddiayı güçlendirdi: hedef veri setinden az sayıda etiketli örnek
  kullanmak, yalnızca covariate alignment yapmaktan daha etkilidir.

Son danışman geri bildirimi sonrası ayrıca şu script eklendi:

```text
3_analysis/coral_source_conformal.py
```

Bu script, CORAL sonrası source-domain conformal prediction'ın target domain'de
güvenilir olup olmadığını test eder.

## 17. Top-k Feature Sweep

Danışman geri bildirimi doğrultusunda SHAP sıralamasına göre top-k feature
sweep yapıldı.

Denenen k değerleri:

- k=3
- k=6
- k=10
- k=12

Sonuçlar:

- HUST within-dataset tarafında az sayıda özellik ile full feature set'e yakın
  performans alınabildi.
- MATR tarafında k arttıkça performans genel olarak iyileşti.
- Cross-dataset tarafında top-k seçimi negatif R2 problemini çözmedi.

Bu sonuç şu anlatıyı destekler:

> Top-k feature selection, within-dataset için kompakt ve kullanışlı bir özellik
> seti sağlayabilir; ancak cross-dataset concept shift problemini tek başına
> çözmez.

Sonuç dosyaları:

```text
outputs/results_v2_topk_shap/topk_within_best.csv
outputs/results_v2_topk_shap/topk_cross_best.csv
```

## 18. Conformal Prediction Çalışmaları

Modelin sadece nokta tahmini değil, belirsizlik aralığı üretmesi için conformal
prediction çalışmaları yapıldı.

Kullanılan temel yöntem:

- MAPIE Split Conformal Prediction
- 90% ve 95% confidence levels
- Source calibration
- Target-domain calibration
- Target-adapted calibration
- Short-life / long-life coverage ayrımı

Karşılaştırılan senaryolar:

1. **Within-dataset CP**: Train/calibration/test aynı veri setinde.
2. **Source CP**: Source calibration ile target üzerinde interval üretme.
3. **CORAL sonrası Source CP**: CORAL hizalaması sonrası source calibration ile
   target üzerinde interval üretme.
4. **Target-domain CP**: Target veri setinden az sayıda calibration hücresiyle
   interval üretme.
5. **Target-adapted CP**: Önce target tarafında residual mean adapter, sonra
   ayrı target calibration setiyle conformal interval.

Ana sonuç:

- Within-dataset CP nominal coverage'a yakın çalıştı.
- Source CP cross-dataset altında ciddi undercoverage gösterdi.
- CORAL sonrası Source CP de target domain'de yeterince güvenilir olmadı.
- Target-domain CP coverage'ı toparladı.
- Target-adapted CP coverage'ı korurken daha kullanışlı interval genişlikleri
  verdi.

Son danışman isteği sonrası üretilen conformal karşılaştırma tablosu:

```text
outputs/advisor_tables/advisor_conformal_comparison.md
```

Yeni CORAL-after-source CP bulgusu:

- 90% hedef coverage için HUST -> MATR yaklaşık 0.526 coverage verdi.
- 90% hedef coverage için MATR -> HUST yaklaşık 0.296 coverage verdi.

Bu sonuç, CORAL hizalamasının source calibration problemini tek başına
çözmediğini gösterir.

## 19. Importance-Weighted Conformal Prediction

Covariate shift varsayımı altında source calibration residual'larını
ağırlıklandırarak conformal prediction yapmanın işe yarayıp yaramadığı test
edildi.

Yapılanlar:

- Source/target ayrımı için logistic dataset discriminator eğitildi.
- Density ratio benzeri importance weight'ler hesaplandı.
- Weight clipping sweep yapıldı.
- Effective sample size incelendi.

Sonuç:

- Dataset discriminator AUC çok yüksekti; yani feature distribution farkı çok
  belirgin.
- Importance-weighted CP çoğu durumda ancak sonsuz veya aşırı geniş intervaller
  ile coverage sağlayabildi.
- Bu nedenle pratik bir çözüm olmadı.

Bu analiz de şu sonucu destekledi:

> Problem sadece covariate shift değildir; target-domain calibration gereklidir.

## 20. Koopman / DMD Pilot Analizi

Erken kapasite eğrilerinin dinamik yapısını incelemek için Hankel-DMD /
Koopman tarzı bir pilot analiz yapıldı.

Amaç:

- MATR ve HUST erken kapasite yörüngeleri aynı dinamik yapıya mı sahip?
- DMD eigenvalue ve mode bilgileri veri seti kimliği taşıyor mu?
- Source veri setinden öğrenilen dinamik operatör target veri setine iyi
  aktarılıyor mu?

Sonuç:

- DMD özetleri dataset kimliği hakkında anlamlı bilgi taşıdı.
- Dinamik seviyede de veri setleri arasında fark olduğu görüldü.
- Bu analiz ana modelleme sonucu değil, concept shift anlatısını destekleyen
  açıklayıcı bir pilot olarak değerlendirildi.

## 21. Son Danışman Geri Bildirimi Sonrası Yapılanlar

Danışman son e-postasında özellikle şunları istedi:

1. Tüm özellikler, top-k, raw transfer, CORAL-only, target calibration ve
   CORAL + target calibration sonuçlarını tek karşılaştırma tablosunda toplamak.
2. Conformal prediction tarafında source CP, CORAL sonrası source CP,
   target-domain CP ve target-adapted CP karşılaştırmasını eklemek.
3. "Target shift" yerine "concept shift" terminolojisini kullanmak.
4. CORAL'i ana çözüm değil, covariate alignment baseline'ı olarak sunmak.

Bu istekler için şu dosyalar eklendi:

```text
3_analysis/coral_source_conformal.py
3_analysis/make_advisor_comparison_tables.py
```

Üretilen yeni advisor-facing çıktılar:

```text
outputs/advisor_tables/advisor_point_comparison.md
outputs/advisor_tables/advisor_point_comparison.csv
outputs/advisor_tables/advisor_conformal_comparison.md
outputs/advisor_tables/advisor_conformal_comparison.csv
outputs/advisor_tables/advisor_followup_narrative.md
outputs/results_v2_coral_source_cp/results_summary.csv
```

Point prediction karşılaştırma tablosunda şunlar birlikte verildi:

- All features, within-dataset
- Best top-k, within-dataset
- Raw transfer, all features
- Best top-k raw transfer
- CORAL-only
- Target calibration
- CORAL + target calibration

Conformal prediction karşılaştırma tablosunda şunlar birlikte verildi:

- Source CP
- CORAL-after-source CP
- Target-domain CP
- Target-adapted CP

Bu son eklemeler, projenin danışman tarafından önerilen savunulabilir anlatıya
uygun hale getirilmesini sağladı.

## 22. Şu Anki Ana Bilimsel Anlatı

Projenin bugünkü ana sonucu şu şekilde özetlenebilir:

1. **Within-dataset tahmin mümkündür**:
   MATR ve HUST içinde erken kapasite özellikleri batarya ömrü hakkında anlamlı
   sinyal taşır.

2. **Top-k feature selection faydalıdır ama sınırlıdır**:
   Özellikle HUST için az sayıda özellikle full feature set'e yakın performans
   alınabilir. Ancak top-k seçimi cross-dataset transfer problemini çözmez.

3. **Cross-dataset raw transfer başarısızdır**:
   MATR -> HUST ve HUST -> MATR yönlerinde R2 negatif kalır.

4. **Problem sadece feature distribution farkı değildir**:
   Q0-normalization, CORAL ve importance weighting gibi covariate alignment
   yaklaşımları tek başına yeterli olmaz.

5. **Doğru kavram concept shift'tir**:
   Aynı özellikler iki veri setinde ömürle aynı ilişkiye sahip değildir.

6. **Target-side calibration güçlü bir çözümdür**:
   Az sayıda target etiketi ile model tahminleri ciddi şekilde iyileşir.

7. **Conformal prediction için target-domain calibration gerekir**:
   Source CP cross-dataset altında undercoverage verir. Target-domain CP ve
   target-adapted CP coverage'ı toparlar.

## 23. Önemli Sonuç Dosyaları

Ana feature ve split dosyaları:

```text
data/intermediate/features_sop12_combined.csv
splits/sop_v2/*.json
```

Within-dataset sonuçları:

```text
outputs/results_v2_34feat_log/results_summary.csv
```

Top-k sonuçları:

```text
outputs/results_v2_topk_shap/topk_within_best.csv
outputs/results_v2_topk_shap/topk_cross_best.csv
```

Target calibration:

```text
outputs/results_v2_target_rescale/results_summary.csv
```

CORAL + target calibration:

```text
outputs/results_v2_coral_target_calibration/results_summary.csv
```

Conformal prediction:

```text
outputs/results_v2_conformal/results_summary.csv
outputs/results_v2_conformal/paper_cp_summary.md
```

Son advisor tabloları:

```text
outputs/advisor_tables/advisor_point_comparison.md
outputs/advisor_tables/advisor_conformal_comparison.md
outputs/advisor_tables/advisor_followup_narrative.md
```

## 24. Rapor ve Sunum İçin Önerilen Akış

Rapor veya sunum şu sırayla kurulabilir:

1. Problem: Erken çevrimlerden batarya ömrü tahmini.
2. Veri setleri: MATR ve HUST; iki farklı domain.
3. SOP düzeltmeleri: Q0, EOL, censoring, feature definitions, splits.
4. Within-dataset sonuçları: Modeller kendi veri setlerinde anlamlı performans
   veriyor.
5. Cross-dataset raw transfer: Negatif R2, başarısız aktarım.
6. Shift analizleri: Özellik dağılımları farklı, ancak sorun bununla sınırlı
   değil.
7. Concept shift: Aynı özelliklerin ömürle ilişkisi veri setleri arasında
   değişiyor.
8. Top-k: Kompakt feature set within-dataset için faydalı, transfer için
   yetersiz.
9. CORAL: Covariate alignment baseline; tek başına yeterli değil.
10. Target calibration: Az sayıda target etiketiyle büyük iyileşme.
11. Conformal prediction: Source CP başarısız, target-domain/adapted CP daha
    güvenilir.
12. Sonuç: Concept shift altında target-side calibration ve target-domain
    uncertainty calibration en savunulabilir yaklaşım.

## 25. Kısa Sonuç

Bu projede başlangıçtaki veri tanımı ve protokol problemleri düzeltilerek
tekrar üretilebilir bir SOPv2 batarya ömrü tahmin pipeline'ı oluşturuldu.
Within-dataset tahminlerde anlamlı performans elde edildi. Ancak cross-dataset
transfer deneyleri, modellerin bir veri setinden diğerine doğrudan
aktarılamadığını gösterdi. Shift analizleri, SHAP, top-k sweep, CORAL,
importance-weighted CP ve conformal prediction deneyleri birlikte
değerlendirildiğinde ana sonuç şudur:

> MATR ve HUST arasında sadece kapasite ölçeği veya feature distribution farkı
> yoktur; aynı özelliklerin batarya ömrüyle ilişkisi değişmektedir. Bu nedenle
> problem concept shift olarak sunulmalı, CORAL gibi yöntemler covariate
> alignment baseline'ı olarak değerlendirilmeli, az sayıda target etiketiyle
> yapılan calibration ve target-domain conformal prediction ise en güçlü
> pratik çözüm olarak raporlanmalıdır.
