# Giriş, Metodoloji ve Tartışma Taslağı

## 1. Giriş

### Araştırma Motivasyonu

Lityum-iyon bataryaların çevrim ömrünün güvenilir biçimde tahmin edilmesi, elektrikli araçlar, enerji depolama sistemleri ve batarya yönetim sistemleri açısından kritik bir problemdir. Geleneksel yaklaşımda bir hücrenin gerçek ömrünü gözlemlemek için kapasitenin belirli bir sağlık eşiğinin altına düşmesi beklenir; ancak bu süreç yüzlerce veya binlerce çevrim sürebilir. Bu nedenle, bataryanın yalnızca erken çevrim davranışından nihai ömrünü tahmin etmek hem deney süresini kısaltmak hem de hücre seçim, kalite kontrol ve bakım kararlarını hızlandırmak açısından önemlidir.

Bu çalışmada, erken çevrim kapasite verilerinden batarya ömrü tahmini yapılması hedeflenmiştir. Erken çevrim tahmini, bataryanın tüm yaşlanma sürecini beklemeden ilk belirli çevrimlerdeki kapasite eğilimi, bozunma hızı, eğrilik, varyans ve benzeri özelliklerin gelecekteki ömürle ilişkisini modellemeye dayanır. Bu yaklaşım, özellikle üretim sonrası hücre sınıflandırma ve hızlı performans değerlendirme süreçleri için pratik bir değer taşımaktadır.

### Veri Setleri ve Literatürdeki Önemi

Çalışmada iki halka açık LFP batarya veri seti kullanılmıştır: MATR/Severson ve HUST. MATR veri seti, batarya ömrü tahmini literatüründe erken çevrim verilerinden ömür kestirimi için en yaygın kullanılan referans veri setlerinden biridir. HUST veri seti ise farklı deney koşulları, farklı kapasite ölçeği ve farklı ömür dağılımı nedeniyle cross-dataset genelleme açısından değerli bir ikinci domain sağlamaktadır.

Bu iki veri setinin birlikte kullanılması, yalnızca aynı veri seti içinde tahmin başarısını değil, bir veri setinde öğrenilen ilişkinin başka bir veri setine aktarılıp aktarılamadığını da test etmeye olanak tanır. Bu yönüyle çalışma, klasik within-dataset performans değerlendirmesinin ötesine geçerek batarya ömrü modellerinin domain değişimi altında ne kadar güvenilir olduğunu incelemektedir.

### Temel Araştırma Soruları

Bu çalışmada dört temel araştırma sorusu ele alınmıştır. İlk olarak, erken çevrim kapasite özellikleri kullanılarak MATR ve HUST veri setleri içinde batarya ömrünün ne kadar iyi tahmin edilebildiği araştırılmıştır. İkinci olarak, bir veri setinde eğitilen modellerin diğer veri setine doğrudan aktarılıp aktarılamadığı test edilmiştir. Üçüncü olarak, cross-dataset başarısızlığın yalnızca özellik dağılımlarındaki farktan mı, yoksa aynı özelliklerin ömürle ilişkisinin veri setleri arasında değişmesinden mi kaynaklandığı incelenmiştir. Son olarak, az sayıda hedef-domain etiketi kullanılarak hem nokta tahmini hem de belirsizlik tahmini performansının iyileştirilip iyileştirilemeyeceği değerlendirilmiştir.

Bu bağlamda çalışmanın ana bilimsel vurgusu **concept shift** üzerinedir. Bulgular, MATR ve HUST arasında yalnızca merkezi bir ömür ölçeği farkı bulunmadığını; aynı erken çevrim özelliklerinin farklı veri setlerinde batarya ömrüyle aynı ilişkiye sahip olmadığını göstermektedir. Bu nedenle problem yalnızca covariate shift veya target shift olarak değil, conditional/concept shift problemi olarak ele alınmıştır.

## 2. Metodoloji

### SOPv2 Protokolü ve Veri Tanımı Düzeltmeleri

Çalışmanın metodolojik temeli, başlangıçtaki veri tanımı ve modelleme protokolü problemlerini gideren SOPv2 protokolü üzerine kurulmuştur. İlk olarak Q0 tanımı standartlaştırılmıştır. Q0, ilk pozitif kapasite değeri yerine çevrim 2-5 arasındaki discharge kapasitesinin medyanı olarak tanımlanmıştır. Bu seçim, ilk çevrimlerdeki ölçüm gürültüsünü azaltarak daha kararlı bir başlangıç kapasitesi sağlamıştır.

İkinci olarak EOL tanımı düzeltilmiştir. Batarya ömrü, discharge kapasitesinin `0.85 * Q0` eşiğinin altına ilk düştüğü çevrim olarak tanımlanmıştır. Bu tanım, tüm hücreler için aynı sağlık eşiğine dayanan tutarlı bir etiketleme süreci sağlamıştır. EOL eşiğine gözlem süresi içinde ulaşmayan censored hücreler modelleme aşamasında dışarıda bırakılmış, ancak survival/censoring analizleriyle ayrıca değerlendirilmiştir.

Üçüncü olarak veri bölme protokolü hücre seviyesinde sabitlenmiştir. Train, calibration ve test ayrımı %70/%15/%15 oranında yapılmış; aynı hücreye ait bilgilerin farklı bölmelere sızması engellenmiştir. Bölmeler beş farklı seed ile tekrar edilmiş ve lifetime quartile stratification kullanılmıştır. Böylece sonuçların tek bir rastgele bölmeye bağımlı kalması önlenmiştir.

### İki Fazlı Pipeline Yapısı

Metodolojik iş akışı iki fazlı bir pipeline olarak tasarlanmıştır. Birinci faz, ham `.pkl` dosyalarından veri audit ve feature extraction çıktılarının üretilmesini kapsamaktadır. Bu fazda MATR ve HUST veri setleri ayrı ayrı okunmuş, Q0 ve EOL hesaplanmış, censored hücreler belirlenmiş ve kapasite eğrileri düzenlenmiştir. Sonuçta modellemeye hazır ara dosyalar ve birleşik feature tablosu oluşturulmuştur.

İkinci faz, modelleme ve analiz aşamasıdır. Bu aşamada `features_sop12_combined.csv` dosyası kullanılarak split üretimi, VIF analizi, within-dataset modelleme, cross-dataset transfer, shift ölçümleri, SHAP analizi, target calibration, domain adaptation ve conformal prediction deneyleri yürütülmüştür. Bu ayrım sayesinde ham veriye tekrar dönmeden farklı modelleme ve analiz deneyleri hızlı biçimde tekrarlanabilir hale getirilmiştir.

### Özellik Çıkarımı ve Modelleme

Özellik çıkarımı kapasite tabanlı olacak şekilde yapılandırılmıştır. Temel SOP12 özelliklerine ek olarak kapasite eğrisinin şekli, bozunma dinamiği, varyansı, eğriliği, entropy ve frekans-benzeri davranışlarını özetleyen ek özellikler dahil edilmiştir. Toplamda 34 kapasite tabanlı özellik kullanılmıştır. Modeller arasında Elastic Net, PLS, Random Forest, XGBoost, CatBoost, Gaussian Process ve Stacking ensemble yer almıştır. Ana deneylerde hedef değişken log-transform ile modellenmiş, performans metrikleri orijinal cycle-life uzayında raporlanmıştır.

### Conformal Prediction Senaryoları

Belirsizlik tahmini için MAPIE tabanlı split conformal prediction yaklaşımı kullanılmıştır. Within-dataset senaryoda model, calibration ve test örnekleri aynı veri setinden alınmıştır. Source CP senaryosunda model ve conformal calibration source veri setinde yapılmış, interval performansı target veri setinde ölçülmüştür. CORAL-after-source CP senaryosunda source ve target feature temsilleri CORAL ile hizalanmış, ancak calibration yine source domain üzerinde yapılmıştır. Target-domain CP senaryosunda az sayıda target etiketi conformal calibration için kullanılmıştır. Target-adapted CP senaryosunda ise önce target tarafında residual-mean adapter uygulanmış, ardından ayrı bir target calibration setiyle interval üretilmiştir.

Bu senaryolar, source-domain belirsizlik kalibrasyonunun dataset shift altında geçerli olup olmadığını ve az sayıda target etiketiyle conformal validity'nin ne ölçüde geri kazanılabildiğini test etmek için tasarlanmıştır.

## 3. Tartışma

### Cross-Dataset Raw Transfer Neden Başarısız Oldu?

Cross-dataset raw transfer deneyleri, MATR ve HUST arasında doğrudan model aktarımının başarısız olduğunu göstermiştir. Within-dataset deneylerde modeller anlamlı performans üretmiş; örneğin MATR için CatBoost, HUST için Random Forest en iyi sonuçları vermiştir. Buna rağmen aynı modeller diğer veri setine doğrudan aktarıldığında R2 değerleri negatif kalmıştır. Negatif R2, modelin hedef veri setinin ortalamasını tahmin etmekten bile daha kötü performans verdiğini göstermektedir.

Bu bulgu, problemin yalnızca model kapasitesi veya model seçimiyle açıklanamayacağını göstermektedir. Modeller kendi domain'lerinde anlamlı sinyal öğrenmiştir; ancak öğrenilen feature-to-lifetime ilişkisi diğer veri setinde korunmamıştır. Dolayısıyla başarısızlık, modellerin "kötü öğrenmesi"nden ziyade, source ve target domain arasındaki ilişkinin yapısal olarak değişmesinden kaynaklanmaktadır.

### Covariate Alignment Olarak CORAL'in Sınırı

CORAL deneyleri, feature dağılımlarını hizalamanın tek başına yeterli olup olmadığını test eden bir baseline olarak kullanılmıştır. CORAL'in amacı, source ve target temsilleri arasındaki covariance farkını azaltarak covariate shift etkisini hafifletmektir. Ancak sonuçlar, CORAL-only yaklaşımının cross-dataset negatif R2 problemini çözmediğini göstermiştir.

Benzer şekilde, CORAL sonrası source-domain conformal prediction da hedef domain'de güvenilir coverage sağlayamamıştır. 90% hedef coverage için CORAL-after-source CP'nin HUST -> MATR yönünde yaklaşık 0.526, MATR -> HUST yönünde ise yaklaşık 0.296 coverage vermesi, feature alignment yapılsa bile source calibration'ın target domain için yeterli olmadığını göstermektedir. Bu sonuç, covariate alignment'ın önemli bir kontrol olduğunu, ancak ana çözüm olarak sunulmaması gerektiğini desteklemektedir.

### Target-Side Calibration'ın Gücü

Target-side calibration deneyleri, az sayıda hedef veri seti etiketi kullanmanın cross-dataset transfer performansını belirgin biçimde iyileştirdiğini göstermiştir. k=20 target hücresiyle yapılan calibration sonrası R2 değerleri ciddi negatif seviyelerden sıfıra yakın değerlere gelmiştir. Bu sonuç, hedef domain'den az miktarda etiketli örneğin, source modelin sistematik hatasını düzeltmek için son derece değerli olduğunu göstermektedir.

Bu bulgu concept shift yorumuyla uyumludur. Eğer problem yalnızca feature distribution farkı olsaydı, CORAL veya importance weighting gibi covariate alignment yaklaşımlarının yeterli olması beklenirdi. Ancak asıl iyileşme target-side calibration ile elde edilmiştir. Bu da hedef domain'deki feature-to-lifetime ilişkisinin doğrudan kalibre edilmesi gerektiğini göstermektedir.

### Importance-Weighted CP ve Covariate Shift Varsayımının Yetersizliği

Importance-weighted conformal prediction, covariate shift varsayımı altında source calibration residual'larını target domain'e daha uygun hale getirmeyi amaçlamıştır. Ancak dataset discriminator AUC değerlerinin çok yüksek olması, MATR ve HUST feature dağılımlarının kolayca ayırt edilebildiğini göstermiştir. Buna rağmen importance weighting pratik, dar ve güvenilir interval üretmekte başarısız olmuştur; coverage çoğu durumda ancak aşırı geniş veya sonsuz intervallerle sağlanabilmiştir.

Bu sonuç, covariate shift düzeltmesinin tek başına yeterli olmadığını açık biçimde göstermektedir. Source residual dağılımını ağırlıklandırmak, target domain'deki conditional ilişki değiştiğinde yeterli bir çözüm sunmamaktadır. Buna karşılık target-domain CP ve target-adapted CP nominal coverage'a daha yakın sonuçlar üretmiştir.

### Concept Shift Kanıtı

Çalışmada concept shift birkaç bağımsız kanıt hattıyla desteklenmiştir. İlk olarak, cross-dataset raw transfer'ın sistematik biçimde negatif R2 vermesi, öğrenilen ilişkinin domain dışında geçersizleştiğini göstermiştir. İkinci olarak, Q0-normalization ve CORAL gibi feature distribution alignment yöntemleri performans problemini tek başına çözmemiştir. Üçüncü olarak, SHAP ve feature stability analizleri, within-domain için önemli olan bazı özelliklerin cross-domain ilişkiler açısından kırılgan olduğunu göstermiştir. Dördüncü olarak, conditional shift analizlerinde 34 özellikten önemli bir kısmında slope-shift gözlenmiştir.

Bu nedenle çalışmanın ana yorumu şudur: MATR ve HUST arasında yalnızca merkezi ömür farkı veya kapasite ölçeği farkı yoktur. Aynı erken çevrim kapasite özellikleri, iki veri setinde batarya ömrüyle aynı fonksiyonel ilişkiye sahip değildir. Bu durum, cross-dataset raw transfer'ın neden başarısız olduğunu ve target-domain calibration'ın neden daha etkili olduğunu açıklamaktadır.
