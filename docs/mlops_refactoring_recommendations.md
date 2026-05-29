# Refactoring ve MLOps Önerileri

Bu doküman, mevcut iki fazlı batarya ömrü tahmin pipeline'ını daha modüler,
konfigürasyon tabanlı, izlenebilir ve tekrar üretilebilir hale getirmek için
önerilen sonraki mimari adımları özetler.

## 1. Mevcut Mimari Değerlendirme

Mevcut proje, bitirme projesi kapsamı için güçlü bir mühendislik seviyesine
ulaşmıştır:

- Ham veri işleme, feature extraction, split üretimi, modelleme ve analiz
  aşamaları ayrı klasörlere bölünmüştür.
- Büyük ham veri işleme Faz A'da, hafif modelleme/analiz işleri Faz B'de
  konumlandırılmıştır.
- Üretilen CSV/JSON çıktıları klasörlenmiş ve çoğu deney tekrar
  çalıştırılabilir hale getirilmiştir.
- 5 seed'li split protokolü sabitlenmiş, böylece sonuçlar tek bir rastgele
  bölmeye bağımlı kalmamıştır.

Bir sonraki kalite seviyesi için temel hedef, script merkezli akışı
**config-driven experiment platform** haline getirmektir.

## 2. Önerilen Hedef Mimari

Önerilen üst seviye yapı:

```text
configs/
├── data/
│   ├── matr_hust.yaml
│   └── paths.yaml
├── features/
│   ├── sop12.yaml
│   ├── extended34.yaml
│   └── topk.yaml
├── experiments/
│   ├── within_dataset.yaml
│   ├── cross_dataset.yaml
│   ├── target_calibration.yaml
│   ├── coral.yaml
│   └── conformal.yaml
├── models/
│   ├── catboost.yaml
│   ├── random_forest.yaml
│   ├── xgboost.yaml
│   └── elastic_net.yaml
└── tracking/
    ├── local.yaml
    ├── mlflow.yaml
    └── wandb.yaml

src/
├── battery_life/
│   ├── data/
│   ├── features/
│   ├── splits/
│   ├── models/
│   ├── evaluation/
│   ├── calibration/
│   ├── conformal/
│   ├── tracking/
│   └── pipelines/
└── cli.py
```

Bu yapıda scriptler korunabilir, ancak ortak fonksiyonlar `src/battery_life/`
altına taşınarak tekrar kullanılabilir hale getirilmelidir.

## 3. YAML veya Hydra ile Konfigürasyon Yönetimi

Mevcut scriptlerde birçok parametre komut satırından verilmektedir:

- seed listesi
- feature set
- model listesi
- output directory
- confidence level
- target calibration k değerleri
- CORAL epoch ve lambda parametreleri

Bunlar YAML tabanlı konfigürasyonlara taşınabilir.

Örnek deney konfigürasyonu:

```yaml
experiment:
  name: cross_dataset_34feat_log
  task: cross_dataset
  source_datasets: ["matr", "hust"]
  target_datasets: ["matr", "hust"]
  seeds: [42, 123, 456, 789, 1011]
  n_cycles: [100]

data:
  features_path: data/intermediate/features_sop12_combined.csv
  splits_dir: splits/sop_v2
  target_column: cycle_life
  log_target: true

features:
  mode: all_non_metadata
  capacity_normalized: false

models:
  include:
    - catboost
    - random_forest
    - gaussian_process
    - elastic_net

outputs:
  root: outputs/runs
  save_detailed: true
  save_summary: true
```

Hydra kullanılırsa deneyler şu şekilde çalıştırılabilir:

```bash
python -m battery_life.cli experiment=cross_dataset_34feat_log
python -m battery_life.cli experiment=conformal target_k_values="[5,10,15,20]"
```

Bu yaklaşımın faydaları:

- Deney ayarları koddan ayrılır.
- Aynı script farklı deneyler için tekrar kullanılabilir.
- Her deneyin tam parametreleri çıktılarla birlikte saklanır.
- Danışmana veya jüriye "bu tablo hangi ayarlarla üretildi?" sorusunun net
  cevabı verilebilir.

## 4. Split Yönetimini Modüler Hale Getirme

Mevcut `splits/sop_v2/*.json` yapısı iyi bir başlangıçtır. Sonraki aşamada
split yönetimi için bir manifest eklenebilir:

```text
splits/
└── sop_v2/
    ├── manifest.yaml
    ├── matr_42.json
    ├── matr_123.json
    └── ...
```

Örnek manifest:

```yaml
protocol: sop_v2
split_type: cell_level
ratios:
  train: 0.70
  calibration: 0.15
  test: 0.15
stratification: lifetime_quartile
seeds: [42, 123, 456, 789, 1011]
datasets: ["matr", "hust"]
created_by: 2_models/generate_splits.py
```

Bu sayede split dosyaları yalnızca JSON listeleri olmaktan çıkar, protokol
bilgisiyle birlikte versioned artifact haline gelir.

## 5. Output Klasörlerinin Standartlaştırılması

Şu anda çıktı klasörleri anlamlıdır ancak deney sayısı arttıkça yönetimi
zorlaşabilir. Önerilen yapı:

```text
outputs/
└── runs/
    └── 2026-05-30_1530_cross_dataset_34feat_log/
        ├── config.yaml
        ├── metrics_summary.csv
        ├── metrics_detailed.csv
        ├── predictions.parquet
        ├── artifacts/
        │   ├── plots/
        │   └── tables/
        └── run_metadata.json
```

Her run klasöründe en az şu dosyalar bulunmalıdır:

- `config.yaml`: Deney parametrelerinin tam kopyası.
- `metrics_summary.csv`: Grup ortalamaları.
- `metrics_detailed.csv`: Seed/repeat seviyesinde detaylar.
- `run_metadata.json`: Git commit, tarih, Python versiyonu, paket versiyonları.
- `predictions.parquet`: Mümkünse test tahminleri ve gerçek değerler.

Bu yapı, sonuçların yalnızca özet metrik olarak değil, ham tahmin düzeyinde de
denetlenebilmesini sağlar.

## 6. MLflow veya WandB Entegrasyonu

Bir sonraki MLOps adımı deney izleme aracıdır.

### MLflow

Yerel ve akademik projeler için MLflow iyi bir seçenektir. Hafif, dosya tabanlı
çalışabilir ve sunucu gerektirmeden kullanılabilir.

Önerilen kayıtlar:

- Parametreler:
  - dataset
  - source/target
  - seed
  - model
  - feature_set
  - n_cycles
  - log_target
  - k_target
  - confidence_level
- Metrikler:
  - MAE
  - sMAPE
  - R2
  - coverage
  - median_width
  - Winkler score
- Artifact'ler:
  - result CSV dosyaları
  - plots
  - advisor markdown tabloları
  - config YAML

Örnek wrapper:

```python
import mlflow

with mlflow.start_run(run_name=config.experiment.name):
    mlflow.log_params(flatten_config(config))
    mlflow.log_metric("R2_mean", summary["R2_mean"].max())
    mlflow.log_metric("MAE_mean", summary["MAE_mean"].min())
    mlflow.log_artifact(summary_path)
    mlflow.log_artifact(config_path)
```

### WandB

WandB, görsel dashboard ve remote tracking için daha kullanışlıdır. Özellikle
çok sayıda model, seed ve k sweep karşılaştırması yapılacaksa iyi bir tercih
olabilir. Ancak tez teslimi için MLflow daha sade ve lokal kalabilir.

Pratik öneri:

- İlk aşamada MLflow local tracking kullanılmalı.
- Daha sonra istenirse WandB opsiyonel backend olarak eklenmeli.

## 7. Logging ve Run Metadata

Mevcut scriptlerde `print` tabanlı çıktı kullanılıyor. Sonraki aşamada standart
`logging` modülüne geçilmesi önerilir.

Önerilen log formatı:

```text
2026-05-30 15:30:04 | INFO | cross_dataset | seed=42 source=hust target=matr model=catboost
```

Her run için şu metadata kaydedilmelidir:

```json
{
  "git_commit": "abc123",
  "python_version": "3.11.8",
  "platform": "macOS",
  "created_at": "2026-05-30T15:30:04",
  "command": "python -m battery_life.cli experiment=cross_dataset",
  "dirty_git_tree": true
}
```

Bu, akademik reproducibility için çok değerlidir.

## 8. Test ve Kalite Kontrol Önerileri

Minimum test seti:

- Feature extraction testleri:
  - Q0 median çevrim 2-5 üzerinden mi hesaplanıyor?
  - EOL ilk `QD <= 0.85 * Q0` çevrimi mi?
  - Censored hücreler doğru işaretleniyor mu?
- Split testleri:
  - Train/cal/test hücreleri kesişmiyor mu?
  - Oranlar beklenen aralıkta mı?
  - Tüm seed dosyaları var mı?
- Metric testleri:
  - MAE, sMAPE, R2 hesapları küçük örneklerle doğrulanıyor mu?
- CP testleri:
  - Calibration residual quantile finite-sample rank doğru mu?
  - k çok küçükse infinite interval durumu doğru yakalanıyor mu?

Önerilen komut:

```bash
pytest tests/
```

CI için GitHub Actions eklenebilir:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

## 9. Veri ve Artifact Versiyonlama

Ham `.pkl` dosyaları büyük olduğu için Git'e alınmamalıdır. Ancak ara CSV'ler,
split JSON'ları ve final summary CSV'leri kontrollü biçimde versiyonlanabilir.

Bir sonraki aşama için seçenekler:

- **DVC**:
  - Büyük raw/intermediate artifact'ler için uygundur.
  - Remote storage olarak Google Drive, S3 veya local NAS kullanılabilir.
- **Git LFS**:
  - Büyük ama az değişen dosyalar için basittir.
  - Çok sayıda deney çıktısı için DVC kadar esnek değildir.
- **Manifest-only yaklaşımı**:
  - Ham veriler Git dışında tutulur.
  - `data_manifest.yaml` içinde dosya hash'leri ve beklenen konumlar saklanır.

Bu proje için önerilen pratik yol:

1. Raw `.pkl` dosyaları Git dışında kalsın.
2. `data/intermediate` içindeki küçük CSV'ler Git'te kalabilir.
3. Büyük figür/prediction artifact'leri için DVC veya release artifact mantığı
   kullanılabilir.

## 10. Kod Refactoring Yol Haritası

Önerilen sırayla yapılacaklar:

1. Ortak sabitleri merkezi hale getir:
   - `META_COLS`
   - seed listesi
   - dataset adları
   - default paths
2. Ortak data loader fonksiyonlarını `src/battery_life/data/` altına taşı.
3. Metric ve bootstrap hesaplarını `src/battery_life/evaluation/` altında
   standartlaştır.
4. Model fit/predict fonksiyonlarını registry yapısına al:

```python
MODEL_REGISTRY = {
    "catboost": fit_catboost,
    "random_forest": fit_random_forest,
    "elastic_net": fit_elastic_net,
}
```

5. Deney runner'larını config alan fonksiyonlara dönüştür:

```python
def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    ...
```

6. Her run sonunda config, metrics, predictions ve metadata kaydet.
7. MLflow logging'i opsiyonel backend olarak ekle.
8. CLI'ı tek giriş noktasına indir:

```bash
python -m battery_life.cli experiment=within_dataset
```

## 11. Akademik Teslim İçin En Faydalı Kısa Vadeli İyileştirmeler

Tez teslimi önceliği varsa en yüksek getirili adımlar:

1. README'nin profesyonel GitHub ana sayfası olarak temiz tutulması.
2. `docs/proje_baslangictan_bugune_ozet.md` dosyasının ana teknik tarihçe
   olarak korunması.
3. `outputs/advisor_tables/` altındaki tabloların rapora doğrudan eklenmesi.
4. Her yeni deney çıktısına `config.json` veya `config.yaml` kopyası yazılması.
5. En azından split ve metric fonksiyonları için küçük `pytest` testleri
   eklenmesi.

## 12. Önerilen Sonuç

Mevcut proje bilimsel olarak güçlü bir noktadadır: SOPv2 düzeltmeleri yapılmış,
within-dataset ve cross-dataset deneyleri tamamlanmış, concept shift argümanı
birden fazla analizle desteklenmiş ve target-domain calibration sonuçları
üretilmiştir.

Bir sonraki mühendislik olgunluk seviyesi, projeyi script koleksiyonundan
konfigürasyonla yönetilen, artifact'leri izlenen, metadata'sı kaydedilen ve
testlerle korunan bir MLOps pipeline'ına dönüştürmektir.
