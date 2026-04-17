# SOP Results Summary

## stored_cycle_life

- Within-dataset results:
  Best MAE: b1_to_b1 | xgboost | 50 cycles | MAE 162.600 | SMAPE 19.192

| Direction | Model | Window | MAE | SMAPE | R2 | TrainRows | TestRows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b1_to_b1 | elastic_net | 50 | 362.224 | 26.542 | -39.418 | 28 | 6 |
| b1_to_b1 | elastic_net | 100 | 234.763 | 28.944 | -4.376 | 28 | 6 |
| b1_to_b1 | xgboost | 50 | 162.600 | 19.192 | -0.706 | 28 | 6 |
| b1_to_b1 | xgboost | 100 | 190.123 | 20.387 | -2.202 | 28 | 6 |

## eol_80pct_q0_observed_only

- Censor rate: 0.878
- Within-dataset results:
  Best MAE: b1_to_b1 | xgboost | 100 cycles | MAE 204.665 | SMAPE 11.266

| Direction | Model | Window | MAE | SMAPE | R2 | TrainRows | TestRows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b1_to_b1 | elastic_net | 50 | 313.698 | 16.555 | NA | 3 | 1 |
| b1_to_b1 | elastic_net | 100 | 644.600 | 47.902 | NA | 3 | 1 |
| b1_to_b1 | xgboost | 50 | 205.790 | 11.306 | NA | 3 | 1 |
| b1_to_b1 | xgboost | 100 | 204.665 | 11.266 | NA | 3 | 1 |

## eol_88ah_observed_only

- Censor rate: 0.878
- Within-dataset results:
  Best MAE: b1_to_b1 | xgboost | 100 cycles | MAE 198.066 | SMAPE 10.965

| Direction | Model | Window | MAE | SMAPE | R2 | TrainRows | TestRows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b1_to_b1 | elastic_net | 50 | 354.959 | 18.031 | NA | 3 | 1 |
| b1_to_b1 | elastic_net | 100 | 632.661 | 46.547 | NA | 3 | 1 |
| b1_to_b1 | xgboost | 50 | 199.018 | 10.996 | NA | 3 | 1 |
| b1_to_b1 | xgboost | 100 | 198.066 | 10.965 | NA | 3 | 1 |

## Short Report Text

SOP12 transition feature set ile `stored_cycle_life` hedefinde en iyi within-dataset sonuc `xgboost` ailesinde elde edildi; en dusuk MAE 162.600 ve 50 cycle penceresinde goruldu.
%80 Q0 observed-only deneyinde sansur oranina ragmen kullanilabilir alt kumeden sonuc alinabildi; en iyi MAE 204.665 ve SMAPE 11.266 oldu.
0.88Ah observed-only deneyinde de benzer sekilde kucuk test ornegiyle sonuc uretildi; en iyi MAE 198.066 seviyesinde kaldi.
Observed-only senaryolarda her seed icin testte tek hucre bulundugu icin R2 degeri anlamsizdir; bu sonuclar pilot/yon gosterici olarak yorumlanmalidir.