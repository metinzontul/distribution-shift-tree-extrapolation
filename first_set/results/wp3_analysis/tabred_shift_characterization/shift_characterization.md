# TabReD natural temporal shift -- characterization (post-hoc / exploratory)

> Numeric (x_num) features only; authored `default` temporal split (fit=train period, OOD=test period). OOR = out-of-range fraction of OOD-test cells vs train support. `_mm` = train [min,max]; `_p1p99` = train [p1,p99]. `Yfeat` = the single most Y-correlated (max |Spearman|) feature = the mechanism's designed extrapolation axis; the construct arm forces this feature to 100% out-of-range at sev50. PSI/SMD = per-feature covariate-shift magnitude (median over features). Y_* = target/prior shift. ridge_ood/tr = crude linear concept-drift probe (OOD RMSE / train RMSE).

| dataset | n_test | F_num | OOR_mm_Yfeat | OOR_mm_maxfeat | OOR_mm_mean | OOR_p1p99_mean | PSI_med | SMD_med | Y_SMD | Y_PSI | Y_stdratio | ridge_ood/tr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sberbank-housing | 4647 | 365 | 0.0 | 1.0 | 0.0917 | 0.1021 | 0.049 | 0.093 | 0.227 | 0.134 | 0.967 | 3.34 |
| cooking-time | 41648 | 186 | 0.0 | 0.0 | 0.0 | 0.013 | 0.006 | 0.051 | 0.1 | 0.01 | 1.015 | 1.05 |
| weather | 40840 | 96 | 0.0 | 0.001 | 0.0 | 0.031 | 0.067 | 0.14 | 0.135 | 0.017 | 0.991 | 0.98 |
| delivery-eta | 36927 | 217 | 0.0 | 0.001 | 0.0 | 0.0179 | 0.005 | 0.023 | 0.093 | 0.009 | 1.003 | 1.03 |
| CONSTRUCT sev50 (median, n=27) | - | - | 1.0 | 1.0 | 0.1561 | 0.2256 | 1.099 | - | - | - | - | - |

## Covariate-shift magnitude and target/prior shift (detail)

| dataset | PSI_mean | PSI_max | frac_PSI>0.25 | SMD_max | frac_SMD>0.5 | Y_mean_tr | Y_mean_te | Y_q50_shift_std | Y_q90_shift_std | Y_KS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sberbank-housing | 1.507 | 12.83 | 0.264 | 14.85 | 0.162 | 11.752 | 11.842 | 0.14 | 0.266 | 0.107 |
| cooking-time | 0.015 | 0.17 | 0.0 | 0.35 | 0.0 | 2.535 | 2.601 | 0.111 | 0.112 | 0.042 |
| weather | 0.268 | 12.79 | 0.168 | 0.76 | 0.094 | 26.749 | 27.274 | 0.129 | 0.257 | 0.095 |
| delivery-eta | 0.883 | 5.7 | 0.242 | 0.95 | 0.037 | 1.746 | 1.814 | 0.098 | 0.089 | 0.039 |

## Construct extrapolation-arm sev50 out-of-range, per dataset (calibration; n=27)

| dataset | F | abs_spearman | split_OOR_mm | maxfeat_OOR_mm | mean_OOR_mm | PSI_med |
| --- | --- | --- | --- | --- | --- | --- |
| naval_propulsion_plant | 11 | 0.1 | 1.0 | 1.0 | 1.0 | 12.434 |
| abalone | 7 | 0.69 | 1.0 | 1.0 | 0.7931 | 12.399 |
| diamonds | 6 | 0.96 | 1.0 | 1.0 | 0.5747 | 11.76 |
| bike_hour | 4 | 0.42 | 1.0 | 1.0 | 0.4993 | 5.257 |
| airfoil_self_noise | 3 | 0.34 | 1.0 | 1.0 | 0.3656 | 1.237 |
| QSAR_fish_toxicity | 4 | 0.67 | 1.0 | 1.0 | 0.3287 | 2.966 |
| Moneyball | 8 | 0.91 | 1.0 | 1.0 | 0.2618 | 3.695 |
| kings_county | 11 | 0.64 | 1.0 | 1.0 | 0.2339 | 1.467 |
| space_ga | 6 | 0.56 | 1.0 | 1.0 | 0.1937 | 0.099 |
| fifa | 27 | 0.77 | 1.0 | 1.0 | 0.183 | 1.135 |
| white_wine | 11 | 0.44 | 1.0 | 1.0 | 0.1801 | 0.78 |
| concrete_compressive_strength | 7 | 0.48 | 1.0 | 1.0 | 0.1726 | 2.929 |
| cpu_activity | 21 | 0.81 | 1.0 | 1.0 | 0.1629 | 2.053 |
| video_transcoding | 9 | 0.31 | 1.0 | 1.0 | 0.1561 | 2.431 |
| puma8NH | 8 | 0.5 | 1.0 | 1.0 | 0.1255 | 0.008 |
| kin8nm | 8 | 0.52 | 1.0 | 1.0 | 0.1253 | 0.012 |
| california_housing | 8 | 0.68 | 1.0 | 1.0 | 0.1251 | 0.376 |
| red_wine | 11 | 0.48 | 1.0 | 1.0 | 0.1235 | 0.487 |
| physiochemical_protein | 9 | 0.39 | 1.0 | 1.0 | 0.1219 | 0.207 |
| miami_housing | 12 | 0.7 | 1.0 | 1.0 | 0.0887 | 0.617 |
| grid_stability | 12 | 0.31 | 1.0 | 1.0 | 0.0843 | 0.006 |
| superconductivity | 79 | 0.75 | 1.0 | 1.0 | 0.068 | 4.991 |
| sarcos | 21 | 0.57 | 1.0 | 1.0 | 0.054 | 0.489 |
| Ailerons | 25 | 0.73 | 1.0 | 1.0 | 0.0401 | 1.099 |
| geographical_origin_of_music | 116 | 0.37 | 1.0 | 1.0 | 0.0353 | 0.277 |
| pumadyn32nh | 32 | 0.37 | 1.0 | 1.0 | 0.0319 | 0.008 |
| wave_energy | 48 | 0.34 | 1.0 | 1.0 | 0.0287 | 0.09 |

