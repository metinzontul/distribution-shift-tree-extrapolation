| dataset | tier | gap_cat-ft_mean | gap_cat-ft_std | seeds | gap_cat-tabpfn_mean | catboost_nOOD | ft_nOOD | tabpfn_nOOD | best_model | worst_family_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooking-time | PRIMARY | -0.004 | 0.008 | 15 | 0.001 | 0.758 | 0.761 | 0.757 | ridge | mlp |
| delivery-eta | PRIMARY | 0.006 | 0.003 | 15 | 0.006 | 0.770 | 0.763 | 0.763 | ft_transformer | mlp |
| sberbank-housing | PRIMARY | 0.004 | 0.073 | 15 | 0.045 | 0.677 | 0.684 | 0.646 | tabpfn | mlp |
| weather | PRIMARY | -0.012 | 0.005 | 15 | 0.003 | 0.431 | 0.444 | 0.428 | tabpfn | mlp |
| air_quality | SECONDARY-proxy | 0.043 | 0.143 | 15 | 0.450 | 1.795 | 1.760 | 1.205 | tabpfn | mlp |
| bike_hour | SECONDARY-proxy | 0.163 | 0.062 | 15 |  | 0.920 | 0.759 |  | ft_transformer | catboost |
