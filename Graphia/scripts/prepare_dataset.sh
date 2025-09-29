python -m Graphia.utils.bwr_ctdg --pred_ratio 0.15 --root Graphia/data/8days_dytag_small_text_en --time_window 86400 --bwr 1980 --use_feature bert --cm_order True --force_reload

python -m Graphia.utils.bwr_ctdg --pred_ratio 0.15 --root Graphia/data/weibo_tech --time_window 86400 --bwr 2048 --use_feature bert --cm_order True --force_reload

python -m Graphia.utils.bwr_ctdg --pred_ratio 0.15 --root Graphia/data/weibo_daily --time_window 86400 --bwr 2048 --use_feature bert --cm_order True --force_reload



# Root: data/8days_dytag_small_text
# Input length: 18
# Prediction length: 4

# Root: data/weibo_tech
# Input length: 5
# Prediction length: 1

# Root: data/weibo_daily
# Input length: 19
# Prediction length: 4