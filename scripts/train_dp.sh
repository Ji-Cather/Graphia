

python -m llmggen.train_node_regression_v2 \
  --data_name 8days_dytag_small_text_en \
  --model_name InformerDecoder \
  --num_runs 1 \
  --gpu 1 \
  --num_epochs 50 \
  --batch_size 1 \
  --test_interval_epochs 5 \
  --quantile_mapping \
  --pred_ratio 0.15 \
  --time_window 86400 \
  --bwr 1980 \
  --use_feature bert \
  --cm_order \
  --rescale