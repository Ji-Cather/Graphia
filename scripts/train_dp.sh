export data_root=/data/oss_bucket_0/jjr/LLMGGen
data_root=/data/jiarui_ji/llmggen/LLMGGen/data

# python -m LLMGGen.train_node_regression_v2 \
#   --data_name 8days_dytag_small_text_en \
#   --data_root ${data_root} \
#   --model_name InformerDecoder \
#   --num_runs 1 \
#   --gpu 1 \
#   --num_epochs 50 \
#   --batch_size 1 \
#   --test_interval_epochs 5 \
#   --quantile_mapping \
#   --pred_ratio 0.15 \
#   --time_window 86400 \
#   --bwr 1980 \
#   --use_feature bert \
#   --cm_order \
#   --rescale >> log_dp.txt 2>&1 &

python -m LLMGGen.train_node_regression_v2 \
  --data_name weibo_daily \
  --data_root ${data_root} \
  --model_name InformerDecoder \
  --num_runs 1 \
  --gpu 1 \
  --num_epochs 50 \
  --batch_size 1 \
  --test_interval_epochs 5 \
  --quantile_mapping \
  --pred_ratio 0.15 \
  --time_window 86400 \
  --bwr 14702 \
  --max_len 14702 \
  --use_feature bert \
  --cm_order \
  --rescale