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
#   --rescale

# # python -m LLMGGen.test_src_degree \
# #   --data_root ${data_root} \
# #   --data_name 8days_dytag_small_text_en \
# #   --time_window 86400 \
# #   --bwr 1980 \
# #   --use_feature bert \
# #   --pred_ratio 0.15 \
# #   --cm_order \
# #   --quantile_mapping \
# #   --rescale


python -m LLMGGen.train_node_regression_v2 \
  --data_name propagate_large_cn \
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
  --bwr 2048 \
  --use_feature bert \
  --cm_order \
  --rescale

# python -m LLMGGen.test_src_degree \
#   --data_root ${data_root} \
#   --data_name propagate_large_cn \
#   --time_window 86400 \
#   --bwr 2048 \
#   --use_feature bert \
#   --pred_ratio 0.15 \
#   --cm_order \
#   --quantile_mapping \
#   --rescale

# python -m LLMGGen.train_node_regression_v2 \
#   --data_name weibo_daily \
#   --data_root ${data_root} \
#   --model_name InformerDecoder \
#   --num_runs 1 \
#   --gpu 0 \
#   --num_epochs 50 \
#   --batch_size 1 \
#   --test_interval_epochs 5 \
#   --pred_ratio 0.15 \
#   --time_window 86400 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --quantile_mapping \
#   --rescale

# python -m LLMGGen.test_src_degree \
#   --data_root ${data_root} \
#   --data_name weibo_daily \
#   --pred_ratio 0.15 \
#   --time_window 86400 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order 


# python -m LLMGGen.train_node_regression_v2 \
#   --data_name weibo_tech \
#   --data_root ${data_root} \
#   --model_name InformerDecoder \
#   --num_runs 1 \
#   --gpu 0 \
#   --num_epochs 50 \
#   --batch_size 1 \
#   --test_interval_epochs 5 \
#   --pred_ratio 0.15 \
#   --time_window 86400 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --quantile_mapping \
#   --rescale

# python -m LLMGGen.test_src_degree \
#   --data_root ${data_root} \
#   --data_name weibo_tech \
#   --pred_ratio 0.15 \
#   --time_window 86400 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --quantile_mapping \
#   --rescale

# python -m LLMGGen.train_node_regression_v2 \
#   --data_name imdb \
#   --data_root ${data_root} \
#   --model_name InformerDecoder \
#   --num_runs 1 \
#   --gpu 0 \
#   --num_epochs 50 \
#   --batch_size 1 \
#   --test_interval_epochs 5 \
#   --pred_ratio 0.15 \
#   --time_window 31536000 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --quantile_mapping \
#   --rescale


# python -m LLMGGen.test_src_degree \
#   --data_root ${data_root} \
#   --data_name imdb \
#   --pred_ratio 0.15 \
#   --time_window 31536000 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --rescale


# python -m LLMGGen.train_node_regression_v2 \
#   --data_name cora \
#   --data_root ${data_root} \
#   --model_name InformerDecoder \
#   --num_runs 1 \
#   --gpu 0 \
#   --num_epochs 50 \
#   --batch_size 1 \
#   --test_interval_epochs 5 \
#   --pred_ratio 0.15 \
#   --time_window 31536000 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --quantile_mapping \
#   --rescale


# python -m LLMGGen.test_src_degree \
#   --data_root ${data_root} \
#   --data_name cora \
#   --pred_ratio 0.15 \
#   --time_window 31536000 \
#   --bwr 2048 \
#   --use_feature bert \
#   --cm_order \
#   --rescale