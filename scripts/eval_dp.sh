#!/bin/bash
# eval src degree prediction model
for data_name in 8days_dytag_large 8days_dytag_small; do
    
    # python eval/eval_src_edges.py \
    # --data_name $data_name \
    # --bwr 2048 \
    # --use_feature no \
    # --val_ratio 0.125 \
    # --test_ratio 0.125

    python eval_utils/eval_src_degree.py \
    --data_name $data_name \
    --time_window 86400 --bwr 1980 --use_feature bert
    --pred_ratio 0.15 \
done
