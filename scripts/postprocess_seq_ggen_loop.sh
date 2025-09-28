#!/bin/bash

# === 配置变量 ===
data_root=${data_root:-"/path/to/data"}
llm_save_root=${llm_save_root:-"/path/to/save"}
report_save_root=${report_save_root:-"/path/to/report"}
seq_dx_src_root=${seq_dx_src_root:-"/path/to/dx_src"}
data_name=${data_name:-"your_dataset"}

# === 函数：设置 teacher-forcing 路径 ===
set_tf_paths() {
    query_file_name="query_examples.csv"
    query_save_path="${llm_save_root}/${data_name}/test/teacher_forcing/${query_file_name}"
    query_result_path="${llm_save_root}/${data_name}/test/teacher_forcing/query_ggen.csv"
    graph_report_path="${report_save_root}/${data_name}/test/teacher_forcing/graph_matrix.csv"
}

# === 函数：设置 seq_inference 路径 ===
set_inf_paths() {
    query_file_name="query_examples.csv"
    query_save_path="${llm_save_root}/${data_name}/test/seq_inference/${query_file_name}"
    query_result_path="${llm_save_root}/${data_name}/test/seq_inference/query_ggen.csv"
    graph_report_path="${report_save_root}/${data_name}/test/seq_inference/graph_matrix.csv"
    graph_list_report_path="${report_save_root}/${data_name}/test/seq_inference/graph_list_matrix.csv"
    graph_macro_report_path="${report_save_root}/${data_name}/test/inference/graph_macro_matrix.csv"
}

# === 主逻辑 ===
case "$mode" in
    "process_tf")
        echo "Mode: Process tdgg query"
        set_tf_paths
        python -m LLMGGen.src_edge_offline \
            --data_root "$data_root" \
            --data_name "$data_name" \
            --time_window $time_window \
            --bwr $bwr \
            --use_feature bert \
            --pred_ratio 0.15 \
            --split test \
            --cm_order True \
            --query_save_path "$query_save_path" \
            --query_result_path "$query_result_path" \
            --process_query_result \
            # --reward_sel gnn \
            # --gnn_save_model_path "LLMGGen/lp_models/saved_models/GraphMixer/8days_dytag_small_text_en/GraphMixer_seed0bert/GraphMixer_seed0bert.pkl" \
            # --gnn_model_name GraphMixer
        ;;
    "eval_tf")
        echo "Mode: Evaluate tdgg query"
        set_tf_paths
        python -m LLMGGen.ggen_eval \
            --data_root "$data_root" \
            --data_name "$data_name" \
            --time_window $time_window \
            --bwr $bwr \
            --use_feature bert \
            --pred_ratio 0.15 \
            --split test \
            --cm_order True \
            --graph_result_path "$query_result_path" \
            --graph_report_path "$graph_report_path"
            # --node_msg \
            # --edge_msg
        ;;
    "process_inf")
        echo "Mode: Process idgg query"
        set_inf_paths
        python -m LLMGGen.src_edge_offline \
            --data_root "$data_root" \
            --data_name "$data_name" \
            --time_window $time_window \
            --bwr $bwr \
            --use_feature bert \
            --pred_ratio 0.15 \
            --split test \
            --cm_order True \
            --dx_src_path "${seq_dx_src_root}/test_degree.pt" \
            --query_save_path "$query_save_path" \
            --query_result_path "$query_result_path" \
            --process_query_result
        ;;
    "eval_inf")
        echo "Mode: Evaluate idgg query"
        set_inf_paths
        python -m LLMGGen.ggen_eval \
            --data_root "$data_root" \
            --data_name "$data_name" \
            --time_window $time_window \
            --bwr $bwr \
            --use_feature bert \
            --pred_ratio 0.15 \
            --split test \
            --cm_order True \
            --graph_result_path "$query_result_path" \
            --graph_macro_report_path "$graph_macro_report_path" \
            # --graph_report_path "$graph_report_path" \
            # --graph_list_report_path "$graph_list_report_path" \
        ;;
    "help"|"")
        echo "Usage: mode={process_tf|eval_tf|process_inf|eval_inf}"
        echo "Current mode: '$mode'"
        exit 1
        ;;
    *)
        echo "Unknown mode: $mode"
        echo "Available: process_tf, eval_tf, process_inf, eval_inf"
        exit 1
        ;;
esac
