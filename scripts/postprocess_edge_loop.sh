# === Configuration Variables ===
data_root=${data_root:-"/path/to/data"}
llm_save_root=${llm_save_root:-"/path/to/save"}
report_save_root=${report_save_root:-"/path/to/report"}
dx_src_root=${dx_src_root:-"/path/to/dx_src"}
data_name=${data_name:-"your_dataset"}

# === Functions: Set Teacher-Forcing Paths ===
set_tf_paths() {
    edge_save_path="${llm_save_root}/${data_name}/test/teacher_forcing/edge_text_prompt.csv"
    edge_result_path="${llm_save_root}/${data_name}/test/teacher_forcing/edge_ggen.csv"
    edge_text_result_path="${llm_save_root}/${data_name}/test/teacher_forcing/edge_text_eval_prompt.csv"
    edge_report_path="${report_save_root}/${data_name}/test/teacher_forcing/edge_matrix.csv"
}

# === Functions: Set Inference Paths ===
set_inf_paths() {
    query_result_path="${query_llm_save_root}/${data_name}/test/inference/query_ggen.csv"
    edge_save_path="${llm_save_root}/${data_name}/test/inference/edge_text_examples.csv"
    edge_result_path="${llm_save_root}/${data_name}/test/inference/edge_ggen.csv"
    edge_text_result_path="${llm_save_root}/${data_name}/test/inference/edge_text_eval_prompt.csv"
    graph_report_path="${report_save_root}/${data_name}/test/inference/graph_matrix_msg.csv"
}

set_inf_paths_broadcast() {
    query_result_path="${query_llm_save_root}/${data_name}/test/inference/query_ggen.csv"
    edge_save_path="${llm_save_root}/${data_name}/test/inference/edge_text_examples_broadcast.csv"
    edge_result_path="${llm_save_root}/${data_name}/test/inference/edge_ggen_broadcast.csv"
    edge_text_result_path="${llm_save_root}/${data_name}/test/inference/edge_text_eval_broadcast_prompt.csv"
    graph_report_path="${report_save_root}/${data_name}/test/inference/graph_matrix_broadcast_msg.csv"
}

# === Main Logic ===
case "$mode" in
    "process_tf")
        echo "Mode: Process tdgg edge"
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
            --edge_save_path "$edge_save_path" \
            --edge_result_path "$edge_result_path" \
            --process_edge_result
        ;;
    "eval_tf")
        echo "Mode: Evaluate tdgg edge"
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
            --edge_result_path "$edge_result_path" \
            --edge_report_path "$edge_report_path" \
            --edge_text_result_path "$edge_text_result_path"
        ;;
    "pre_process_inf")
        echo "Mode: pre-Process idgg edge"
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
            --save_root "$save_root" \
            --dx_src_path "${dx_src_root}/test_degree.pt" \
            --query_result_path "$query_result_path" \
            --model_config_name "$model_config_name" \
            --infer_edge
        ;;
    "pre_process_inf_broadcast")
        echo "Mode: pre-Process idgg edge"
        set_inf_paths_broadcast
        python -m LLMGGen.src_edge_offline \
            --data_root "$data_root" \
            --data_name "$data_name" \
            --time_window $time_window \
            --bwr $bwr \
            --use_feature bert \
            --pred_ratio 0.15 \
            --split test \
            --cm_order True \
            --save_root "$save_root" \
            --dx_src_path "${dx_src_root}/test_degree.pt" \
            --query_result_path "$query_result_path" \
            --model_config_name "$model_config_name" \
            --infer_edge \
            --broadcast
        ;;
    "process_inf_broadcast")
        echo "Mode: Process idgg edge"
        set_inf_paths_broadcast
        python -m LLMGGen.src_edge_offline \
            --data_root "$data_root" \
            --data_name "$data_name" \
            --time_window $time_window \
            --bwr $bwr \
            --use_feature bert \
            --pred_ratio 0.15 \
            --split test \
            --cm_order True \
            --dx_src_path "${dx_src_root}/test_degree.pt" \
            --edge_save_path "$edge_save_path" \
            --edge_result_path "$edge_result_path" \
            --process_edge_result
        ;;
    "process_inf")
        echo "Mode: Process idgg edge"
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
            --dx_src_path "${dx_src_root}/test_degree.pt" \
            --edge_save_path "$edge_save_path" \
            --edge_result_path "$edge_result_path" \
            --process_edge_result
        ;;
    "eval_inf")
        echo "Mode: Evaluate idgg edge"
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
            --node_msg \
            --edge_msg \
            --graph_result_path "$edge_result_path" \
            --graph_report_path "$graph_report_path" 
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