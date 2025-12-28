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
    edge_text_result_path="${llm_save_root}/${data_name}/test/teacher_forcing/edge_text_eval_prompt_${eval_llm}.csv"
    edge_report_path="${report_save_root}/${data_name}/test/teacher_forcing/edge_matrix_${eval_llm}.csv"
}

# === Main Logic ===
case "$mode" in
    "eval_tf")
        echo "Mode: Evaluate tdgg edge"
        for eval_llm in llama31-70B llama33-70B Qwen3-32B
            do
            set_tf_paths
            python -m Graphia.ggen_eval \
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
        done
        ;;
    "help"|"")
        echo "Usage: mode={eval_tf}"
        echo "Current mode: '$mode'"
        exit 1
        ;;
    *)
        echo "Unknown mode: $mode"
        echo "Available: eval_tf"
        exit 1
        ;;
esac