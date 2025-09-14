
# process without reward selection
edge_save_path=${llm_save_root}/${data_name}/test/teacher_forcing/edge_text_prompt.csv
edge_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/edge_ggen.csv
edge_text_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/edge_text_eval_prompt.csv
edge_report_path=${report_save_root}/${data_name}/test/teacher_forcing/edge_matrix.csv


if [ "$mode" == "process_tf" ]; then
    echo "Mode is 'process tdgg edge'. Performing processing tasks..."
    python -m LLMGGen.src_edge_offline \
        --data_root $data_root \
        --data_name $data_name \
        --time_window 86400 --bwr 1980 --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True\
        --edge_save_path ${edge_save_path} \
        --edge_result_path ${edge_result_path} \
        --process_edge_result \
        
else
    echo "Mode is 'eval tdgg edge'. Performing eval tasks..."
    python -m LLMGGen.ggen_eval \
        --data_root $data_root \
        --data_name $data_name \
        --time_window 86400 --bwr 1980 --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True \
        --edge_result_path ${edge_result_path} \
        --edge_report_path ${edge_report_path} \
        # --edge_text_result_path ${edge_text_result_path}
        # --node_msg \
        # --edge_msg \
fi
