
export data_root=/data/oss_bucket_0/jjr/LLMGGen

export llm=qwen3
# export llm=qwen3_sft
export data_name=8days_dytag_small_text_en
export dx_src_root=saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}

export llm_save_root=results/${llm}
export report_save_root=reports/${llm}

export data_root=LLMGGen/data
export llm_save_root=LLMGGen/results/${llm}
export report_save_root=LLMGGen/reports/${llm}

# python  -m LLMGGen.convert_prompt_csv --phase "j2c" \
#         --output_dir "/home/jijiarui.jjr/ROLL/LLMGGen/results" \
#         --root_dir "/data/oss_bucket_0/jjr/LLMGGen/results"

# # eval baseline
# python -m LLMGGen.eval_utils.eval_src_edges \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --cut_off_baseline edge 
#     # --node_msg \
#     # --edge_msg \


query_save_path=${llm_save_root}/${data_name}/test/teacher_forcing/query_examples.csv
query_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/query_ggen.csv
graph_report_path=${report_save_root}/${data_name}/test/teacher_forcing/graph_matrix.csv

edge_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/edge_ggen.csv
edge_eval_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/edge_text_eval_result.csv
edge_report_path=${report_save_root}/${data_name}/test/teacher_forcing/edge_matrix.csv


# python -m LLMGGen.src_edge_offline \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --query_save_path ${query_save_path} \
#     --query_result_path ${query_result_path} \
#     --process_query_result

# python -m LLMGGen.ggen_eval \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --graph_result_path ${query_result_path} \
#     --graph_report_path ${graph_report_path}



python -m LLMGGen.ggen_eval \
    --data_root $data_root \
    --data_name $data_name \
    --time_window 86400 --bwr 1980 --use_feature bert \
    --pred_ratio 0.15 \
    --split test \
    --cm_order True \
    --graph_result_path ${edge_result_path} \
    --edge_report_path ${edge_report_path} \
    --graph_report_path ${graph_report_path} \
#     --edge_text_eval_path ${edge_text_result_path}



# query_save_path=${llm_save_root}/${data_name}/test/inference/query_examples.csv
# query_result_path=${llm_save_root}/${data_name}/test/inference/query_ggen.csv
# graph_report_path=${report_save_root}/${data_name}/test/inference/graph_matrix.csv

# python -m LLMGGen.src_edge_offline \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --dx_src_path ${dx_src_root}/test_degree.pt \
#     --query_save_path ${query_save_path} \
#     --query_result_path ${query_result_path} \
#     --process_query_result


# python -m LLMGGen.ggen_eval \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --graph_result_path ${query_result_path} \
#     --graph_report_path ${graph_report_path}

