
# query graph
data_root=/data/oss_bucket_0/jjr/LLMGGen
data_name=8days_dytag_small_text_en
dx_src_path=saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}

# # eval baseline
python -m LLMGGen.eval_utils.eval_src_edges \
    --data_root $data_root \
    --data_name $data_name \
    --time_window 86400 --bwr 1980 --use_feature bert \
    --pred_ratio 0.15 \
    --split test \
    --cm_order \
    --cut_off_baseline edge 
    # --node_msg \
    # --edge_msg \
    
# python -m LLMGGen.ggen_eval \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order \
#     --graph_result_path reports/baselines/${data_name}/test/query_ggen.csv \
#     --graph_report_path reports/baselines/${data_name}/test/query_report.csv
    # --node_msg \
    # --edge_msg \

python -m LLMGGen.ggen_eval \
    --data_root $data_root \
    --data_name $data_name \
    --time_window 86400 --bwr 1980 --use_feature bert \
    --pred_ratio 0.15 \
    --split test \
    --cm_order \
    --graph_result_path LLMGGen/reports/results/${data_name}/test/edge_ggen.csv \
    --edge_report_path LLMGGen/reports/baselines/${data_name}/test/edge_baseline.csv \
    --graph_report_path reports/baselines/${data_name}/test/graph_report.csv \
    # --edge_text_result_path LLMGGen/reports/results/${data_name}/test/edge_baseline.csv 