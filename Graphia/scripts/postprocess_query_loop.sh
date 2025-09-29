


# # process with reward selection
# # query_file_name=query_examples_8.csv

# # query_save_path=${llm_save_root}/${data_name}/test/teacher_forcing/${query_file_name}
# # query_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/query_ggen_select.csv
# # graph_report_path=${report_save_root}/${data_name}/test/teacher_forcing/graph_matrix_select.csv

# # if [ "$mode" == "process" ]; then
# #     echo "Mode is 'process'. Performing processing tasks..."
# #     python -m Graphia.src_edge_offline \
# #         --data_root $data_root \
# #         --data_name $data_name \
# #         --time_window 86400 --bwr 1980 --use_feature bert \
# #         --pred_ratio 0.15 \
# #         --split test \
# #         --cm_order True\
# #         --query_save_path ${query_save_path} \
# #         --query_result_path ${query_result_path} \
# #         --process_query_result \
# #         --reward_sel gnn \
# #         --gnn_model_name GraphMixer \
# #         --gnn_save_model_path Graphia/lp_models/saved_models/GraphMixer/8days_dytag_small_text_en/GraphMixer_seed0bert/GraphMixer_seed0bert.pkl

# #     # python -m Graphia.src_edge_offline \
# #     #     --data_root $data_root \
# #     #     --data_name $data_name \
# #     #     --time_window 86400 --bwr 1980 --use_feature bert \
# #     #     --pred_ratio 0.15 \
# #     #     --split test \
# #     #     --cm_order True\
# #     #     --edge_save_path ${edge_save_path} \
# #     #     --edge_result_path ${edge_result_path} \
# #     #     --process_edge_result \
# #     #     --reward_sel gnn \
# #     #     --gnn_model_name GraphMixer \
# #     #     --gnn_save_model_path Graphia/ec_models/saved_models/GraphMixer/8days_dytag_small_text_en/edge_classification_GraphMixer_seed0bert/edge_classification_GraphMixer_seed0bert.pkl

        
       
# # else
# #     echo "Mode is 'eval'. Performing eval tasks..."
   
# #     python -m Graphia.ggen_eval \
# #         --data_root $data_root \
# #         --data_name $data_name \
# #         --time_window 86400 --bwr 1980 --use_feature bert \
# #         --pred_ratio 0.15 \
# #         --split test \
# #         --cm_order True \
# #         --graph_result_path ${query_result_path} \
# #         --edge_report_path ${edge_report_path} \
# #         --graph_report_path ${graph_report_path} \
# #         --edge_text_result_path ${edge_text_result_path}
# #         # --node_msg \
# #         # --edge_msg \
# # fi


# # process without reward selection
query_file_name=query_examples.csv
query_save_path=${llm_save_root}/${data_name}/test/teacher_forcing/${query_file_name}
query_result_path=${llm_save_root}/${data_name}/test/teacher_forcing/query_ggen.csv
graph_report_path=${report_save_root}/${data_name}/test/teacher_forcing/graph_matrix.csv


if [ "$mode" == "process_tf" ]; then
    echo "Mode is 'process'. Performing processing tasks..."
    python -m Graphia.src_edge_offline \
        --data_root $data_root \
        --data_name $data_name \
        --time_window 86400 --bwr 1980 --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True\
        --query_save_path ${query_save_path} \
        --query_result_path ${query_result_path} \
        --process_query_result 
elif [ "$mode" == "eval_tf" ];
    echo "Mode is 'eval'. Performing eval tasks..."
    python -m Graphia.ggen_eval \
        --data_root $data_root \
        --data_name $data_name \
        --time_window 86400 --bwr 1980 --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True \
        --graph_result_path ${query_result_path} \
        --graph_report_path ${graph_report_path} 
        # --node_msg \
        # --edge_msg \
fi


# eval inference
query_file_name=query_examples.csv
query_save_path=${llm_save_root}/${data_name}/test/inference/${query_file_name}
query_result_path=${llm_save_root}/${data_name}/test/inference/query_ggen.csv
graph_report_path=${report_save_root}/${data_name}/test/inference/graph_matrix.csv

if [ "$mode" == "process_inf" ]; then
    echo "Mode is 'process'. Performing processing tasks..."
    python -m Graphia.src_edge_offline \
        --data_root $data_root \
        --data_name $data_name \
        --time_window 86400 --bwr 1980 --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True\
        --dx_src_path ${dx_src_root}/test_degree.pt \
        --query_save_path ${query_save_path} \
        --query_result_path ${query_result_path} \
        --process_query_result 
elif [ "$mode" == "eval_inf" ]; then
    python -m Graphia.ggen_eval \
        --data_root $data_root \
        --data_name $data_name \
        --time_window 86400 --bwr 1980 --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True\
        --graph_result_path ${query_result_path} \
        --graph_report_path ${graph_report_path} \
        # --node_msg \
        # --edge_msg \
fi