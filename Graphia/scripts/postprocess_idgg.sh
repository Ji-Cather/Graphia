export data_root=Graphia/data
export save_root="Graphia/"
export dx_src_root=Graphia/saved_results_deg/${dx_model_name}/${data_name}
export HF_ENDPOINT=https://hf-mirror.com

for data_name in 8days_dytag_small_text_en weibo_daily weibo_tech; do

    export data_name
    
    # 根据 data_name 设置不同的路径和参数
    case "$data_name" in
        "8days_dytag_small_text_en")
            export dx_src_root="Graphia/saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            export seq_dx_src_root="Graphia/saved_results_deg/seq_deg/${data_name}"
            export tdgg_rl_query_model="grpo_8days_dytag_small_text_en_rl-all-domain_300epoch"
            export tdgg_rl_edge_model="grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-edge"

            tdgg_rl_seq_query_model="grpo_8days_dytag_small_text_en_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_8days_dytag_small_text_en_sotopia_edge_50epoch"

            export bwr=1980
            export time_window=86400
            ;;
        "weibo_daily")
            export dx_src_root="Graphia/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            export seq_dx_src_root="Graphia/saved_results_deg/seq_deg/${data_name}"
            export tdgg_rl_query_model="grpo_weibo_daily_query_all_dst_50epoch"
            export tdgg_rl_edge_model="grpo_weibo_daily_rl-all-domain_100epoch"

            tdgg_rl_seq_query_model="grpo_weibo_daily_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_weibo_daily_sotopia_edge_50epoch"
            export bwr=2048
            export time_window=86400
            ;;
        "weibo_tech")
            export dx_src_root="Graphia/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            export seq_dx_src_root="Graphia/saved_results_deg/seq_deg/${data_name}"
            export tdgg_rl_query_model="grpo_weibo_tech_query_all_dst_50epoch"
            export tdgg_rl_edge_model="grpo_weibo_tech_rl-all-domain_100epoch"

            tdgg_rl_seq_query_model="grpo_weibo_tech_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_weibo_tech_sotopia_edge_50epoch"
            export bwr=2048
            export time_window=86400
            ;;

        *)
            echo "Unknown dataset: $data_name"
            exit 1
            ;;
    esac


    # baseline eval     
    python -m Graphia.eval_utils.eval_src_edges \
        --data_root $data_root \
        --data_name $data_name \
        --time_window $time_window --bwr $bwr --use_feature bert \
        --pred_ratio 0.15 \
        --split test \
        --cm_order True\
        --graph_macro_report_path Graphia/reports/baselines/${data_name}/test/inference/graph_macro_matrix.csv \
        --graph_report_path Graphia/reports/baselines/${data_name}/test/inference/graph_matrix.csv \
        --graph_list_report_path Graphia/reports/baselines/${data_name}/test/inference/graph_list_matrix.csv 
       

    # query: idgg
    for llm in $tdgg_rl_query_model
    do
        export llm=${llm}
        export llm_save_root=Graphia/results/${llm}
        export report_save_root=Graphia/reports/${llm}
        for mode in process_inf eval_inf
        do
            export mode=${mode}
            bash Graphia/scripts/postprocess_ggen_loop.sh
        done
    done

    # gen edge prompt: idgg
    for llm in $tdgg_rl_edge_model
    do
        export query_llm=$tdgg_rl_query_model
        export query_llm_save_root=Graphia/results/${query_llm}
        export llm=$llm
        export llm_save_root=Graphia/results/${llm}
        export report_save_root=Graphia/reports/${llm}
        export dx_src_path=$dx_src_root/test_degree.pt 
        for mode in pre_process_inf pre_process_inf_broadcast 
        do
            export mode=${mode}
            export model_config_name=${llm}
            bash Graphia/scripts/postprocess_edge_loop.sh
        done
    done
    
    # # edge: idgg
    for llm in $tdgg_rl_edge_model
    do
        export llm=${llm}
        export llm_save_root=Graphia/results/${llm}
        export report_save_root=Graphia/reports/${llm}
        for mode in process_inf process_inf_broadcast 
        do
            export mode=${mode}
            bash Graphia/scripts/postprocess_edge_loop.sh 
        done
    done


    #### seqential data

    # query: idgg
    for llm in $tdgg_rl_seq_query_model qwen3_sft
    do
        export llm=${llm}
        export llm_save_root=Graphia/results/${llm}
        export report_save_root=Graphia/reports/${llm}
        for mode in process_inf eval_inf
        do
            export mode=${mode}
            bash Graphia/scripts/postprocess_seq_ggen_loop.sh
        done
    done

done