export data_root=Graphia/data
export HF_ENDPOINT=https://hf-mirror.com



# Loop through datasets
for data_name in 8days_dytag_small_text_en weibo_daily weibo_tech ; do
    export data_name
    
    # Set paths and parameters based on data_name
    case "$data_name" in
        "8days_dytag_small_text_en")
            export dx_src_root="Graphia/saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_8days_dytag_small_text_en_rl-all-domain_300epoch"
            tdgg_rl_edge_model="grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-edge"

            tdgg_rl_seq_query_model="grpo_8days_dytag_small_text_en_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_8days_dytag_small_text_en_sotopia_edge_50epoch"
            export bwr=1980
            export time_window=86400
            ;;
        "weibo_daily")
            export dx_src_root="Graphia/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_weibo_daily_query_all_dst_50epoch"
            tdgg_rl_edge_model="grpo_weibo_daily_rl-all-domain_100epoch"

            tdgg_rl_seq_query_model="grpo_weibo_daily_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_weibo_daily_sotopia_edge_50epoch"
            export bwr=2048
            export time_window=86400
            ;;
        "weibo_tech")
            export dx_src_root="Graphia/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_weibo_tech_query_all_dst_50epoch"
            tdgg_rl_edge_model="grpo_weibo_tech_rl-all-domain_100epoch"

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

    # query: tdgg
    for llm in $tdgg_rl_query_model $tdgg_rl_seq_query_model qwen3 qwen3_sft Meta-Llama-3.1-70B-Instruct Qwen3-32B DeepSeek-R1-Distill-Qwen-32B
    do
        export llm=${llm}
        export llm_save_root=Graphia/results/${llm}
        export report_save_root=Graphia/reports/${llm}
        for mode in process_tf
        do
            export mode=${mode}
            bash Graphia/scripts/postprocess_ggen_loop.sh
        done
    done

    # edge: tdgg
    for llm in $tdgg_rl_edge_model $tdgg_rl_seq_edge_model qwen3 qwen3_sft Meta-Llama-3.1-70B-Instruct Qwen3-32B DeepSeek-R1-Distill-Qwen-32B
    do
        export llm=${llm}
        export llm_save_root=Graphia/results/${llm}
        export report_save_root=Graphia/reports/${llm}
        for mode in eval_tf
        do
            export mode=${mode}
            bash Graphia/scripts/postprocess_edge_loop.sh
        done
    done

done
