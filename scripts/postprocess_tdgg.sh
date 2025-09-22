export data_root=LLMGGen/data
export HF_ENDPOINT=https://hf-mirror.com

## these code should be executed after llm inference
python  -m LLMGGen.convert_prompt_csv --phase "j2c" \
        --output_dir "/home/jijiarui.jjr/ROLL/LLMGGen/results" \
        --root_dir "/data/oss_bucket_0/jjr/LLMGGen/results"

# Loop through datasets
# for data_name in 8days_dytag_small_text_en weibo_daily weibo_tech imdb propagate_large_cn; do
for data_name in propagate_large_cn imdb ; do
    export data_name
    
    # Set paths and parameters based on data_name
    case "$data_name" in
        "8days_dytag_small_text_en")
            export dx_src_root="LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_8days_dytag_small_text_en_rl-all-domain_300epoch"
            tdgg_rl_edge_model="grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-edge"

            tdgg_rl_seq_query_model="grpo_8days_dytag_small_text_en_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_8days_dytag_small_text_en_sotopia_edge_50epoch"
            export bwr=1980
            export time_window=86400
            ;;
        "weibo_daily")
            export dx_src_root="LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_weibo_daily_query_all_dst_50epoch"
            tdgg_rl_edge_model="grpo_weibo_daily_rl-all-domain_100epoch"

            tdgg_rl_seq_query_model="grpo_weibo_daily_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_weibo_daily_sotopia_edge_50epoch"
            export bwr=2048
            export time_window=86400
            ;;
        "weibo_tech")
            export dx_src_root="LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_weibo_tech_query_all_dst_50epoch"
            tdgg_rl_edge_model="grpo_weibo_tech_rl-all-domain_100epoch"

            tdgg_rl_seq_query_model="grpo_weibo_tech_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_weibo_tech_sotopia_edge_50epoch"
            export bwr=2048
            export time_window=86400
            ;;
        "imdb")
            export dx_src_root="LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_imdb_query_all_dst_200epoch"
            tdgg_rl_edge_model="grpo_imdb_rl-all-domain_100epoch"

            tdgg_rl_seq_query_model="grpo_imdb_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_imdb_sotopia_edge_50epoch"
            
            export bwr=2048
            export time_window=31536000
            ;;
        "propagate_large_cn")
            export dx_src_root="LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}"
            tdgg_rl_query_model="grpo_propagate_large_cn_query_all_dst_200epoch"
            tdgg_rl_edge_model="grpo_propagate_large_cn_rl-all-domain_150epoch"

            tdgg_rl_seq_query_model="grpo_propagate_large_cn_LIKR_reward_query_50epoch"
            tdgg_rl_seq_edge_model="grpo_propagate_large_cn_sotopia_edge_50epoch"
            export bwr=2048
            export time_window=86400
            ;;
        *)
            echo "Unknown dataset: $data_name"
            exit 1
            ;;
    esac

    # query: tdgg
    # for llm in $tdgg_rl_query_model qwen3 qwen3_sft Mixtral-8x7B-v0.1 Meta-Llama-3.1-70B-Instruct Qwen3-32B DeepSeek-R1-Distill-Qwen-32B
    # for llm in qwen3 qwen3_sft
    # for llm in grpo_propagate_large_cn_query_all_dst_50epoch grpo_propagate_large_cn_query_all_dst_150epoch
    for llm in Meta-Llama-3.1-70B-Instruct $tdgg_rl_seq_query_model
    # for llm in $tdgg_rl_query_model 
    do
        export llm=${llm}
        export llm_save_root=LLMGGen/results/${llm}
        export report_save_root=LLMGGen/reports/${llm}
        for mode in process_tf
        do
            export mode=${mode}
            bash LLMGGen/scripts/postprocess_ggen_loop.sh
        done
    done

    # edge: tdgg
    # for llm in $tdgg_rl_edge_model
    for llm in Meta-Llama-3.1-70B-Instruct $tdgg_rl_seq_edge_model
    # for llm in $tdgg_rl_edge_model
    do
        export llm=${llm}
        export llm_save_root=LLMGGen/results/${llm}
        export report_save_root=LLMGGen/reports/${llm}
        for mode in process_tf eval_tf
        do
            export mode=${mode}
            bash LLMGGen/scripts/postprocess_edge_loop.sh
        done
    done

done

# python  -m LLMGGen.convert_prompt_csv --phase "c2j" \
#         --output_dir "/data/oss_bucket_0/jjr/LLMGGen/prompt_data" \
#         --root_dir "/home/jijiarui.jjr/ROLL/LLMGGen/prompts"