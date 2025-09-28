export data_root=LLMGGen/data
export HF_ENDPOINT=https://hf-mirror.com

## these code should be executed after llm inference
## these code should be executed after llm inference



export data_name=8days_dytag_small_text_en
export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}
tdgg_rl_edge_model=grpo_8days_dytag_small_text_en_rl-all-domain_300epoch
export time_window=86400
export bwr=1980

export data_name=weibo_daily
export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}
tdgg_rl_edge_model=grpo_weibo_daily_rl-all-domain_100epoch
tdgg_rl_query_model=


# export data_name=weibo_tech
# export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}
# tdgg_rl_edge_model=grpo_weibo_tech_rl-all-domain_100epoch
# tdgg_rl_query_model=grpo_weibo_tech_query_all_dst_50epoch



export data_name=8days_dytag_small_text_en
export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}
tdgg_rl_edge_model=grpo_8days_dytag_small_text_en_rl-all-domain_300epoch
export time_window=86400
export bwr=1980

export data_name=weibo_daily
export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}
tdgg_rl_edge_model=grpo_weibo_daily_rl-all-domain_100epoch
tdgg_rl_query_model=


# export data_name=weibo_tech
# export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue/${data_name}
# tdgg_rl_edge_model=grpo_weibo_tech_rl-all-domain_100epoch
# tdgg_rl_query_model=grpo_weibo_tech_query_all_dst_50epoch


## baseline eval
# python -m LLMGGen.eval_utils.eval_src_edges \
#     --data_root $data_root \
#     --data_name 8days_dytag_small_text_en \
#     --data_name 8days_dytag_small_text_en \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --cut_off_baseline edge \



# query: tdgg
# for llm in $tdgg_rl_query_model qwen3 qwen3_sft 
# do
#     export llm=${llm}
#     export llm_save_root=LLMGGen/results/${llm}
#     export report_save_root=LLMGGen/reports/${llm}
#     for mode in process_tf eval_tf
#     do
#         export mode=${mode}
#         bash LLMGGen/scripts/postprocess_ggen_loop.sh >> query_${data_name}.log
#     done
# done


# edge: tdgg
# get edge text, and edge text eval
# for llm in grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-edge qwen3 qwen3_sft 
for llm in $tdgg_rl_edge_model qwen3 qwen3_sft 
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

# # edge text eval result: tdgg
# for llm in $tdgg_rl_edge_model qwen3 qwen3_sft 
# do
#     export llm=${llm}
#     export llm_save_root=LLMGGen/results/${llm}
#     export report_save_root=LLMGGen/reports/${llm}
#     for mode in eval_tf
#     do
#         export mode=${mode}
#         bash LLMGGen/scripts/postprocess_edge_loop.sh
#     done
# done



## query: idgg
# for llm in $tdgg_rl_query_model qwen3 qwen3_sft 
# do
#     export llm=${llm}
#     export llm_save_root=LLMGGen/results/${llm}
#     export report_save_root=LLMGGen/reports/${llm}
#     for mode in process_inf eval_inf
#     do
#         export mode=${mode}
#         bash LLMGGen/scripts/postprocess_ggen_loop.sh
#     done
# done