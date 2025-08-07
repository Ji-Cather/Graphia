export data_name=8days_dytag_small_text_en
export data_root=/data/oss_bucket_0/jjr/LLMGGen
# export data_root=LLMGGen/data
export dx_src_root=LLMGGen/saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}
export HF_ENDPOINT=https://hf-mirror.com


python  -m LLMGGen.convert_prompt_csv --phase "j2c" \
        --output_dir "/home/jijiarui.jjr/ROLL/LLMGGen/results" \
        --root_dir "/data/oss_bucket_0/jjr/LLMGGen/results"

## baseline eval
# python -m LLMGGen.eval_utils.eval_src_edges \
#     --data_root $data_root \
#     --data_name $data_name \
#     --time_window 86400 --bwr 1980 --use_feature bert \
#     --pred_ratio 0.15 \
#     --split test \
#     --cm_order True\
#     --cut_off_baseline edge \
#     # --node_msg \
#     # --edge_msg \


for llm in grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-dst-2 
# for llm in grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-edge qwen3 qwen3_sft 
do
    export llm=${llm}
    export llm_save_root=LLMGGen/results/${llm}
    export report_save_root=LLMGGen/reports/${llm}
    for mode in process_inf eval_inf
    do
        export mode=${mode}
        bash LLMGGen/scripts/postprocess_ggen_loop.sh
        # bash LLMGGen/scripts/postprocess_edge_loop.sh
    done
done

