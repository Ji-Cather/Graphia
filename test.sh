export data_root=/data/oss_bucket_0/jjr/LLMGGen
# export data_root=LLMGGen/data

for llm in qwen3 qwen3_sft grpo_8days_dytag_small_text_en_qwen3_sft-8b-nothink-easy-dst
do
    export llm=${llm}
    # 在这里添加你需要对每个llm执行的命令
done

export data_name=8days_dytag_small_text_en
export dx_src_root=saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}

export llm_save_root=results/${llm}
export report_save_root=reports/${llm}

export llm_save_root=LLMGGen/results/${llm}
export report_save_root=LLMGGen/reports/${llm}