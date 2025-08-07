export PYTHONPATH=/home/jijiarui.jjr/ROLL:/home/jijiarui.jjr/.local/lib/python3.10/site-packages:$PYTHONPATH

export data_root=/data/oss_bucket_0/jjr/LLMGGen
export data_root=./data

export data_name=8days_dytag_small_text_en

# bash scripts/train_dp.sh

export dx_src_root=saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}
bash scripts/prepare_prompt.sh