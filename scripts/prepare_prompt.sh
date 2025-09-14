
# query graph
export data_root=/data/oss_bucket_0/jjr/LLMGGen
# export data_root=./data

## 8days_dytag_small_text_en
# export data_name=8days_dytag_small_text_en
# dx_model_name=InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue_rescaleTrue

# weibo_daily
# export data_name=weibo_daily
# dx_model_name=InformerDecoder_seed0_bwr2048_qmFalse_ufbert_cmTrue_rescaleFalse

# # weibo_tech
# export data_name=weibo_tech
# dx_model_name=InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue

## imdb
# export data_name=imdb
# dx_model_name=InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue


export dx_src_root=/home/jijiarui.jjr/ROLL/saved_results_deg/${dx_model_name}/${data_name}

save_root="/home/jijiarui.jjr/ROLL/LLMGGen"

## tdgg
# nohup python -m LLMGGen.src_edge_offline  --data_name $data_name --split train --data_root ${data_root} --rl --sft --save_root ${save_root} --idgg_rl --dx_src_path $dx_src_root/test_degree.pt 
# nohup python -m LLMGGen.src_edge_offline  --data_name $data_name --split val  --data_root ${data_root}  --rl --save_root ${save_root}
# nohup python -m LLMGGen.src_edge_offline --data_name $data_name --split test  --data_root ${data_root} --rl --dx_src_path $dx_src_root/test_degree.pt --infer_dst --save_root ${save_root}


## idgg
echo "python -m LLMGGen.src_edge_offline  --data_name $data_name --split train --data_root ${data_root} --save_root ${save_root} --idgg_rl --dx_src_path $dx_src_root/test_degree.pt"
python -m LLMGGen.src_edge_offline  --data_name $data_name --split train --data_root ${data_root} --save_root ${save_root} --idgg_rl --dx_src_path $dx_src_root/test_degree.pt
python -m LLMGGen.src_edge_offline --data_name $data_name --split test  --data_root ${data_root} --dx_src_path $dx_src_root/test_degree.pt --infer_dst --save_root ${save_root}



python  -m LLMGGen.convert_prompt_csv --phase "c2j" \
        --output_dir "/data/oss_bucket_0/jjr/LLMGGen/prompt_data" \
        --root_dir "/home/jijiarui.jjr/ROLL/LLMGGen/prompts"