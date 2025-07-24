
# query graph


export data_root=/data/oss_bucket_0/jjr/LLMGGen
# export data_root=./data
export data_name=8days_dytag_small_text_en
export dx_src_root=saved_results_deg/InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue/${data_name}
save_root="/home/jijiarui.jjr/ROLL/LLMGGen"

python -m llmggen.src_edge_offline  --data_name $data_name --split train --data_root ${data_root} --rl --sft --save_root ${save_root}
python -m llmggen.src_edge_offline  --data_name $data_name --split val  --data_root ${data_root}  --rl --save_root ${save_root}
# python -m llmggen.src_edge_offline --data_name $data_name --split test  --data_root ${data_root} --dx_src_path $dx_src_root/test_degree.pt --infer_dst --save_root ${save_root}


# echo "python -m llmggen.src_edge_offline --data_name $data_name --split test  --data_root ${data_root} --dx_src_path $dx_src_root/test_degree.pt --infer_dst --rl "

