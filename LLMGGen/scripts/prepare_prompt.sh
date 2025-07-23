
# query graph




python -m llmggen.src_edge_offline  --data_name $data_name --split train --data_root ${data_root} --rl --sft
python -m llmggen.src_edge_offline  --data_name $data_name --split val  --data_root ${data_root}  --rl 
python -m llmggen.src_edge_offline --data_name $data_name --split test  --data_root ${data_root} --dx_src_path $dx_src_root/test_degree.pt --infer_dst --rl 


# echo "python -m llmggen.src_edge_offline --data_name $data_name --split test  --data_root ${data_root} --dx_src_path $dx_src_root/test_degree.pt --infer_dst --rl "

