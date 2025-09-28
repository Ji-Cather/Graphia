#!/bin/bash

# 定义数据根目录
export data_root=LLMGGen/data
export save_root="LLMGGen"

# 遍历数据集
for data_name in 8days_dytag_small_text_en weibo_daily weibo_tech; do
    export data_name
    
    # 根据 data_name 设置不同的模型名称和参数
    case "$data_name" in
        "8days_dytag_small_text_en")
            dx_model_name=InformerDecoder_seed0_bwr1980_qmTrue_ufbert_cmTrue_rescaleTrue
            ;;
        "propagate_large_cn"|"weibo_daily"|"weibo_tech"|"imdb")
            dx_model_name=InformerDecoder_seed0_bwr2048_qmTrue_ufbert_cmTrue_rescaleTrue
            ;;
        *)
            echo "Unknown dataset: $data_name"
            continue
            ;;
    esac
    
    # 设置当前数据集的源路径
    export dx_src_root=/data/jiarui_ji/llmggen/LLMGGen/saved_results_deg/${dx_model_name}/${data_name}
    
    echo "Processing dataset: $data_name with degree model: $dx_model_name"
    
    # 执行相应的处理命令（这里以 seq 模式为例）
    python -m LLMGGen.src_edge_offline \
        --data_name $data_name \
        --split train \
        --data_root ${data_root} \
        --edge_seq \
        # --dst_seq \
        # --edge_seq \
    
    # 如果需要处理 test 集，可以取消注释以下内容
    # python -m LLMGGen.src_edge_offline \
    #     --data_name $data_name \
    #     --split test \
    #     --data_root ${data_root} \
    #     --dx_src_path $dx_src_root/test_degree.pt \
    #     --infer_dst \
    #     --save_root ${save_root}
    
    # 如果需要处理 val 集，可以取消注释以下内容
    # python -m LLMGGen.src_edge_offline \
    #     --data_name $data_name \
    #     --split val \
    #     --data_root ${data_root} \
    #     --rl \
    #     --save_root ${save_root}
done

# 转换提示CSV（所有数据集处理完后执行一次）
python -m LLMGGen.convert_prompt_csv \
    --phase "c2j" \
    --output_dir "/data/oss_bucket_0/jjr/LLMGGen/prompt_data" \
    --root_dir "/home/jijiarui.jjr/ROLL/LLMGGen/prompts"