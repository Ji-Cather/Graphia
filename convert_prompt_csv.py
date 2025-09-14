import os
import pandas as pd
import glob
from pathlib import Path

def safe_eval(value):
    """
    安全地将字符串转换为列表，保留非字符串值。
    """
    if isinstance(value, str):
        try:
            return eval(value)
        except Exception as e:
            print(f"转换失败: {value}，错误: {e}")
            return value  # 转换失败时保留原值
    return value  # 非字符串直接返回

def convert_csv_to_jsonl(convert_directory,root_dir = "data"):
    # Define the root directory
    

    # Walk through all subdirectories and files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_path = os.path.join(dirpath, filename)
                jsonl_path = os.path.splitext(csv_path)[0].replace(root_dir,convert_directory) + ".jsonl"

                # Read CSV file
                df = pd.read_csv(csv_path)
                
                if "gt_dst_idxs_unique" in df.columns:
                    df["gt_dst_idxs_unique"] = df["gt_dst_idxs_unique"].map(safe_eval)
                    df["gt_dst_idxs_all"] = df["gt_dst_idxs_all"].map(safe_eval)
                

                # Reset index and rename it to 'id'
                df.reset_index(inplace=True)
                df.rename(columns={"index": "id"}, inplace=True)
                
                os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                # Write to JSONL (each row is a JSON object on a new line)
                df.to_json(jsonl_path, orient="records", lines=True)

                print(f"Converted: {csv_path} -> {jsonl_path}")


def test_data_load():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from roll.distributed.scheduler.protocol import DataProto
    from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys

    tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct/")
    dataset_path = "data/test_ctdg.jsonl"
    dataset = load_dataset("json", data_files=dataset_path)["train"]


    # 加上format，然后转ids的func
    def encode_function(data_i):
        text_list = []
        for instruct in data_i["prompt"]:
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": instruct},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_list.append(text)
        encodings = tokenizer(text_list)
        return encodings


    # 处理数据
    print(dataset)
    dataset = dataset.map(encode_function, batched=True, desc="Encoding dataset")
    print(dataset)
    # 过滤cutoff
    # dataset = dataset.filter(lambda data_i: len(data_i["input_ids"]) <= 512, desc="Filtering dataset")
    # print(dataset)
    # ------
    data_collator = DataCollatorWithPaddingForPaddedKeys(
        tokenizer=tokenizer,
        max_length=1024,
        padding="max_length",
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=data_collator)

    for batch_dict in tqdm(dataloader):
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        print(type(batch.non_tensor_batch))
        for key, value in batch.non_tensor_batch.items():
            print("-" * 20)
            print(key, type(value), value.shape)
            print(value[0])
            print(value[1])
            print(value[2])
            new_value = np.repeat(value, 3)
            print("*" * 20)
            print(new_value.shape)
            print(new_value[0])
            print(new_value[1])
            print(new_value[2])
            print(new_value[3])
            print(new_value[4])
            print(new_value[5])
            print(new_value[6])
            print(new_value[7])
            print(new_value[8])
        break

def assign_difficulty(jsonl_path):
    import os
    import json
    import numpy as np

    # First pass: collect lengths from this file
    lengths = []
    with open(jsonl_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            lengths.append(data.get("gt_dx_src_unique", 0))

    # Compute quantiles
    quantiles = np.percentile(lengths, [30, 70])

    # Create a temporary file
    temp_path = jsonl_path + ".tmp"

    # Second pass: write updated content to temp file
    with open(jsonl_path, 'r') as infile, open(temp_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            length = data.get("gt_dx_src_unique", 0)

            if length <= quantiles[0]:
                difficulty = 3
            elif length <= quantiles[1]:
                difficulty = 2
            else:
                difficulty = 1
            data["difficulty"] = difficulty
            outfile.write(json.dumps(data) + "\n")

    # Replace original file with updated file
    os.replace(temp_path, jsonl_path)

    print(f"Assigned difficulty and overwritten: {jsonl_path}")


def filter_and_overwrite_difficulty_1(
    source_path="/data/oss_bucket_0/jjr/LLMGGen/prompt_data/8days_dytag_small_text_en/teacher_forcing/train_query_examples.jsonl",
    target_path="/data/oss_bucket_0/jjr/LLMGGen/prompt_data/8days_dytag_small_text_en/teacher_forcing/test_ctdg.jsonl"
):
    import os
    import json

    # Create a temporary file
    temp_path = target_path + ".tmp"

    with open(source_path, 'r') as infile, open(temp_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            if data.get("difficulty") == 1:
                outfile.write(json.dumps(data) + "\n")

    # Replace original target file with filtered content
    os.replace(temp_path, target_path)
    print(f"Filtered and overwritten {target_path} with only difficulty=1 samples.")

def convert_difficulty_dir(output_dir: str):
    for dirpath, _, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.endswith(".jsonl") and "query" in filename:
                jsonl_path = os.path.join(dirpath, filename)
                assign_difficulty(jsonl_path)


def convert_jsonl_to_csv(root_dir, output_dir):


    # 遍历所有子目录并筛选叶子文件夹
    for root, dirs, files in os.walk(root_dir):
        # 判断是否是叶子文件夹（没有子文件夹）
        if not dirs:
            for pattern, cols_save in zip(["query_examples.jsonl_*", 
                                            "query_examples_8.jsonl_*", 
                                            "edge_text_prompt.jsonl_*",
                                            "edge_text_eval_prompt.jsonl_*"],
                                            [["predict","src_idx"],
                                             ["predict","src_idx"],
                                              ["predict","src_idx","dst_idx","edge_id","prompt"],
                                              ["predict","edge_id"]]):
                print(pattern)
                # 查找当前目录下所有匹配的 jsonl 文件
                jsonl_files = glob.glob(os.path.join(root, pattern))
                if len(jsonl_files)==0: continue
                jsonl_path = os.path.join(root, jsonl_files[0])
                csv_path = os.path.splitext(jsonl_path)[0].replace(root_dir,output_dir) + ".csv"
                
                if jsonl_files:
                    print(f"Processing folder: {root}")
                    try:
                        # 读取所有 jsonl 文件到 DataFrame
                        df_list = []
                        for file in jsonl_files:
                            df = pd.read_json(file, lines=True)
                            df_list.append(df)
                    except:
                        continue
                    
                    # 合并所有文件
                    merged_df = pd.concat(df_list, ignore_index=True)
                    
                    # 检查是否存在 'id' 列
                    if 'id' in merged_df.columns:
                        # 设置 id 列为索引
                        merged_df.set_index('id', inplace=True)
                    else:
                        print(f"Warning: 'id' column not found in {root}, using default index")
                    
                    for col in ["gt_label","output","gt_dst_idxs_unique","gt_dx_src_unique"]:
                        if col in merged_df.columns:
                            cols_save.append(col)
                    try:
                        merged_df = merged_df[cols_save]
                    except:
                        print(jsonl_path)
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    # Write to JSONL (each row is a JSON object on a new line)
                    print(f"Merged {len(jsonl_files)} files into {csv_path}")
                    merged_df.to_csv(csv_path, index=True, escapechar='\\')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="", help="files to be saved")
    parser.add_argument('--root_dir', type=str, default="", help="files to be converted")
    parser.add_argument('--phase', type=str, default="c2j", help="files to be converted") # c2j or j2c
    args = parser.parse_args()

    if args.phase == "c2j":
        convert_csv_to_jsonl(args.output_dir,args.root_dir)
        
    elif args.phase == "j2c":
        convert_jsonl_to_csv(args.root_dir,args.output_dir)
    