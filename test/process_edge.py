import json

data_set = "weibo_tech"
data_set = "imdb"

for split in ["val"]:
    # 读取原始 JSON 文件
    input_path = f'/data/oss_bucket_0/jjr/LLMGGen/prompt_data/{data_set}/{split}/teacher_forcing/edge_text_examples.jsonl'         # 输入文件路径
    output_path = f'/data/oss_bucket_0/jjr/LLMGGen/prompt_data/{data_set}/{split}/teacher_forcing/edge_text_examples_cut.jsonl'  # 输出文件路径
    if split == 'val':
        sample_num = 300

    filtered_examples = []

    # 逐行读取并解析每个 JSON 对象
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                example = json.loads(line)
                # 检查 difficulty 是否存在且等于 1
                if example.get('domain') != "dst_rule":
                    filtered_examples.append(example)
            except json.JSONDecodeError as e:
                print(f"跳过无法解析的行: {line}, 错误: {e}")


    import random
    filtered_examples = random.sample(filtered_examples, sample_num)  # without replacement

    # 保存到指定路径
    # 写入 JSONL 文件：每行一个 JSON 对象
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in filtered_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')