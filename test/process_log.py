import os
import csv
import re
from pathlib import Path

# 提取配置参数，例如 dataset, bwr, qm, uf，从路径中根据你的格式做微调
def extract_config_from_path(path):
    # 示例路径：./saved_models/.../InformerTranverse_seed0_bwr10000_qmFalse_ufno/...
    match = re.search(r'bwr(\d+)_qm(\w+)_uf(\w+)_cm(\w+)', path)
    if match:
        bwr, qm, uf,cm = match.groups()
    else:
        bwr, qm, uf,cm = 'unknown', 'unknown', 'unknown','unknown'
    dataset = Path(path).parts[-3]  # 修改这个索引以适配你的目录层级
    return dataset, bwr, qm, uf,cm

def parse_value(value_str):
    """
    处理常规 float 值，或者形如 '[tensor(1.2345e-04)]' 的 PyTorch tensor 打印值
    """
    try:
        # 直接尝试转换
        return float(value_str)
    except ValueError:
        # 尝试从 tensor() 中提取
        match = re.search(r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+', value_str)
        if match:
            return float(match.group())
        else:
            return None  # 或 raise ValueError("Unrecognized value: " + value_str)

# 处理单个文件
def process_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()[-18:]  # 最后18行

    results = {'degree': {}, 'unique': {}}
    for line in lines:
        if 'INFO - test' in line:
            parts = line.strip().split('INFO - test ')[-1].split(', ')
            if len(parts) == 2:
                key, value = parts
                value = parse_value(value)
                if value is None:
                    continue
                if key.endswith('_degree'):
                    metric = key.replace('_degree', '')
                    results['degree'][metric] = float(value)
                elif key.endswith('_unique'):
                    metric = key.replace('_unique', '')
                    results['unique'][metric] = float(value)

    return results

# 写入 CSV
def write_to_csv(filepaths, output_path='parsed_metrics.csv'):
    fieldnames = [
        'dataset', 'bwr', 'qm', 'uf', 'cm','type',
        'mae_allbatches', 'mse_allbatches',
        'histogram_wasserstein_distance', 'histogram_kl_divergence', 'histogram_mmd_distance'
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in filepaths:
            results = process_log_file(filepath)
            dataset, bwr, qm, uf, cm = extract_config_from_path(filepath)

            for typ in ['degree', 'unique']:
                row = {
                    'dataset': dataset,
                    'bwr': bwr,
                    'qm': qm,
                    'uf': uf,
                    'type': typ,
                    'cm':cm,
                    'mae_allbatches': results[typ].get('mae_allbatches', ''),
                    'mse_allbatches': results[typ].get('mse_allbatches', ''),
                    'histogram_wasserstein_distance': results[typ].get('histogram_wasserstein_distance', ''),
                    'histogram_kl_divergence': results[typ].get('histogram_kl_divergence', ''),
                    'histogram_mmd_distance': results[typ].get('histogram_mmd_distance', ''),
                }
                writer.writerow(row)


import pandas as pd

def add_rank_score(csv_path, output_path='ranked_output.csv'):
    df = pd.read_csv(csv_path)

    # 需要排序的指标列
    metric_cols = [
        'mae_allbatches',
        'mse_allbatches',
        'histogram_wasserstein_distance',
        'histogram_kl_divergence',
        'histogram_mmd_distance'
    ]

    # 对每个 metric 列按升序排名（最小为1）
    for col in metric_cols:
        rank_col = col + '_rank'
        df[rank_col] = df[col].rank(method='min', ascending=True)

    # 总排名分数
    df['rank_score'] = df[[c + '_rank' for c in metric_cols]].sum(axis=1)

    # 按 rank_score 升序排序
    df = df.sort_values('rank_score')

    df.to_csv(output_path, index=False)
    print(f"Ranked results written to: {output_path}")



# 示例使用
if __name__ == "__main__":
    import sys
    log_files = [
        "logs_deg/InformerTranverse_seed0_bwr2048_qmFalse_ufno_cmFalse/8days_dytag_large/InformerTranverse_seed0_bwr2048_qmFalse_ufno_cmFalse/1747921535.9037557.log",
        "logs_deg/InformerTranverse_seed0_bwr2048_qmFalse_ufno_cmTrue/8days_dytag_large/InformerTranverse_seed0_bwr2048_qmFalse_ufno_cmTrue/1747990305.0879898.log",
        "logs_deg/InformerTranverse_seed0_bwr2048_qmTrue_ufno_cmFalse/8days_dytag_large/InformerTranverse_seed0_bwr2048_qmTrue_ufno_cmFalse/1747922135.8406422.log",
        "logs_deg/InformerTranverse_seed0_bwr2048_qmTrue_ufno_cmTrue/8days_dytag_large/InformerTranverse_seed0_bwr2048_qmTrue_ufno_cmTrue/1748001751.3364367.log",
        "logs_deg/InformerTranverse_seed0_bwr10000_qmFalse_ufno_cmTrue/8days_dytag_large/InformerTranverse_seed0_bwr10000_qmFalse_ufno_cmTrue/1747989379.095886.log"
    ]
    output_path = "result_deg_large.csv"
    
    log_files = [
        "logs_deg/InformerTranverse_seed0_bwr2048_qmFalse_ufno_cmFalse/8days_dytag_small/InformerTranverse_seed0_bwr2048_qmFalse_ufno_cmFalse/1747921348.3259983.log",
        "logs_deg/InformerTranverse_seed0_bwr2048_qmTrue_ufno_cmTrue/8days_dytag_small/InformerTranverse_seed0_bwr2048_qmTrue_ufno_cmTrue/1748001768.4705093.log",
        
        
    ]
    output_path = "result_deg_small.csv"
    if not log_files:
        print("请传入日志文件路径，如：python parse_logs_to_csv.py log1.txt log2.txt")
    else:
        write_to_csv(log_files,output_path=output_path)
        
    # add_rank_score(output_path,output_path=output_path)
