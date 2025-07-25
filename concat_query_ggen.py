import pandas as pd
import os
import glob
import argparse

def concat_model_results_to_baseline(data_name, split, phase="inference"):
    """
    将各模型的*_matrix.csv结果合并到baseline的matrix.csv中，model列为模型名/匹配方式
    """
    # baseline路径
    baseline_dir = os.path.join("reports", "baselines", data_name, split, phase)
    baseline_pattern = os.path.join(baseline_dir, "*_matrix.csv")
    baseline_files = glob.glob(baseline_pattern)
    if not baseline_files:
        print(f"未找到baseline文件: {baseline_pattern}")
        return
    baseline_file = baseline_files[0]
    # 读取baseline
    baseline_df = pd.read_csv(baseline_file)
    # baseline的匹配方式
    baseline_match = os.path.basename(baseline_file).replace("_matrix.csv", "")
    baseline_df["model"] = f"baseline/{baseline_match}"

    # 查找其他模型的matrix.csv
    reports_dir = os.path.join("reports")
    model_dirs = [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d)) and d != "baselines"]
    all_dfs = [baseline_df]

    for model in model_dirs:
        model_matrix_dir = os.path.join(reports_dir, model, data_name, split, phase)
        if not os.path.exists(model_matrix_dir):
            continue
        matrix_files = glob.glob(os.path.join(model_matrix_dir, "*_matrix.csv"))
        for matrix_file in matrix_files:
            match_type = os.path.basename(matrix_file).replace("_matrix.csv", "")
            df = pd.read_csv(matrix_file)
            df["model"] = f"{model}/{match_type}"
            all_dfs.append(df)

    # 合并所有结果
    concat_df = pd.concat(all_dfs, ignore_index=True)
    # dataset和split列替换为args.data_name和args.split（如果存在）
    if "dataset" in concat_df.columns:
        concat_df["dataset"] = data_name
    if "split" in concat_df.columns:
        concat_df["split"] = split
    
    # 将model列放到第一列
    cols = list(concat_df.columns)
    if "model" in cols:
        cols.insert(0, cols.pop(cols.index("model")))
        concat_df = concat_df[cols]
    # 保存到baseline目录下
    output_file = os.path.join(baseline_dir, "all_models_matrix.csv")
    concat_df.to_csv(output_file, index=False)
    print(f"已保存合并结果到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--split', type=str, required=True, help='数据集分割')
    parser.add_argument('--phase', type=str, default="inference", help='数据集分割')
    # 允许传递任意额外参数
    args, unknown = parser.parse_known_args()
   
    concat_model_results_to_baseline(args.data_name, args.split, args.phase)
