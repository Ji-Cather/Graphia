import os
from pathlib import Path
import pandas as pd
import argparse

def merge_graph_matrices(root_dir="LLMGGen/reports", 
                         output_root="merged_graph_matrix.csv",
                         file_name="graph_matrix.csv"):
    root = Path(root_dir)
    all_data = []

    # 递归查找所有 graph_matrix.csv 文件
    for csv_path in root.rglob(file_name):
        try:
            # 解析路径：model, dataset, split, task
            parts = csv_path.relative_to(root).parts
            if len(parts) < 5:
                print(f"[警告] 路径层级不足，跳过: {csv_path}")
                continue

            model, dataset, split, task = parts[:4]

            # 读取 CSV
            if file_name =="dst_retrival_matrix.csv":
                df = pd.read_csv(csv_path)
            else:
                df = pd.read_csv(csv_path, index_col=0)

            # 添加来源信息
            df['model'] = model
            df['dataset'] = dataset
            df['split'] = split
            df['task'] = task

            all_data.append(df)
            print(f"✅ 已加载: {csv_path}")

        except Exception as e:
            print(f"[错误] 无法读取 {csv_path}: {e}")

    if not all_data:
        print("⚠️ 未找到任何 graph_matrix.csv 文件，请检查路径或文件是否存在。")
        return

    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)

    # 保存结果
    output_file = os.path.join(output_root, f"merged_{file_name}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file)
    print(f"📁 输出文件: {output_file}")
    print(f"📊 总记录数: {len(merged_df)} 来自 {len(all_data)} 个文件")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LLMGGen 报告目录下的所有 graph_matrix.csv 文件")
    parser.add_argument("--root", type=str, default="LLMGGen/reports", help="根目录路径，默认: LLMGGen/reports")
    parser.add_argument("--output_root", type=str, default="LLMGGen/reports/concat", help="输出文件名，默认: merged_graph_matrix.csv")
    parser.add_argument("--file_name", type=str, default="graph_matrix.csv", help="要合并的文件名，默认: graph_matrix.csv")
    ## opptional [dst_retrival_matrix.csv, edge_matrix.csv, graph_matrix.csv]

    args = parser.parse_args()
    merge_graph_matrices(root_dir=args.root, output_root=args.output_root, file_name=args.file_name)
