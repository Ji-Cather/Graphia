# Graphia/eval_utils/latex_tdgg.py
import pandas as pd
import numpy as np
from pathlib import Path

def format_value(value):
    """格式化数值，处理NaN值"""
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f"{value:.4f}"
    else:
        return "-"

def format_latex_table_by_dataset(df, columns, caption="", label="", score_column=""):
    """
    按数据集分组生成LaTeX表格，并为每个数据集添加分隔线，
    同时为每个数据集的top1和top2模型的数值添加格式
    """
    # 重命名模型以缩短名称
    model_rename_map = {
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    df = df.copy()
    df['model'] = df['model'].replace(model_rename_map)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\begin{tabular}{llrrrrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & \\multicolumn{2}{c}{Easy} & \\multicolumn{2}{c}{Hard} & \\multicolumn{2}{c}{All} \\\\")
    latex_lines.append(" &  & Hit@100 & Recall@100 & Hit@100 & Recall@100 & Hit@100 & Recall@100 \\\\")
    latex_lines.append("\\midrule")
    
    # 定义模型顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'Qwen3-32B',
        'DeepSeek-Q-32B',
        'Llama3-70B',
        'Graphia-seq',
        'Graphia'
    ]
    
    # 按数据集分组处理
    for dataset in df['dataset'].unique():
        # 格式化数据集名称（下划线变空格，首字母大写）
        formatted_dataset = dataset.replace('_', ' ').title()
        
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # 根据模型顺序重新排列数据
        ordered_data = []
        for model_name in model_order:
            model_data = dataset_df[dataset_df['model'] == model_name]
            if not model_data.empty:
                ordered_data.append(model_data)
        
        # 合并排序后的数据
        if ordered_data:
            sorted_dataset_df = pd.concat(ordered_data, ignore_index=True)
        else:
            sorted_dataset_df = dataset_df.reset_index(drop=True)
        
        # 为每个数值列计算排名（降序，因为都是正向指标）
        # 包括score_column在内
        for col in columns:
            if col in sorted_dataset_df.columns:
                sorted_dataset_df[f'{col}_rank'] = sorted_dataset_df[col].rank(method='min', ascending=False)
        
        # 添加数据行（保持模型顺序）
        for idx, row in sorted_dataset_df.iterrows():
            # 第一列是数据集名称（仅在第一行显示）
            dataset_name = formatted_dataset if idx == 0 else ""
            
            line = f"{dataset_name} & {row['model']}"
            
            # 为top1和top2添加格式
            for col in columns:
                value = row[col]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    formatted_value = f"{value:.4f}"
                    # 检查是否需要加粗或下划线
                    if f'{col}_rank' in sorted_dataset_df.columns:
                        rank = row[f'{col}_rank']
                        if rank == 1:
                            # top1 加粗
                            formatted_value = f"\\textbf{{{formatted_value}}}"
                        elif rank == 2:
                            # top2 下划线
                            formatted_value = f"\\underline{{{formatted_value}}}"
                    
                    line += f" & {formatted_value}"
                else:
                    line += f" & -"
            line += " \\\\"
            latex_lines.append(line)
        
        # 清理临时列
        for col in columns:
            if f'{col}_rank' in sorted_dataset_df.columns:
                sorted_dataset_df.drop(f'{col}_rank', axis=1, inplace=True)
        
        latex_lines.append("\\midrule")
    
    # 移除最后的 \midrule
    if latex_lines[-1] == "\\midrule":
        latex_lines.pop()
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

def format_latex_table_recall_only(df, columns, caption="", label="", score_column=""):
    """
    生成只包含Recall@100指标的LaTeX表格
    """
    # 重命名模型以缩短名称
    model_rename_map = {
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    df = df.copy()
    df['model'] = df['model'].replace(model_rename_map)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\begin{tabular}{lrrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & Easy & Hard & All \\\\")
    latex_lines.append("\\midrule")
    
    # 定义模型顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'Qwen3-32B',
        'DeepSeek-Q-32B',
        'Llama3-70B',
        'Graphia-seq',
        'Graphia'
    ]
    
    # 按数据集分组处理
    for dataset in df['dataset'].unique():
        # 格式化数据集名称（下划线变空格，首字母大写）
        formatted_dataset = dataset.replace('_', ' ').title()
        
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # 根据模型顺序重新排列数据
        ordered_data = []
        for model_name in model_order:
            model_data = dataset_df[dataset_df['model'] == model_name]
            if not model_data.empty:
                ordered_data.append(model_data)
        
        # 合并排序后的数据
        if ordered_data:
            sorted_dataset_df = pd.concat(ordered_data, ignore_index=True)
        else:
            sorted_dataset_df = dataset_df.reset_index(drop=True)
        
        # 为每个数值列计算排名（降序，因为都是正向指标）
        # 包括score_column在内
        for col in columns:
            if col in sorted_dataset_df.columns:
                sorted_dataset_df[f'{col}_rank'] = sorted_dataset_df[col].rank(method='min', ascending=False)
        
        # 添加数据行（保持模型顺序）
        for idx, row in sorted_dataset_df.iterrows():
            # 第一列是数据集名称（仅在第一行显示）
            dataset_name = formatted_dataset if idx == 0 else ""
            
            line = f"{dataset_name} & {row['model']}"
            
            # 为top1和top2添加格式
            for col in columns:
                value = row[col]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    formatted_value = f"{value:.4f}"
                    # 检查是否需要加粗或下划线
                    if f'{col}_rank' in sorted_dataset_df.columns:
                        rank = row[f'{col}_rank']
                        if rank == 1:
                            # top1 加粗
                            formatted_value = f"\\textbf{{{formatted_value}}}"
                        elif rank == 2:
                            # top2 下划线
                            formatted_value = f"\\underline{{{formatted_value}}}"
                    
                    line += f" & {formatted_value}"
                else:
                    line += f" & -"
            line += " \\\\"
            latex_lines.append(line)
        
        # 清理临时列
        for col in columns:
            if f'{col}_rank' in sorted_dataset_df.columns:
                sorted_dataset_df.drop(f'{col}_rank', axis=1, inplace=True)
        
        latex_lines.append("\\midrule")
    
    # 移除最后的 \midrule
    if latex_lines[-1] == "\\midrule":
        latex_lines.pop()
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

def generate_retrieval_table(df, output_path):
    """
    生成retrieval metrics表格（完整版）
    """
    # 获取retrieval相关的列，按特定顺序排列
    retrieval_columns = [
        'selection_hit@100_Easy', 'selection_recall@100_Easy',
        'selection_hit@100_Hard', 'selection_recall@100_Hard',
        'selection_hit@100_All', 'selection_recall@100_All',
        # 'selection_score'
    ]
    
    # 生成LaTeX表格
    latex_table = format_latex_table_by_dataset(
        df, 
        retrieval_columns,
        caption="Retrieval Metrics for Different Datasets",
        label="tab:retrieval_metrics",
        score_column="selection_score"
    )
    
    # 保存到文件
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Retrieval表格已保存至: {output_path}")
    return latex_table

def generate_retrieval_recall_table(df, output_path):
    """
    生成只包含Recall@100指标的表格
    """
    # 获取recall@100相关的列
    recall_columns = [
        'selection_recall@100_Easy',
        'selection_recall@100_Hard',
        'selection_recall@100_All',
        # 'selection_score'
    ]
    
    # 生成LaTeX表格
    latex_table = format_latex_table_recall_only(
        df,
        recall_columns,
        caption="Retrieval Recall@100 Metrics for Different Datasets",
        label="tab:retrieval_recall_metrics",
        score_column="selection_score"
    )
    
    # 保存到文件
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Retrieval Recall表格已保存至: {output_path}")
    return latex_table

def generate_edge_table(df, output_path):
    """
    生成edge metrics表格
    """
    # 重命名模型以缩短名称
    model_rename_map = {
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    df = df.copy()
    df['model'] = df['model'].replace(model_rename_map)
    
    # 获取edge相关的列
    edge_columns = [
        'edge_label_acc', 
        'edge_ROUGE_L', 
        'edge_BERTScore_F1',
        'edge_score'
    ]
    
    # 生成LaTeX表格
    latex_table = format_latex_table_by_dataset(
        df,
        edge_columns,
        caption="Edge Metrics and Average LLM Rating",
        label="tab:edge_metrics",
        score_column="edge_score"
    )
    
    # 保存到文件
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Edge表格已保存至: {output_path}")
    return latex_table

def process_tdgg_for_latex(
    input_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
    retrieval_output_path="Graphia/reports/latex_retrieval_table.tex",
    retrieval_recall_output_path="Graphia/reports/latex_retrieval_recall_table.tex",
    edge_output_path="Graphia/reports/latex_edge_table.tex"
):
    """
    主函数：处理tdgg_social_fidelity_scores.csv并生成LaTeX表格
    """
    # 读取数据
    df = pd.read_csv(input_file_path)
    
    # 重命名模型
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft'
    }
    df['model'] = df['model'].replace(model_rename_map)
    
    # 重命名数据集
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
       
    }
    df['dataset'] = df['dataset'].replace(dataset_rename_map)
    
    # 确保输出目录存在
    output_dir = Path(retrieval_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成retrieval表格（完整版）
    retrieval_latex = generate_retrieval_table(df, retrieval_output_path)
    
    # 生成retrieval表格（仅Recall@100）
    retrieval_recall_latex = generate_retrieval_recall_table(df, retrieval_recall_output_path)
    
    # 生成edge表格
    edge_latex = generate_edge_table(df, edge_output_path)
    
    print("✅ LaTeX表格生成完成!")
    print(f"📊 数据形状: {df.shape}")
    print("📋 包含的模型:")
    for model in df['model'].unique():
        print(f"  - {model}")
    
    return retrieval_latex, retrieval_recall_latex, edge_latex

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成TDGG评估结果的LaTeX表格")
    parser.add_argument("--input_file", type=str,
                        default="Graphia/reports/tdgg_social_fidelity_scores_cut.csv",
                        help="输入的CSV文件路径")
    parser.add_argument("--retrieval_output", type=str,
                        default="Graphia/reports/latex_retrieval_table.tex",
                        help="retrieval表格输出路径")
    parser.add_argument("--retrieval_recall_output", type=str,
                        default="Graphia/reports/latex_retrieval_recall_table.tex",
                        help="retrieval recall表格输出路径")
    parser.add_argument("--edge_output", type=str,
                        default="Graphia/reports/latex_edge_table.tex",
                        help="edge表格输出路径")
    
    args = parser.parse_args()
    
    # 执行处理
    process_tdgg_for_latex(
        input_file_path=args.input_file,
        retrieval_output_path=args.retrieval_output,
        retrieval_recall_output_path=args.retrieval_recall_output,
        edge_output_path=args.edge_output
    )