import pandas as pd
import numpy as np
from pathlib import Path

def generate_combined_table(df, output_path):
    """
    生成合并的LaTeX表格，左边是DST任务，右边是Edge任务
    
    Args:
        dst_df: 包含DST任务结果的DataFrame
        edge_df: 包含Edge任务结果的DataFrame
        output_path: 输出文件路径
    """
    # 重命名模型（保持一致性）
    model_rename_map = {
        'qwen3': 'Qwen3-8B',
        'qwen3_sft': 'Qwen3-8B-sft',
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    
    # 重命名数据集
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
        'propagate_large_cn': 'Propagate-Zh'
    }
    
    # 处理DST数据
    df = df.copy()
    df['model'] = df['model'].replace(model_rename_map)
    df['dataset'] = df['dataset'].replace(dataset_rename_map)
    
    
    # 获取DST相关的列（只使用Recall@100）
    dst_columns = [
        'selection_recall@100_Easy',
        'selection_recall@100_Hard',
        'selection_recall@100_All',
        'selection_score',
        'selection_rank_score',
        
    ]
    
    # 获取Edge相关的列
    edge_columns = [
        'edge_label_acc', 
        'edge_ROUGE_L', 
        'edge_BERTScore_F1',
        'edge_score',
        'edge_rank_score'
    ]
    
    # 定义模型顺序
    model_order = [
        'Qwen3-8B',
        'Qwen3-8B-sft', 
        'Qwen3-32B',
        'DeepSeek-Q-32B',
        'Llama3-70B',
        'LLMGGen-seq',
        'LLMGGen'
    ]
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\caption{Combined Results for DST Selection and Edge Generation Tasks}")
    latex_lines.append("\\label{tab:combined_results}")
    # 表格列定义：Model, DST部分(4列), Edge部分(4列), TDGG Score
    # latex_lines.append("\\begin{tabular}{lrrrrrrrrr}")
    latex_lines.append("\\begin{tabular}{l|rrrrr|rrrrr|rr}")
    latex_lines.append("\\toprule")
    # latex_lines.append("Model & \\multicolumn{4}{c}{DST Selection (Recall@100)} & \\multicolumn{4}{c}{Edge Generation} & \\multicolumn{1}{c}{TDGG} \\\\")
    latex_lines.append("Model & \\multicolumn{5}{c|}{DST Selection (Recall@100)} & \\multicolumn{5}{c|}{Edge Generation} & \\multicolumn{2}{c}{TDGG} \\\\")
    latex_lines.append(r"& Easy $\uparrow$ & Hard $\uparrow$ & All $\uparrow$ & Score $\uparrow$ & Rank $\downarrow$ & Acc $\uparrow$ & ROUGE-L $\uparrow$ & BERTScore $\uparrow$ & Score $\uparrow$ & Rank $\downarrow$ & Score $\uparrow$ & Rank $\downarrow$ \\")
    latex_lines.append("\\midrule")
    
    # 按数据集分组处理
    all_datasets = sorted(set(list(df['dataset'].unique())))

    first_dataset = True
    for dataset in all_datasets:
        # 格式化数据集名称
        formatted_dataset = dataset.replace('_', ' ').title()
        
        # 获取该数据集的DST和Edge数据
        dataset_df = df[df['dataset'] == dataset].copy() if dataset in df['dataset'].values else pd.DataFrame()
        
            
        # 根据模型顺序重新排列数据
        ordered_data = []
        for model_name in model_order:
            model_data = dataset_df[dataset_df['model'] == model_name]
            if not model_data.empty:
                ordered_data.append(model_data)
        
        # 合并排序后的数据
        if ordered_data:
            sorted_combined_df = pd.concat(ordered_data, ignore_index=True)
        else:
            sorted_combined_df = dataset_df.reset_index(drop=True)
            
        
        # 为每个数值列计算排名
        for col in dst_columns:
            col_name = col
            if "rank_score" in col_name:
                sorted_combined_df[f'{col}_rank'] = sorted_combined_df[col_name].rank(method='max', ascending=True)
            elif col_name in sorted_combined_df.columns:
                sorted_combined_df[f'{col}_rank'] = sorted_combined_df[col_name].rank(method='min', ascending=False)
        
        for col in edge_columns:
            col_name = col
            if "rank_score" in col_name:
                sorted_combined_df[f'{col}_rank'] = sorted_combined_df[col_name].rank(method='max', ascending=True)
            elif col_name in sorted_combined_df.columns:
                # Edge任务中的Acc, ROUGE-L, BERTScore, Score都是越大越好
                sorted_combined_df[f'{col}_rank'] = sorted_combined_df[col_name].rank(method='min', ascending=False)
        
        # 为TDGG Score计算排名
        sorted_combined_df['tdgg_social_fidelity_score_rank'] = sorted_combined_df['tdgg_social_fidelity_score'].rank(method='min', ascending=False)
        
        sorted_combined_df['tdgg_social_fidelity_rank_score_rank'] = sorted_combined_df['tdgg_social_fidelity_rank_score'].rank(method='max', ascending=True)
        
        # 添加数据集标题行（除了第一个数据集前不需要添加 \midrule）
        if not first_dataset:
            latex_lines.append("\\midrule")
        latex_lines.append("\\addlinespace")
        # latex_lines.append(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{formatted_dataset}}}}} \\\\")
        latex_lines.append(f"\\multicolumn{{13}}{{l}}{{\\textbf{{{formatted_dataset}}}}} \\\\")
        latex_lines.append("\\midrule")
        
        # 添加数据行
        for idx, row in sorted_combined_df.iterrows():
            line = f" {row['model']}"  # 不再添加数据集名称
            
            # 添加DST列数据
            for col in dst_columns:
                col_name = col
                # 处理可能存在的列名后缀
                if col_name not in row.index and f"{col}_dst" in row.index:
                    col_name = f"{col}_dst"
                    
                value = row.get(col_name, np.nan) if col_name in row.index else np.nan
                if isinstance(value, (int, float)) and not pd.isna(value):
                    formatted_value = f"{value:.4f}"
                    # 检查是否需要加粗或下划线
                    rank_col = f'{col}_rank'
                    if rank_col in sorted_combined_df.columns:
                        rank = row[rank_col]
                        if rank == 1:
                            # top1 加粗
                            formatted_value = f"\\textbf{{{formatted_value}}}"
                        elif rank == 2:
                            # top2 下划线
                            formatted_value = f"\\underline{{{formatted_value}}}"
                    line += f" & {formatted_value}"
                else:
                    line += f" & -"
            
            # 添加Edge列数据
            for col in edge_columns:
                col_name = col
                # 处理可能存在的列名后缀
                if col_name not in row.index and f"{col}_edge" in row.index:
                    col_name = f"{col}_edge"
                    
                value = row.get(col_name, np.nan) if col_name in row.index else np.nan
                if isinstance(value, (int, float)) and not pd.isna(value):
                    formatted_value = f"{value:.4f}"
                    # 检查是否需要加粗或下划线
                    rank_col = f'{col}_rank'
                    if rank_col in sorted_combined_df.columns:
                        rank = row[rank_col]
                        if rank == 1:
                            # top1 加粗
                            formatted_value = f"\\textbf{{{formatted_value}}}"
                        elif rank == 2:
                            # top2 下划线
                            formatted_value = f"\\underline{{{formatted_value}}}"
                    line += f" & {formatted_value}"
                else:
                    line += f" & -"
            
            # 添加TDGG Score列数据
            tdgg_score = row.get('tdgg_social_fidelity_score', np.nan)
            if isinstance(tdgg_score, (int, float)) and not pd.isna(tdgg_score):
                formatted_value = f"{tdgg_score:.4f}"
                # 检查是否需要加粗或下划线
                rank = row['tdgg_social_fidelity_score_rank']
                if rank == 1:
                    # top1 加粗
                    formatted_value = f"\\textbf{{{formatted_value}}}"
                elif rank == 2:
                    # top2 下划线
                    formatted_value = f"\\underline{{{formatted_value}}}"
                line += f" & {formatted_value}"
            else:
                line += f" & -"

            #  添加TDGG Rank列数据
            tdgg_score = row.get('tdgg_social_fidelity_rank_score', np.nan)
            if isinstance(tdgg_score, (int, float)) and not pd.isna(tdgg_score):
                formatted_value = f"{int(tdgg_score)}"
                # 检查是否需要加粗或下划线
                rank = row['tdgg_social_fidelity_rank_score_rank']
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
        for col in dst_columns + edge_columns:
            rank_col = f'{col}_rank'
            if rank_col in sorted_combined_df.columns:
                sorted_combined_df.drop(rank_col, axis=1, inplace=True)
        
        first_dataset = False
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # 保存到文件
    latex_table = "\n".join(latex_lines)
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Combined表格已保存至: {output_path}")
    return latex_table

def process_combined_for_latex(
    dst_input_file_path="LLMGGen/reports/tdgg_social_fidelity_scores.csv",
    edge_input_file_path="LLMGGen/reports/concat/merged_edge_matrix.csv",
    combined_output_path="LLMGGen/reports/latex_combined_table.tex"
):
    """
    主函数：处理两个输入文件并生成合并的LaTeX表格
    """
    # 读取数据
    df = pd.read_csv(dst_input_file_path)
    
    # 确保输出目录存在
    output_dir = Path(combined_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成合并表格
    combined_latex = generate_combined_table(df, combined_output_path)
    
    print("✅ 合并LaTeX表格生成完成!")
    return combined_latex

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成TDGG合并评估结果的LaTeX表格")
    parser.add_argument("--dst_input", type=str,
                        default="LLMGGen/reports/tdgg_social_fidelity_scores.csv",
                        help="DST任务输入的CSV文件路径")
    parser.add_argument("--edge_input", type=str,
                        default="LLMGGen/reports/concat/merged_edge_matrix.csv",
                        help="Edge任务输入的CSV文件路径")
    parser.add_argument("--output", type=str,
                        default="LLMGGen/reports/latex_combined_table.tex",
                        help="合并表格输出路径")
    
    args = parser.parse_args()
    
    # 执行处理
    process_combined_for_latex(
        dst_input_file_path=args.dst_input,
        edge_input_file_path=args.edge_input,
        combined_output_path=args.output
    )