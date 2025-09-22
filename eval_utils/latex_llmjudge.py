import pandas as pd
import numpy as np

def process_edge_matrix_data(file_path="LLMGGen/reports/concat/merged_edge_matrix.csv"):
    """
    处理edge matrix数据，重命名数据集和模型，并生成合并表格
    
    Parameters:
    file_path (str): merged_edge_matrix.csv 文件路径
    
    Returns:
    tuple: (combined_table, overall_table)
    """
    
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 选择需要的列
    columns_needed = ['model', 'dataset', 'GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']
    df = df[columns_needed]
    
    # 移除包含缺失值的行
    df = df.dropna()
    
    # 重命名数据集
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
        'propagate_large_cn': 'propagate_zh'
    }
    
    df['dataset'] = df['dataset'].replace(dataset_rename_map)
    
    # 重命名模型
    # 对于符合grpo*sotopia*格式的，重命名为LLMGGen-seq
    # 对于符合grpo*但不符合grpo*sotopia*的，重命名为LLMGGen
    def rename_model(model_name):
        model_map = {
            'qwen3': 'Qwen3-8b',
            'qwen3_sft': 'Qwen3-8b-sft',
        }
        if pd.isna(model_name):
            return model_name
        if 'grpo' in model_name.lower() and 'sotopia' in model_name.lower():
            return 'LLMGGen-seq'
        elif 'grpo' in model_name.lower():
            return 'LLMGGen'
        elif model_name in model_map:
            return model_map[model_name]
        else:
            return model_name
    
    df['model'] = df['model'].apply(rename_model)
    
    # 定义模型排序顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft',
        'LLMGGen-seq',
        'LLMGGen'
    ]
    
    # 按照指定顺序排序模型
    ordered_data = []
    for model_name in model_order:
        model_data = df[df['model'] == model_name]
        if not model_data.empty:
            ordered_data.append(model_data)
    
    # 合并排序后的数据
    if ordered_data:
        reordered_df = pd.concat(ordered_data, ignore_index=True)
    else:
        # 如果没有匹配的模型，按原始顺序排序
        reordered_df = df.sort_values('model').reset_index(drop=True)
    
    # 创建总表（按模型分组，计算各指标的平均值）
    overall_table = df.groupby('model')[['GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']].mean().reset_index()
    
    # 按照指定顺序排序模型
    ordered_data = []
    for model_name in model_order:
        model_data = overall_table[overall_table['model'] == model_name]
        if not model_data.empty:
            ordered_data.append(model_data)
    
    # 合并排序后的数据
    if ordered_data:
        reordered_overall = pd.concat(ordered_data, ignore_index=True)
    else:
        # 如果没有匹配的模型，按原始顺序排序
        reordered_overall = overall_table.sort_values('model').reset_index(drop=True)
    
    return reordered_df, reordered_overall

def add_rankings_to_table(df):
    """
    为表格添加排名信息，标识每个数据集中每个指标的第一名和第二名
    """
    # 复制数据框以避免修改原始数据
    ranked_df = df.copy()
    
    # 获取所有指标列（除了model和dataset）
    metric_columns = ['GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']
    
    # 为每个数据集和每个指标添加排名
    for dataset in ranked_df['dataset'].unique():
        dataset_mask = ranked_df['dataset'] == dataset
        for metric in metric_columns:
            # 按指标值降序排列（数值越大排名越前）
            ranked_df.loc[dataset_mask, f'{metric}_rank'] = \
                ranked_df.loc[dataset_mask, metric].rank(method='min', ascending=False)
    
    return ranked_df

def create_combined_latex_table_with_rankings(combined_table):
    """
    创建带排名的合并LaTeX表格，按数据集分组显示，第一名加粗，第二名下划线
    """
    # 添加排名信息
    ranked_table = add_rankings_to_table(combined_table)
    
    # 获取所有唯一的数据集
    datasets = ranked_table['dataset'].unique()
    
    # 创建表格头部
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Model Performance Across Different Datasets with Rankings (\\textbf{First Place}, \\underline{Second Place})}\n"
    latex_table += "\\begin{tabular}{lcccccccc}\n"
    latex_table += "\\hline\n"
    latex_table += "\\textbf{Dataset} & \\textbf{Model} & \\textbf{GF} & \\textbf{CF} & \\textbf{PD} & \\textbf{DA} & \\textbf{IQ} & \\textbf{CR} & \\textbf{Average} \\\\\n"
    latex_table += "\\hline\n"
    
    # 为每个数据集添加数据
    for dataset in datasets:
        dataset_data = ranked_table[ranked_table['dataset'] == dataset]
        dataset_name = dataset.replace('_', ' ').title()
        
        # 添加数据集名称作为行
        first_row = True
        for _, row in dataset_data.iterrows():
            if first_row:
                latex_table += f"\\multirow{{{len(dataset_data)}}}{{*}}{{{dataset_name}}} & {row['model']} & "
                first_row = False
            else:
                latex_table += f" & {row['model']} & "
            
            # 为每个指标添加带格式的值
            metric_columns = ['GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']
            for i, metric in enumerate(metric_columns):
                value = row[metric]
                rank = row[f'{metric}_rank']
                
                # 根据排名添加格式
                if rank == 1:
                    formatted_value = f"\\textbf{{{value:.3f}}}"
                elif rank == 2:
                    formatted_value = f"\\underline{{{value:.3f}}}"
                else:
                    formatted_value = f"{value:.3f}"
                
                # 添加值到表格
                if i == len(metric_columns) - 1:  # 最后一列
                    latex_table += f"{formatted_value} \\\\\n"
                else:
                    latex_table += f"{formatted_value} & "
        
        latex_table += "\\hline\n"
    
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    return latex_table

def print_results_with_rankings(combined_table, overall_table):
    """
    打印带排名的结果表格
    """
    # 添加排名信息
    ranked_table = add_rankings_to_table(combined_table)
    
    print("=== COMBINED TABLE WITH RANKINGS (BY DATASET) ===")
    # 获取所有唯一的数据集
    datasets = ranked_table['dataset'].unique()
    
    # 为每个数据集打印数据
    for dataset in datasets:
        print(f"\n{dataset.replace('_', ' ').title()} Dataset:")
        dataset_data = ranked_table[ranked_table['dataset'] == dataset]
        
        # 选择要显示的列
        display_columns = ['model', 'GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']
        display_data = dataset_data[display_columns].copy()
        
        # 为每个指标添加排名标识
        metric_columns = ['GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']
        for metric in metric_columns:
            ranks = dataset_data[f'{metric}_rank']
            display_data[metric] = display_data[metric].astype(str)
            for idx, rank in enumerate(ranks):
                if rank == 1:
                    display_data.iloc[idx, display_data.columns.get_loc(metric)] = f"**{display_data.iloc[idx, display_data.columns.get_loc(metric)]}**"
                elif rank == 2:
                    display_data.iloc[idx, display_data.columns.get_loc(metric)] = f"__{display_data.iloc[idx, display_data.columns.get_loc(metric)]}__"
        
        print(display_data.to_string(index=False))
        print("-" * 120)
    
    print("\n\n=== OVERALL TABLE (MEAN ACROSS ALL DATASETS) ===")
    print(overall_table.to_string(index=False, float_format="%.3f"))

# 使用示例
if __name__ == "__main__":
    # 处理数据
    combined_table, overall_table = process_edge_matrix_data()
    
    # 打印结果
    print_results_with_rankings(combined_table, overall_table)
    
    # 如果需要LaTeX格式
    print("\n\n=== LATEX FORMAT WITH RANKINGS ===")
    try:
        latex_table = create_combined_latex_table_with_rankings(combined_table)
        print(latex_table)
    except:
        print("Note: For the LaTeX table to work properly, you need to include \\usepackage{multirow} in your LaTeX preamble.")
        # 提供一个不使用multirow的简化版本
        print("\nSimplified LaTeX table (without multirow):")
        
        # 添加排名信息
        ranked_table = add_rankings_to_table(combined_table)
        
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\caption{Model Performance Across Different Datasets with Rankings (\\textbf{First Place}, \\underline{Second Place})}")
        print("\\begin{tabular}{lcccccccc}")
        print("\\hline")
        print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{GF} & \\textbf{CF} & \\textbf{PD} & \\textbf{DA} & \\textbf{IQ} & \\textbf{CR} & \\textbf{Average} \\\\")
        print("\\hline")
        
        datasets = ranked_table['dataset'].unique()
        for dataset in datasets:
            dataset_data = ranked_table[ranked_table['dataset'] == dataset]
            dataset_name = dataset.replace('_', ' ').title()
            
            for _, row in dataset_data.iterrows():
                print(f"{dataset_name} & {row['model']} & ", end="")
                
                # 为每个指标添加带格式的值
                metric_columns = ['GF', 'CF', 'PD', 'DA', 'IQ', 'CR', 'average']
                for i, metric in enumerate(metric_columns):
                    value = row[metric]
                    rank = row[f'{metric}_rank']
                    
                    # 根据排名添加格式
                    if rank == 1:
                        formatted_value = f"\\textbf{{{value:.3f}}}"
                    elif rank == 2:
                        formatted_value = f"\\underline{{{value:.3f}}}"
                    else:
                        formatted_value = f"{value:.3f}"
                    
                    # 添加值到表格
                    if i == len(metric_columns) - 1:  # 最后一列
                        print(f"{formatted_value} \\\\")
                    else:
                        print(f"{formatted_value} & ", end="")
            
            print("\\hline")
        
        print("\\end{tabular}")
        print("\\end{table}")