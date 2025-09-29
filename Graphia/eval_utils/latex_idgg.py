import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def df_to_latex_idgg(df, 
                     group_col='dataset', 
                     model_col='model',
                     caption="IDGG Social Fidelity Scores by Dataset", 
                     label="tab:idgg_results"):
    """
    将IDGG评估结果转换为LaTeX表格格式，按数据集分组，每组包含三大列：
    1. Macro Structure指标（包含多个子指标）+ macro_structure_score + macro_structure_rank_score
    2. Macro Phenomenon指标（包含多个子指标）+ macro_phenomenon_score + macro_phenomenon_rank_score
    3. IDGG Social Fidelity指标（idgg_social_fidelity_score）+ idgg_rank_score
    """
    
    # 清理列名和数据
    df.columns = df.columns.str.strip()
    df[model_col] = df[model_col].fillna("Unknown").astype(str).str.strip()
    df[group_col] = df[group_col].fillna("Unknown_Dataset").astype(str).str.strip()
    
    # 参考plot_idgg_scores.py中的重命名规则
    # 重命名模型
    model_rename_map = {
        'qwen3_sft': 'Qwen3-8B-sft',
        'DGGen': 'DGGen',
        'DYMOND': 'DYMOND',
        'tigger': 'Tigger',
        'idgg_csv_processed': 'GAG-general',
        'qwen3': 'Qwen3-8B',
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    df[model_col] = df[model_col].replace(model_rename_map)
    
    # 重命名数据集
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
       
    }
    df[group_col] = df[group_col].replace(dataset_rename_map)
    
    # 格式化数据集名称（参考plot_idgg_scores.py#L504-L508）
    def format_dataset_name(name):
        return name.replace('_', ' ').title()
    
    # 定义负向指标（需要反向）和正向指标
    # 参考eval_idgg.py#L16-L19
    negative_metrics = [
        'graph_list_degree_mmd', 'graph_list_cluster_mmd', 'graph_list_spectra_mmd',
        'graph_macro_num_chambers_diff',  'graph_macro_alpha_gap',
        'macro_structure_rank_score', 'macro_phenomenon_rank_score', 'idgg_social_fidelity_rank_score'
    ]
    
    # 参考eval_idgg.py#L22-L24
    positive_metrics = [
        'graph_edge_overlap', 'graph_macro_precision@100pagerank-hub'
    ]
    
    # 定义三大列的指标
    # 参考eval_idgg.py#L84-L87
    macro_structure_metrics = [
        'graph_list_degree_mmd', 
        'graph_list_cluster_mmd', 
        'graph_list_spectra_mmd',
        'graph_edge_overlap',
        'macro_structure_score',
        'macro_structure_rank_score'
    ]
    
    # 参考eval_idgg.py#L90-L92
    macro_phenomenon_metrics = [
        'graph_macro_precision@100pagerank-hub', 
        'graph_macro_num_chambers_diff',
        'graph_macro_alpha_gap',
        'macro_phenomenon_score',
        'macro_phenomenon_rank_score'
    ]
    
    idgg_metrics = ['idgg_social_fidelity_score','idgg_social_fidelity_rank_score']
    
    # 所有需要的指标
    all_metrics = macro_structure_metrics + macro_phenomenon_metrics + idgg_metrics
    
    # 检查所有指标是否存在于数据中
    missing_metrics = [m for m in all_metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metrics in data: {missing_metrics}")
    
    # 参考plot_idgg_scores.py中的模型顺序（L245-L254）
    model_order = [
        'Qwen3-8B',
        'Qwen3-8B-sft', 
        'Qwen3-32B',
        'DeepSeek-Q-32B',
        'Llama3-70B',
        'Graphia-seq',
        'Graphia'
    ]

    # 按数据集和模型排序
    def sort_models_by_order(df, model_col, order):
        # 为每个模型分配排序权重
        def model_sort_key(model):
            try:
                return order.index(model)
            except ValueError:
                # 如果模型不在预定义顺序中，放在最后
                return len(order)
        
        # 添加排序列
        df['_model_sort_key'] = df[model_col].apply(model_sort_key)
        # 按照排序键排序
        df_sorted = df.sort_values(by=['_model_sort_key']).drop('_model_sort_key', axis=1)
        return df_sorted.reset_index(drop=True)
    
    # 先按数据集分组，然后在每个数据集中按模型顺序排序
    sorted_dfs = []
    for dataset in df[group_col].unique():
        dataset_df = df[df[group_col] == dataset].copy()
        sorted_dataset_df = sort_models_by_order(dataset_df, model_col, model_order)
        sorted_dfs.append(sorted_dataset_df)
    
    # 合并所有排序后的数据
    df = pd.concat(sorted_dfs, ignore_index=True)
    

    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \begin{adjustbox}{width=\textwidth, totalheight=\textheight, keepaspectratio}")
    # latex_lines.append(r"  \begin{tabular}{l|ccccc|cccc|cc}")
    latex_lines.append(r"  \begin{tabular}{l|cccccc|ccccc|cc}")
    latex_lines.append(r"    \toprule")
    
    # 表头
    # header = f"Model & \\multicolumn{{5}}{{c|}}{{Macro Structure}} & \\multicolumn{{4}}{{c|}}{{Macro Phenomenon}} & \\multicolumn{{2}}{{c}}{{IDGG}}"
    header = f"Model & \\multicolumn{{6}}{{c|}}{{Macro Structure}} & \\multicolumn{{5}}{{c|}}{{Macro Phenomenon}} & \\multicolumn{{2}}{{c}}{{IDGG}}"
    latex_lines.append(f"    {header} \\\\")
    
    # 子表头，添加方向箭头符号
    subheader_parts = [""]
    # Macro Structure 子指标
    subheader_parts.append(r"MMD.D $\downarrow$")
    subheader_parts.append(r"MMD.C $\downarrow$")
    subheader_parts.append(r"MMD.S $\downarrow$")
    subheader_parts.append(r"Edge Overlap $\uparrow$")
    subheader_parts.append(r"Score $\uparrow$")
    subheader_parts.append(r"Rank $\downarrow$")
    
    # Macro Phenomenon 子指标
    # subheader_parts.append(r"D $\downarrow$")
    subheader_parts.append(r"P@100Hub $\uparrow$")
    subheader_parts.append(r"Chambers Diff $\downarrow$")
    subheader_parts.append(r"$\Delta \alpha$ $\downarrow$")
    subheader_parts.append(r"Score $\uparrow$")
    subheader_parts.append(r"Rank $\downarrow$")
    
    # IDGG 指标
    subheader_parts.append(r"Score $\uparrow$")
    subheader_parts.append(r"Rank $\downarrow$")
    
    subheader = " & ".join(subheader_parts)
    latex_lines.append(f"    {subheader} \\\\")
    latex_lines.append(r"    \midrule")

    # 按数据集分组处理
    for dataset in df[group_col].unique():
        dataset_df = df[df[group_col] == dataset].copy()
        # 再次确保数据集内模型按顺序排列
        dataset_df = sort_models_by_order(dataset_df, model_col, model_order)
        
        # 添加数据集标题行（使用multicolumn）
        if len(latex_lines) > 7:  # 如果不是第一个数据集，添加额外的间距
            latex_lines.append(r"    \addlinespace")
        
        # 格式化数据集名称（参考plot_idgg_scores.py#L504-L508）
        formatted_dataset_name = format_dataset_name(dataset)
        latex_lines.append(f"    \\multicolumn{{14}}{{c}}{{\\textbf{{{formatted_dataset_name}}}}} \\\\")
        latex_lines.append(r"    \midrule")
        
        # 为每个模型添加一行，按照指定顺序
        for _, row in dataset_df.iterrows():
            model_name = row[model_col]
            
            # 获取并格式化各项指标值
            values = [model_name]
            
            # 处理 Macro Structure 指标
            for i, metric in enumerate(macro_structure_metrics):
                try:
                    val = float(row[metric])
                    # 格式化为4位小数，去掉末尾的0
                    val_str = f"{val:.4f}"
                    
                    # 查找该指标在当前数据集中的最大值和次大值
                    col_vals = pd.to_numeric(dataset_df[metric], errors='coerce').dropna()
                    if len(col_vals) > 0:
                        # 对于负向指标，数值越小越好，所以按升序排列
                        if metric in negative_metrics or "rank_score" in metric:
                            sorted_vals = col_vals.sort_values(ascending=True)
                        else:
                            # 对于正向指标，数值越大越好，所以按降序排列
                            sorted_vals = col_vals.sort_values(ascending=False)
                            
                        top1_val = sorted_vals.iloc[0]
                        top2_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else top1_val
                        
                        # 如果是最佳值，加粗显示
                        if abs(val - top1_val) < 1e-6:
                            val_str = f"\\textbf{{{val_str}}}"
                        # 如果是次佳值，加下划线显示
                        elif abs(val - top2_val) < 1e-6:
                            val_str = f"\\underline{{{val_str}}}"
                except:
                    val_str = str(row[metric])
                
                values.append(val_str)
            
           
            
            # 处理 Macro Phenomenon 指标
            for i, metric in enumerate(macro_phenomenon_metrics):
                try:
                    val = float(row[metric])
                    # 格式化为4位小数，去掉末尾的0
                    val_str = f"{val:.4f}"
                    
                    # 查找该指标在当前数据集中的最大值和次大值
                    col_vals = pd.to_numeric(dataset_df[metric], errors='coerce').dropna()
                    if len(col_vals) > 0:
                        # 对于负向指标，数值越小越好，所以按升序排列
                        if metric in negative_metrics or "rank_score" in metric:
                            sorted_vals = col_vals.sort_values(ascending=True)
                        else:
                            # 对于正向指标，数值越大越好，所以按降序排列
                            sorted_vals = col_vals.sort_values(ascending=False)
                            
                        top1_val = sorted_vals.iloc[0]
                        top2_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else top1_val
                        
                        # 如果是最佳值，加粗显示
                        if abs(val - top1_val) < 1e-6:
                            val_str = f"\\textbf{{{val_str}}}"
                        # 如果是次佳值，加下划线显示
                        elif abs(val - top2_val) < 1e-6:
                            val_str = f"\\underline{{{val_str}}}"
                except:
                    val_str = str(row[metric])
                
                values.append(val_str)
            
            
            
            # 处理 IDGG Social Fidelity 指标
            for metric in idgg_metrics:
                try:
                    val = float(row[metric])
                    # 格式化为4位小数，去掉末尾的0
                    val_str = f"{val:.4f}".rstrip('0').rstrip('.')
                    
                    # 查找该指标在当前数据集中的最大值和次大值
                    col_vals = pd.to_numeric(dataset_df[metric], errors='coerce').dropna()
                    if len(col_vals) > 0:
                       # 对于负向指标，数值越小越好，所以按升序排列
                        if metric in negative_metrics or "rank" in metric:
                            sorted_vals = col_vals.sort_values(ascending=True)
                        else:
                            # 对于正向指标，数值越大越好，所以按降序排列
                            sorted_vals = col_vals.sort_values(ascending=False)
                        top1_val = sorted_vals.iloc[0]
                        top2_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else top1_val
                        
                        # 如果是最佳值，加粗显示
                        if abs(val - top1_val) < 1e-6:
                            val_str = f"\\textbf{{{val_str}}}"
                        # 如果是次佳值，加下划线显示
                        elif abs(val - top2_val) < 1e-6:
                            val_str = f"\\underline{{{val_str}}}"
                except:
                    val_str = str(row[metric])
                
                values.append(val_str)
            
            
            
            # 添加模型行
            row_str = "    " + " & ".join(values) + r" \\"
            latex_lines.append(row_str)
        
        # 数据集结束添加中等横线
        latex_lines.append(r"    \midrule")
    
    # 移除最后的midrule
    if latex_lines[-1] == r"    \midrule":
        latex_lines.pop()

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"  \end{adjustbox}")
    latex_lines.append(f"  \\caption{{{caption}}}")
    latex_lines.append(f"  \\label{{{label}}}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table for IDGG Social Fidelity Scores")
    parser.add_argument("--input", type=str, default="Graphia/reports/idgg_social_fidelity_scores.csv", help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="Graphia/reports/idgg_social_fidelity_scores.tex",  help="Output .tex file path")
    parser.add_argument("--caption", type=str, default="IDGG Social Fidelity Scores by Dataset", 
                       help="Table caption")
    parser.add_argument("--label", type=str, default="tab:idgg_results", help="Table label")

    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.input)
    
    # 生成LaTeX代码
    latex_code = df_to_latex_idgg(
        df,
        group_col='dataset',
        model_col='model',
        caption=args.caption,
        label=args.label
    )

    # 确定输出路径
    output_path = args.output
   
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    print(f"✅ IDGG LaTeX table saved to: {output_path}")

if __name__ == "__main__":
    main()