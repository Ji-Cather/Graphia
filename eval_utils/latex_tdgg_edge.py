import pandas as pd
import numpy as np
import argparse
import re

# 对 edge_df 应用重命名规则
def rename_edge_model(model_name):
    if re.match(r'grpo_.*_sotopia_edge_.*', model_name):
        return 'LLMGGen-seq'
    elif model_name.startswith('grpo_'):
        return 'LLMGGen'
    return model_name
def df_to_latex_multidataset(df, 
                             group_col='dataset', 
                             model_col='model',
                             exclude_cols=None,
                             metric_order=None,
                             caption="Performance by Dataset", 
                             label="tab:results",
                             negative_edge_metrics = []):
    if exclude_cols is None:
        exclude_cols = [model_col, group_col, 'split', 'task', "Unnamed: 0"]
    
    df.columns = df.columns.str.strip()  # 清理列名
    df = df.rename(columns={
        "average":"LLM Rating",
        "label_acc":"Acc Category",
        "ROUGE_L":"ROUGE-L",
        "BERTScore_F1":"BERTScore"
    })
    df['model'] = df['model'].apply(rename_edge_model)
    df[model_col] = df[model_col].fillna("Unknown").astype(str).str.strip()
    df[group_col] = df[group_col].fillna("Unknown_Dataset").astype(str).str.strip()

    # 添加模型重命名逻辑（类似 latex_tdgg_dst.py）
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft',
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    

    df = df.copy()
    df[model_col] = df[model_col].replace(model_rename_map)
    
    # 添加数据集重命名逻辑
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
        'propagate_large_cn': 'Propagate-Zh',
        'weibo_tech':'Weibo Tech',
        'weibo_daily':'Weibo Daily',
        'imdb':'Imdb',
    }
    df[group_col] = df[group_col].replace(dataset_rename_map)

    metric_cols = [col for col in df.columns if col not in exclude_cols]
    
    if metric_order:
        ordered_metrics = [m for m in metric_order if m in metric_cols]
        remaining = [m for m in metric_cols if m not in metric_order]
        final_metric_cols = ordered_metrics
    else:
        final_metric_cols = metric_cols

    # 添加模型排序逻辑（类似 latex_tdgg_dst.py）
    model_order = [

        'Qwen3-32B',
        'DeepSeek-Q-32B',
        'Llama3-70B',
        'LLMGGen-seq',
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'LLMGGen'
    ]
    
    # 按数据集分组处理并应用模型排序
    sorted_rows = []
    for dataset in df[group_col].unique():
        dataset_df = df[df[group_col] == dataset].copy()
        
        # 根据模型顺序重新排列数据
        ordered_data = []
        for model_name in model_order:
            model_data = dataset_df[dataset_df[model_col] == model_name]
            if not model_data.empty:
                ordered_data.append(model_data)
        
        # 合并排序后的数据
        if ordered_data:
            sorted_dataset_df = pd.concat(ordered_data, ignore_index=True)
        else:
            sorted_dataset_df = dataset_df.reset_index(drop=True)
            
        sorted_rows.append(sorted_dataset_df)
    
    # 合并所有排序后的数据
    if sorted_rows:
        df = pd.concat(sorted_rows, ignore_index=True)
    else:
        df = df.sort_values(by=[group_col, model_col]).reset_index(drop=True)

    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \small")
    latex_lines.append(r"  \begin{tabular}{l" + "c" * (1 + len(final_metric_cols)) + r"}")  # +1 for model
    latex_lines.append(r"    \toprule")
    
    header = f"{group_col.title()} & {model_col}" + "".join(f" & {col}" for col in final_metric_cols)
    latex_lines.append(f"    {header} \\\\")
    latex_lines.append(r"    \midrule")

    last_dataset = None
    dataset_group_rows = []
    current_block_model_vals = []

    def add_multirow_block():
        if not dataset_group_rows:
            return
        n_models = len(dataset_group_rows)
        for i, row in enumerate(dataset_group_rows):
            
            if i == 0:
                # last_dataset = last_dataset.replace("_"," ").title()
                cells = [f"\\multirow{{{n_models}}}{{*}}{{{last_dataset}}}", row[0]] + row[1:]
            else:
                cells = ["", row[0]] + row[1:]
            latex_lines.append("    " + " & ".join(cells) + r" \\")
        latex_lines.append(r"    \midrule")

    for idx, row in df.iterrows():
        current_dataset = row[group_col]
        model_name = row[model_col]

        if current_dataset != last_dataset and last_dataset is not None:
            add_multirow_block()
            dataset_group_rows = []

        # Format metrics
        values = [model_name]  # ← model 是单独一列
        dataset_df = df[df[group_col] == current_dataset]
        for col in final_metric_cols:
            cell_val = row[col]
            try:
                num_val = float(cell_val)
                col_vals = pd.to_numeric(dataset_df[col], errors='coerce').dropna()
                if len(col_vals) > 0:
                    if col in negative_edge_metrics:
                        sorted_vals = col_vals.sort_values(ascending=True)
                    else:
                        sorted_vals = col_vals.sort_values(ascending=False)
                    top1_val = sorted_vals.iloc[0]
                    top2_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else top1_val

                    val_str = f"{num_val:.3f}".rstrip('0').rstrip('.')
                    if abs(num_val - top1_val) < 1e-6:
                        val_str = f"\\textbf{{{val_str}}}"
                    elif abs(num_val - top2_val) < 1e-6:
                        val_str = f"\\underline{{{val_str}}}"
                else:
                    val_str = str(cell_val)
            except:
                val_str = str(cell_val)
            values.append(val_str)

        dataset_group_rows.append(values)
        last_dataset = current_dataset

    if dataset_group_rows:
        add_multirow_block()
        if latex_lines[-1] == r"    \midrule":
            latex_lines.pop()

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(f"  \\caption{{{caption}}}")
    latex_lines.append(f"  \\label{{{label}}}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table with multirow datasets and custom metric order.")
    parser.add_argument("--input", type=str, default="LLMGGen/reports/concat/merged_edge_matrix.csv", help="Path to input CSV file")
    parser.add_argument("--output", type=str, help="Output .txt file path")
    parser.add_argument("--caption", type=str, default="Performance by Dataset", help="Table caption")
    parser.add_argument("--label", type=str, default="tab:results", help="Table label")
    parser.add_argument("--metrics", type=str, nargs="+", help="Optional: specify metric order, e.g., --metrics precision@20%N_hub auc@20%N_hub f1@20%N_hub")

    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.input)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Output path
    output_path = args.input.replace(".csv", ".tex")

    # graph_metrics = [
    #     "precision@20%N_hub","auc@20%N_hub","edge_overlap","GraphEmbedding-Metric","wedge_count","triangle_count","clustering_coefficient","d_mean",
    # ]

    # negative_edge_metrics = [
    #     "wedge_count","triangle_count","clustering_coefficient","d_mean",
    # ]

    # # Generate LaTeX
    # latex_code = df_to_latex_multidataset(
    #     df,
    #     group_col='dataset',
    #     model_col='model',
    #     metric_order=graph_metrics,
    #     caption=args.caption,
    #     label=args.label,
    #     negative_edge_metrics = negative_edge_metrics
    # )


    edge_all_metrics = [
        "GF","CF","PD","DA","IQ","CR","average"
    ]
    edge_part_metrics = [
        "LLM Rating",
        "Acc Category",
        "ROUGE-L",
       "BERTScore"
    ]
    df["average"] = np.array(df["average"].values)*5
    # Generate LaTeX
    latex_code = df_to_latex_multidataset(
        df,
        group_col='dataset',
        model_col='model',
        metric_order= edge_part_metrics,
        caption=args.caption,
        label=args.label,
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    print(f"✅ Multi-dataset LaTeX table saved to: {output_path}")
