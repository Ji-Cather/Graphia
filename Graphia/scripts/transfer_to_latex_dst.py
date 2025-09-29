import pandas as pd
import numpy as np
import argparse
import re
def rename_edge_model(model_name):
    if re.match(r'grpo_.*sotopia_edge.*', model_name):
        return 'Graphia-seq'
    elif model_name.startswith('grpo_'):
        return 'Graphia'
    return model_name

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
def df_to_latex_grouped_by_dataset(df,
                                   group_col='Group',
                                   dataset_col='dataset',
                                   model_col='model',
                                   exclude_cols=None,
                                   metric_order=None,
                                   caption="Performance",
                                   label="tab:results"):
    """
    Generate ONE LaTeX table for a single Group (e.g., Easy), with:
      - Rows grouped by dataset (using \multirow)
      - Model in second column
      - Metrics with Top-1/Top-2 highlighting within each dataset
    """
    if exclude_cols is None:
        exclude_cols = [group_col, dataset_col, model_col, 'split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub',
                        'format_rate', 'model','dataset']

    # Clean data
    df.columns = df.columns.str.strip()
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
       ,
        'weibo_daily':'Weibo Daily',
        'weibo_tech':'Weibo Tech',
    }
    df[dataset_col] = df[dataset_col].replace(dataset_rename_map)
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
    df['model'] = df['model'].apply(rename_edge_model)
    model_order = [
        # 'DGGen',
        # 'DYMOND',
        # 'Tigger',
        # 'GAG-general',
        'Qwen3-8B'
        'Qwen3-8B-sft',
        'Graphia-seq',
        'Graphia'
    ]
    df = df[df['model'].isin(model_order)]
    for col in [dataset_col, model_col]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Select and order metrics
    metric_cols = [col for col in df.columns if col not in exclude_cols]
    
    if metric_order:
        ordered_metrics = [m for m in metric_order if m in metric_cols]
        remaining = [m for m in metric_cols if m not in metric_order]
        final_metric_cols = ordered_metrics + remaining
    else:
        final_metric_cols = metric_cols

    # 为每个模型分配排序权重
    def model_sort_key(model):
        try:
            return model_order.index(model)
        except ValueError:
            # 如果模型不在预定义顺序中，放在最后
            return len(model_order)
    
    # 创建用于排序的临时列
    df['_model_sort_key'] = df[model_col].apply(model_sort_key)
    
    # 先按数据集分组，然后在每个数据集内按模型顺序排序
    df = df.sort_values(by=[dataset_col, '_model_sort_key']).reset_index(drop=True)
    
    # 为Top-K计算创建排序后的数据副本
    df_for_ranking = df.copy()
    df_for_ranking = df_for_ranking.sort_values(by=[dataset_col, '_model_sort_key']).reset_index(drop=True)
    
    # 删除临时排序列
    df = df.drop('_model_sort_key', axis=1)
    df_for_ranking = df_for_ranking.drop('_model_sort_key', axis=1)

    latex_lines = []
    latex_lines.append(r"\begin{table*}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \small")
    n_cols = 2 + len(final_metric_cols)  # dataset + model + metrics
    latex_lines.append(r"  \begin{adjustbox}{width=\textwidth, totalheight=\textheight, keepaspectratio}")
    latex_lines.append(r"  \begin{tabular}{l|" + "c" * (n_cols - 1) + r"}")
    latex_lines.append(r"    \toprule")

    header = f"Dataset & Model" + "".join(f" & {col}" for col in final_metric_cols)
    latex_lines.append(f"    {header} \\\\")
    latex_lines.append(r"    \midrule")

    last_dataset = None
    dataset_rows = []  # Each: [model_name, val1_str, val2_str, ...]

    def flush_block():
        nonlocal last_dataset, dataset_rows
        if not dataset_rows:
            return
        n_models = len(dataset_rows)
        for i, row in enumerate(dataset_rows):
            if i == 0:
                cells = [f"\\multirow{{{n_models}}}{{*}}{{{last_dataset}}}", row[0]] + row[1:]
            else:
                cells = ["", row[0]] + row[1:]
            latex_lines.append("    " + " & ".join(cells) + r" \\")
        latex_lines.append(r"    \midrule")
        dataset_rows = []

    # 按照正确排序的数据逐行处理
    for _, row in df.iterrows():
        current_dataset = row[dataset_col]
        model_name = row[model_col]

        if last_dataset is not None and current_dataset != last_dataset:
            flush_block()

        # Format metric values
        formatted_vals = [model_name]
        # 获取当前数据集的所有数据，用于Top-K计算
        dataset_df = df_for_ranking[df_for_ranking[dataset_col] == current_dataset]
          
        for col in final_metric_cols:
            cell_val = row[col]
            try:
                num_val = float(cell_val)
                col_vals = pd.to_numeric(dataset_df[col], errors='coerce').dropna()
                if len(col_vals) == 0:
                    val_str = str(cell_val)
                else:
                    sorted_vals = col_vals.sort_values(ascending=False)
                    top1 = sorted_vals.iloc[0]
                    top2 = sorted_vals.iloc[1] if len(sorted_vals) > 1 else top1

                    val_str = f"{num_val:.4f}".rstrip('0').rstrip('.')
                    if abs(num_val - top1) < 1e-6:
                        val_str = f"\\textbf{{{val_str}}}"
                    elif abs(num_val - top2) < 1e-6:
                        val_str = f"\\underline{{{val_str}}}"
            except (ValueError, TypeError):
                val_str = str(cell_val)
            formatted_vals.append(val_str)

        dataset_rows.append(formatted_vals)
        last_dataset = current_dataset

    # Flush last block
    flush_block()

    # Remove last \midrule if exists
    if latex_lines[-1] == r"    \midrule":
        latex_lines.pop()

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"  \end{adjustbox}")
    latex_lines.append(f"  \\caption{{{caption}}}")
    latex_lines.append(f"  \\label{{{label}}}")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separate LaTeX tables for each Group, grouped by dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file (e.g., dst_retrival_matrix.csv)")
    parser.add_argument("--metrics", type=str, nargs="+", help="Optional: specify custom metric order")

    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.input)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    if "Group" not in df.columns:
        raise ValueError("Column 'Group' not found in CSV.")

    # Output path
    output_path = args.input.replace(".csv", "_grouped_by_dataset.txt")

    # Define groups
    groups = ["Easy", "Hard", "All"]
    valid_groups = df["Group"].dropna().unique()

    full_output = []

    for group in groups:
        if group not in valid_groups:
            print(f"⚠️ Group '{group}' not found. Skipping.")
            continue

        group_df = df[df["Group"] == group].copy()

        caption = f"Retrieval Performance on {group} Nodes"
        label = f"tab:retrieval_{group.lower()}"

        table = df_to_latex_grouped_by_dataset(
            group_df,
            group_col='Group',
            dataset_col='dataset',
            model_col='model',
            # exclude_cols=['split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub'],
            metric_order=args.metrics,
            caption=caption,
            label=label
        )
        full_output.append(table)

    # Save all tables
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(full_output))

    print(f"✅ LaTeX tables saved to: {output_path}")
