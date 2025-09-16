import pandas as pd
import numpy as np
import argparse

def df_to_latex_grouped_by_dataset(df,
                                   group_col='Group',
                                   dataset_col='dataset',
                                   model_col='model',
                                   exclude_cols=None,
                                   metric_order=None,
                                   caption="Performance",
                                   label="tab:results"):
    """
    Generate ONE LaTeX table for a single Group (e.g., Hub), with:
      - Rows grouped by dataset (using \multirow)
      - Model in second column
      - Metrics with Top-1/Top-2 highlighting within each dataset
    """
    if exclude_cols is None:
        exclude_cols = [group_col, dataset_col, model_col, 'split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub',]

    # Clean data
    df.columns = df.columns.str.strip()
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

    # Sort
    df = df.sort_values(by=[dataset_col, model_col]).reset_index(drop=True)

    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \small")
    n_cols = 2 + len(final_metric_cols)  # dataset + model + metrics
    latex_lines.append(r"  \begin{tabular}{l" + "c" * (n_cols - 1) + r"}")
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

    for _, row in df.iterrows():
        current_dataset = row[dataset_col]
        model_name = row[model_col]

        if last_dataset is not None and current_dataset != last_dataset:
            flush_block()

        # Format metric values
        formatted_vals = [model_name]
        dataset_df = df[df[dataset_col] == current_dataset]
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
    latex_lines.append(f"  \\caption{{{caption}}}")
    latex_lines.append(f"  \\label{{{label}}}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separate LaTeX tables for each Group, grouped by dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file (e.g., dst_retrival_matrix.csv)")
    parser.add_argument("--output", type=str, help="Output .txt file path")
    parser.add_argument("--metrics", type=str, nargs="+", help="Optional: specify custom metric order")

    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.input)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    if "Group" not in df.columns:
        raise ValueError("Column 'Group' not found in CSV.")

    # Output path
    output_path = args.output or args.input.replace(".csv", "_grouped_by_dataset.txt")

    # Define groups
    groups = ["Hub", "Normal", "All"]
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
            exclude_cols=['split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub'],
            metric_order=args.metrics,
            caption=caption,
            label=label
        )
        full_output.append(table)

    # Save all tables
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(full_output))




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
    groups = ["Hub", "Normal", "All"]
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
            exclude_cols=['split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub'],
            metric_order=args.metrics,
            caption=caption,
            label=label
        )
        full_output.append(table)

    # Save all tables
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(full_output))

    print(f"✅ LaTeX tables saved to: {output_path}")
