import pandas as pd
import numpy as np
import argparse


def df_to_latex_combined_groups(df,
                                dataset_col='dataset',
                                model_col='model',
                                group_col='Group',
                                exclude_cols=None,
                                metric_dict=None,
                                default_metrics=None,
                                caption="Retrieval Performance Across Groups",
                                label="tab:retrieval_combined"):
    """
    Generate a single LaTeX table with two-level headers:
      - Row 1: Group names (e.g., Hub, Normal, All), each spanning its own metrics
      - Row 2: Metrics specific to each group (customizable via metric_dict)
      - Rows grouped by dataset (multirow), one row per model

    :param df: Input DataFrame
    :param dataset_col: column name for dataset
    :param model_col: column name for model
    :param group_col: column name for group (e.g., Hub/Normal/All)
    :param exclude_cols: columns to exclude from metrics
    :param metric_dict: dict like {'Hub': ['hit@10', 'recall@10'], 'All': ['hit@50']} — per-group metrics
    :param default_metrics: fallback if metric_dict doesn't specify
    :param caption: table caption
    :param label: table label
    """
    if exclude_cols is None:
        exclude_cols = [group_col, dataset_col, model_col, 'split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub']

    # Clean data
    df.columns = df.columns.str.strip()
    for col in [dataset_col, model_col, group_col]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Default metrics
    all_metric_cols = [col for col in df.columns if col not in exclude_cols]
    if default_metrics is None:
        default_metrics = all_metric_cols
    else:
        default_metrics = [m for m in default_metrics if m in all_metric_cols]

    # Use metric_dict or assign default to all groups
    if metric_dict is None:
        metric_dict = {grp: default_metrics for grp in df[group_col].dropna().unique()}
    else:
        # Fill missing groups with default
        existing_groups = df[group_col].dropna().unique()
        for grp in existing_groups:
            if grp not in metric_dict:
                metric_dict[grp] = default_metrics

    # Only keep valid groups and metrics
    valid_groups = [g for g in ["Hub", "Normal", "All"] if g in metric_dict and g in df[group_col].dropna().values]
    if not valid_groups:
        raise ValueError("No valid groups found in metric_dict and data.")

    # Build column structure: [(group, metric), ...]
    col_structure = []
    for group in valid_groups:
        for metric in metric_dict[group]:
            if metric in all_metric_cols:
                col_structure.append((group, metric))

    n_cols = 2 + len(col_structure)  # Dataset + Model + metrics
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \small")
    latex_lines.append(r"  \begin{tabular}{l" + "c" * (n_cols - 1) + r"}")
    latex_lines.append(r"    \toprule")

    # Header Row 1: Group spans
    header1_cells = ["Dataset", "Model"]
    for group in valid_groups:
        n_group_metrics = len([m for m in metric_dict[group] if m in all_metric_cols])
        header1_cells.append(rf"\multicolumn{{{n_group_metrics}}}{{c}}{{{group}}}")
    latex_lines.append(f"    {' & '.join(header1_cells)} \\\\")

    # Header Row 2: Specific metrics per group
    header2_cells = ["", ""]  # Dataset and Model are empty here
    for group in valid_groups:
        for metric in metric_dict[group]:
            if metric in all_metric_cols:
                header2_cells.append(metric)
    latex_lines.append(f"    {' & '.join(header2_cells)} \\\\")
    latex_lines.append(r"    \midrule")

    # Process datasets
    unique_datasets = sorted(df[dataset_col].dropna().unique())
    long_df = df[df[group_col].isin(valid_groups)].copy()
    long_df['dataset'] = pd.Categorical(long_df['dataset'], categories=unique_datasets, ordered=True)
    long_df = long_df.sort_values(['dataset', 'model']).reset_index(drop=True)

    last_dataset = None
    dataset_rows = []  # Each element: list of formatted cells [model_name, val1, val2, ...]

    def flush_block():
        nonlocal last_dataset, dataset_rows
        if not dataset_rows:
            return
        n_models = len(dataset_rows)
        for idx, row in enumerate(dataset_rows):
            if idx == 0:
                row[0] = rf"\multirow{{{n_models}}}{{*}}{{{last_dataset}}}"
            latex_lines.append("    " + " & ".join(row) + r" \\")
        latex_lines.append(r"    \midrule")
        dataset_rows.clear()

    for ds in unique_datasets:
        ds_data = long_df[long_df[dataset_col] == ds]
        # For each (group, metric), compute top-1/top-2 in this dataset
        metric_top_vals = {}
        for (group, metric) in col_structure:
            group_data = ds_data[(ds_data[group_col] == group)]
            vals = pd.to_numeric(group_data[metric], errors='coerce').dropna()
            if len(vals) == 0:
                top1 = top2 = np.nan
            else:
                sorted_vals = vals.sort_values(ascending=False)
                top1 = sorted_vals.iloc[0]
                top2 = sorted_vals.iloc[1] if len(sorted_vals) > 1 else top1
            metric_top_vals[(group, metric)] = (top1, top2)

        models_in_ds = ds_data[model_col].dropna().unique()
        for model in sorted(models_in_ds):  # Lexicographic order
            model_row = ds_data[(ds_data[model_col] == model)]
            cells = ["", model]  # Will set dataset later

            for (group, metric) in col_structure:
                group_filter = model_row[model_row[group_col] == group]
                if group_filter.empty or metric not in group_filter.columns:
                    val_str = "-"
                else:
                    raw_val = group_filter[metric].iloc[0]
                    try:
                        num_val = float(raw_val)
                        top1, top2 = metric_top_vals[(group, metric)]
                        val_str = f"{num_val:.4f}".rstrip('0').rstrip('.')
                        if not np.isnan(top1) and abs(num_val - top1) < 1e-6:
                            val_str = rf"\textbf{{{val_str}}}"
                        elif not np.isnan(top2) and abs(num_val - top2) < 1e-6:
                            val_str = rf"\underline{{{val_str}}}"
                    except (ValueError, TypeError):
                        val_str = str(raw_val)
                cells.append(val_str)

            dataset_rows.append(cells)
        last_dataset = ds
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
    parser = argparse.ArgumentParser(description="Generate combined LaTeX table with customizable per-group metrics.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--hub-metrics", type=str, nargs="+", default=["hit@100", "recall@100"], help="Metrics for Hub group")
    parser.add_argument("--normal-metrics", type=str, nargs="+", default=["hit@100", "recall@100"], help="Metrics for Normal group")
    parser.add_argument("--all-metrics", type=str, nargs="+", default=["hit@100", "recall@100"], help="Metrics for All group")
    parser.add_argument("--caption", type=str, default="Retrieval Performance Across Groups", help="Table caption")
    parser.add_argument("--label", type=str, default="tab:retrieval_combined", help="Table label")

    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.input)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    if "Group" not in df.columns:
        raise ValueError("Column 'Group' not found in CSV.")

    # Define per-group metrics
    metric_dict = {
        "Hub": args.hub_metrics,
        "Normal": args.normal_metrics,
        "All": args.all_metrics
    }

    # Output path
    output_path = args.input.replace(".csv", "_combined.tex")

    table = df_to_latex_combined_groups(
        df,
        dataset_col='dataset',
        model_col='model',
        group_col='Group',
        exclude_cols=['split', 'task', 'ndcg@10_node', 'ndcg@20%N_hub'],
        metric_dict=metric_dict,
        caption=args.caption,
        label=args.label
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(table)

    print(f"✅ Combined LaTeX table saved to: {output_path}")
