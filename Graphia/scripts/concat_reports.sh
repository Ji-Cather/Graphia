

for file_name in dst_retrival_matrix_raw.csv edge_matrix.csv graph_list_matrix.csv graph_matrix.csv graph_macro_matrix.csv; do
    echo $file_name
    python Graphia/scripts/concat_reports.py --file_name $file_name
done