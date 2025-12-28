# for file_name in edge_matrix_llama31-70B.csv edge_matrix_llama33-70B.csv edge_matrix_Qwen3-32B.csv; do # abalate eval llms
for file_name in dst_retrival_matrix_raw.csv edge_matrix.csv graph_list_matrix.csv graph_matrix.csv graph_macro_matrix.csv; do
    echo $file_name
    python Graphia/scripts/concat_reports.py --file_name $file_name
done