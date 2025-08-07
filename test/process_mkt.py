import pandas as pd


edge_path = "data/8days_dytag_small_text/raw/edge_feature.csv"
# edge_path = "data/weibo_tech/raw/edge_feature.csv"

edge_df =pd.read_csv(edge_path)

# 将时间戳转换为天数
edge_df['day'] = edge_df['ts'] // (24*60*60)

# 按天分组并计算每天的边数量
edge_counts = edge_df.groupby('day').size()

# 计算统计信息
mean_edges = edge_counts.mean()
min_edges = edge_counts.min()
max_edges = edge_counts.max()

print(f"平均每天边数量: {mean_edges:.2f}")
print(f"最小每天边数量: {min_edges}")
print(f"最大每天边数量: {max_edges}")
