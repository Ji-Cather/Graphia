import pandas as pd
import os


def process_weibo_tech(DGGB_root, save_root):
    edge_df = pd.read_csv(os.path.join(DGGB_root, 'edge_weibo.csv'),index_col=0)

    # 将timestamp列转换为datetime格式
    edge_df['timestamp'] = pd.to_datetime(edge_df['timestamp'])

    # 将datetime转换为timestamp
    edge_df['timestamp'] = edge_df['timestamp'].astype(int) // 10**9

    edge_df = edge_df.rename(columns={'timestamp': 'ts',
                                    "source_user_id":"src",
                                    "destination_user_id":"dst"})

    node_df = pd.read_csv(os.path.join(DGGB_root, 'node_weibo.csv'),index_col=0)

    # 创建node_id到新id的映射
    node_order_mapping = {old: new + 1 for new, old in enumerate(node_df['node_id'].unique())}

    # 对node_df的node_id进行映射
    node_df['node_id'] = node_df['node_id'].map(node_order_mapping)

    # 对edge_df的src和dst进行映射
    edge_df['src'] = edge_df['src'].map(node_order_mapping)
    edge_df['dst'] = edge_df['dst'].map(node_order_mapping)

    # 删除自环 src == dst
    edge_df = edge_df[edge_df['src'] != edge_df['dst']]

    # 确保node_id为整数类型
    node_df['node_id'] = node_df['node_id'].astype(int)
    edge_df['src'] = edge_df['src'].astype(int)
    edge_df['dst'] = edge_df['dst'].astype(int)

    # 计算每个时间戳的边数量
    window_size = 24*60*60
    # 将时间戳转换为天数
    edge_df['day'] = edge_df['ts'] // window_size

    # 按天分组并计算每天的边数量
    edge_counts = edge_df.groupby('day').size()

    edge_df = edge_df.groupby(['src', 'dst', 'day']).first().reset_index()

    node_df.to_csv(os.path.join(save_root, 'node_feature.csv'), index=False)
    edge_df.to_csv(os.path.join(save_root, 'edge_feature.csv'), index=False)
    print(node_df.shape)
    print(edge_df.shape)
   

    # 按天分组并计算每天的边数量
    edge_counts = edge_df.groupby('day').size()
    # 计算统计信息
    mean_edges = edge_counts.mean()
    min_edges = edge_counts.min()
    max_edges = edge_counts.max()
    print(f"len group: {len(edge_counts)}")
    print(f"平均每天边数量: {mean_edges:.2f}")
    print(f"最小每天边数量: {min_edges}")
    print(f"最大每天边数量: {max_edges}")
    
def process_weibo_daily(DGGB_root, save_root):
    edge_df = pd.read_csv(os.path.join(DGGB_root, 'edge_weibo.csv'),index_col=0)
    node_df = pd.read_csv(os.path.join(DGGB_root, 'node_weibo.csv'),index_col=0)
    # 将timestamp列转换为datetime格式
    edge_df['timestamp'] = pd.to_datetime(edge_df['timestamp'])

    # 将datetime转换为timestamp
    edge_df['timestamp'] = edge_df['timestamp'].astype(int) // 10**9

    edge_df = edge_df.rename(columns={'timestamp': 'ts',
                                    "source_user_id":"src",
                                    "destination_user_id":"dst"})

    

    # 创建node_id到新id的映射
    node_order_mapping = {old: new + 1 for new, old in enumerate(node_df['node_id'].unique())}

    # 对node_df的node_id进行映射
    node_df['node_id'] = node_df['node_id'].map(node_order_mapping)

    # 对edge_df的src和dst进行映射
    edge_df['src'] = edge_df['src'].map(node_order_mapping)
    edge_df['dst'] = edge_df['dst'].map(node_order_mapping)

    # 删除自环 src == dst
    edge_df = edge_df[edge_df['src'] != edge_df['dst']]

    # 确保node_id为整数类型
    node_df['node_id'] = node_df['node_id'].astype(int)
    edge_df['src'] = edge_df['src'].astype(int)
    edge_df['dst'] = edge_df['dst'].astype(int)

    # 计算每个时间戳的边数量    # 计算每个时间戳的边数量
    window_size = 24*60*60
    # 将时间戳转换为天数
    edge_df['day'] = edge_df['ts'] // window_size

    # 按天分组并计算每天的边数量
    edge_counts = edge_df.groupby('day').size()

    edge_df = edge_df.groupby(['src', 'dst', 'day']).first().reset_index()

    node_df.to_csv(os.path.join(save_root, 'node_feature.csv'), index=False)
    edge_df.to_csv(os.path.join(save_root, 'edge_feature.csv'), index=False)
    print(node_df.shape)
    print(edge_df.shape)


    # 按天分组并计算每天的边数量
    edge_counts = edge_df.groupby('day').size()
    # 计算统计信息
    mean_edges = edge_counts.mean()
    min_edges = edge_counts.min()
    max_edges = edge_counts.max()
    print(f"len group: {len(edge_counts)}")
    print(f"平均每天边数量: {mean_edges:.2f}")
    print(f"最小每天边数量: {min_edges}")
    print(f"最大每天边数量: {max_edges}")
    
    

if __name__ == "__main__":
    # DGGB_root = 'data/DGGB'
    # save_root = 'data/weibo_tech'
    process_weibo_daily("Graphia/data/weibo_daily/raw", "Graphia/data/weibo_daily/raw")
    process_weibo_tech("Graphia/data/weibo_tech/raw", "Graphia/data/weibo_tech/raw")
    