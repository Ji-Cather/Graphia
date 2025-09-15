


import pandas as pd
import networkx as nx
import numpy as np

import torch
from torch import nn
from argparse import ArgumentParser
import os
from torch_geometric.data import TemporalData

parser = ArgumentParser()
parser.add_argument('--data_root', type=str, default="./data", help='data root dir')
parser.add_argument('--data_name', type=str, required=True, help='数据集名称')
parser.add_argument('--bwr', type=int, default=2048, help='BWR参数')
parser.add_argument('--use_feature', type=str, default="no", help='是否使用特征')
parser.add_argument('--pred_ratio', type=float, default=0.15, help='预测比例')
parser.add_argument('--cm_order', action="store_true", help='是否使用cm_order')
parser.add_argument('--time_window', type=int, default=24*60*60, help='时间窗口大小')
parser.add_argument("--quantile_mapping", action="store_true", help="Whether to postprocess the prediction with quantile mapping")
parser.add_argument("--rescale", action="store_true", help="Whether to rescale the outdegrees by the maximum edge number")
args = parser.parse_args()



def get_ctdg_outdegrees(data:TemporalData, max_node_number,
                        time_window = 24*60*60):
    """
    从边文件构建时序图快照并计算度分布
    :param edge_file: 边文件路径
    :param max_node_number: 最大节点数
    :param time_window: 时间窗口大小
    :param undirected: 是否构建无向图
    :return: 出度列表和唯一出度列表
    """
    # 获取唯一的时间戳
    unique_times = sorted(np.unique(data.t.cpu().numpy()//time_window)* time_window)
    
    # 初始化存储列表
    out_degrees_list = []
    unique_out_degrees_list = []
    
    # 对每个时间窗口进行统计
    for t in unique_times:
        # 获取当前时间窗口的边
        window_edges = data[data.t == t]
        
        # 统计当前时间窗口内每个源节点的度
        unique_src, src_counts = np.unique(window_edges.src.cpu().numpy(), return_counts=True)
        src_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
        out_degrees_list.append(src_counts)
        
        # 计算唯一出度(去除重复边)
        edge_list = list(zip(window_edges.src.cpu().numpy(), window_edges.dst.cpu().numpy()))
        unique_edge = list(set(edge_list))
        unique_src, src_counts = np.unique([e[0] for e in unique_edge], return_counts=True)
        src_unique_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
        unique_out_degrees_list.append(src_unique_counts)
    
    # 转换为numpy数组格式
    out_degrees_array = np.array([[src_degree_list.get(int(src_id), 0) 
                                 for src_degree_list in out_degrees_list] 
                                for src_id in range(max_node_number + 1)])
    
    unique_out_degrees_array = np.array([[src_unique_degree_list.get(int(src_id), 0) 
                                        for src_unique_degree_list in unique_out_degrees_list] 
                                       for src_id in range(max_node_number + 1)])
    
    return out_degrees_array, unique_out_degrees_array


       
# 添加直方图匹配损失函数
class HistogramMatchingLoss(nn.Module):
    def __init__(self, num_bins=10, use_wasserstein=True, use_kl=True, use_mmd=True, kernel_type='gaussian', sigma=1.0):
        """
        基于直方图匹配的损失函数
        :param num_bins: 直方图的箱数
        :param use_wasserstein: 是否使用Wasserstein距离，否则使用KL散度
        """
        super().__init__()
        self.num_bins = num_bins
        self.use_wasserstein = use_wasserstein
        self.use_kl = use_kl
        self.use_mmd = use_mmd
        self.kernel_type = kernel_type
        self.sigma = sigma
        
    def rbf_kernel(self, x, y):
        """
        计算高斯核
        :param x: 第一个样本
        :param y: 第二个样本
        :return: 核函数值
        """
        return torch.exp(-torch.sum((x - y) ** 2) / (2 * self.sigma ** 2))
    
    
    
    
    
    def mmd_distance(self, x, y):
        """
        计算MMD距离
        :param x: 第一个分布的样本
        :param y: 第二个分布的样本
        :return: MMD距离
        """
        # 确保x和y是张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        # 计算核矩阵
        xx = torch.zeros((x.shape[0], x.shape[0]))
        yy = torch.zeros((y.shape[0], y.shape[0]))
        xy = torch.zeros((x.shape[0], y.shape[0]))
        
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                xx[i, j] = self.rbf_kernel(x[i], x[j])
                
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                yy[i, j] = self.rbf_kernel(y[i], y[j])
                
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                xy[i, j] = self.rbf_kernel(x[i], y[j])
                
        # 计算MMD距离
        mmd_squared = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        mmd = torch.sqrt(torch.clamp(mmd_squared, min=1e-10))  # 防止负数或数值不稳定
        return mmd
        
    def compute_histogram(self, degrees, num_bins=None):
        """
        计算度数直方图
        :param degrees: 节点度数张量
        :param num_bins: 直方图的箱数
        :return: 直方图分布
        """
        if num_bins is None:
            num_bins = self.num_bins
            
        # 将度数展平
        flat_degrees = degrees.flatten()
        
        # 计算直方图
        hist, _ = np.histogram(flat_degrees, bins=int(num_bins), density=False)
        hist = hist[1:]
        
        # 归一化
        if np.sum(hist) > 0 :
            hist = hist / np.sum(hist)
            
        return hist
    
    def forward(self, pred_degrees, target_degrees, name = "degree"):
        """
        计算直方图匹配损失
        :param pred_degrees: 预测的节点度数
        :param target_degrees: 目标节点度数
        :return: 损失值
        """
        # 计算预测和目标直方图
        pred_hist = self.compute_histogram(pred_degrees)
        target_hist = self.compute_histogram(target_degrees)
        
        loss = {}
        
        if self.use_wasserstein:
            # 使用Wasserstein距离
            # 创建累积分布函数
            pred_cdf = np.cumsum(pred_hist)
            target_cdf = np.cumsum(target_hist)
            
            # 计算Wasserstein距离
            distance = np.sum(np.abs(pred_cdf - target_cdf)) / len(pred_cdf)
            loss[f"histogram_wasserstein_distance_{name}"] = torch.tensor(distance, device=pred_degrees.device)
            
        if self.use_kl:
            # 使用KL散度
            eps = 1e-10
            kl_div = np.sum(target_hist * np.log((target_hist + eps) / (pred_hist + eps)))
            loss[f"histogram_kl_divergence_{name}"] = torch.tensor(kl_div, device=pred_degrees.device)
            
        if self.use_mmd:
            mmd_distance = self.mmd_distance(pred_hist, target_hist)
            loss[f"histogram_mmd_distance_{name}"] = torch.tensor(mmd_distance, device=pred_degrees.device)
            
        return loss

def get_graph_snapshots(path,
                        max_node_number,
                        time_window=86400, 
                        undirected=False):
    """
    按照指定的时间窗口切分数据，创建图的快照，并统计每个快照的出度分布
    
    参数:
    path: CSV文件路径
    time_window: 时间窗口大小，默认为86400（一天）
    undirected: 是否path记录的是无向图，默认是False；如果是true，则
    
    返回:
    out_degrees_list: 所有快照图的出度列表
    snapshots: 所有快照图的列表
    
    
    """
    # 读取CSV文件
    df = pd.read_csv(path)
    
    # Ensure 'src', 'dst', and 't' columns are integers
    df['src'] = df['src'].astype(int)
    df['dst'] = df['dst'].astype(int)
    # Convert 't' column to numeric, coercing errors to NaN, then fill NaN with 0 or drop as needed
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df['t'] = df['t'].fillna(0)
    # Use np.int64 if possible, otherwise fallback to np.float64
    try:
        if df['t'].max() <= np.iinfo(np.int64).max:
            df['t'] = df['t'].astype(np.int64)
        else:
            df['t'] = df['t'].astype(np.float64)
    except OverflowError:
        df['t'] = df['t'].astype(np.float64)
    
    # 确保时间戳按升序排序
    df = df.sort_values(by='t')
    
    # 计算时间范围
    min_time = df['t'].min()
    max_time = min(df['t'].max(), time_window*10)
    
    # 创建时间窗口边界
    time_boundaries = np.arange(min_time, max_time + time_window, time_window)
    
    out_degrees_list = []
    
    # 按时间窗口切分数据并创建图
    for i in range(len(time_boundaries))[:10]:
        start_time = time_boundaries[i]
        if i == len(time_boundaries) - 1:
            end_time = max_time + 1
        else:
            end_time = time_boundaries[i + 1]
        
        # 筛选当前时间窗口内的数据
        window_df = df[(df['t'] >= start_time) & (df['t'] < end_time)]
        
        if not window_df.empty:
            # 统计当前时间窗口内每个源节点的度
            unique_src, src_counts = np.unique(window_df['src'].values, return_counts=True)
            src_degree_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
            
            # 计算唯一出度(去除重复边)
            edge_list = list(zip(window_df['src'].values, window_df['dst'].values))
            unique_edge = list(set(edge_list))
            unique_src, src_counts = np.unique([e[0] for e in unique_edge], return_counts=True)
            src_unique_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
            
            # 转换为numpy数组格式
            out_degrees = torch.zeros(max_node_number+1)
            for src_id in range(max_node_number + 1):
                if undirected:
                    out_degrees[src_id] = src_unique_counts.get(int(src_id), 0) / 2
                else:
                    out_degrees[src_id] = src_degree_counts.get(int(src_id), 0)
            

            out_degrees_list.append(out_degrees.tolist())
    
    return out_degrees_list

import pandas as pd

def add_rank_score(csv_path, output_path='ranked_output.csv'):
    df = pd.read_csv(csv_path)

    # 需要排序的指标列
    metric_cols = [
        'mae_allbatches',
        'mse_allbatches',
        'histogram_wasserstein_distance',
        'histogram_kl_divergence',
        'histogram_mmd_distance'
    ]

    # 对每个 metric 列按升序排名（最小为1）
    for col in metric_cols:
        rank_col = col + '_rank'
        df[rank_col] = df[col].rank(method='min', ascending=True)

    # 总排名分数
    df['rank_score'] = df[[c + '_rank' for c in metric_cols]].sum(axis=1)

    # 按 rank_score 升序排序
    df = df.sort_values('rank_score')

    df.to_csv(output_path, index=False)
    print(f"Ranked results written to: {output_path}")



# 计算MAE和MSE损失
def calculate_mae_mse(predictions, targets, name = "degree"):
    """
    计算MAE和MSE损失
    :param predictions: 预测值张量
    :param targets: 目标值张量
    :return: 包含MAE和MSE的字典
    """
    # 确保输入是张量
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
        
    # 计算MAE
    mae = torch.mean(torch.abs(predictions - targets).float())
    
    # 计算MSE
    mse = torch.mean((predictions - targets).float() ** 2)

    return {
        f'mae_allbatches_{name}': mae,
        f'mse_allbatches_{name}': mse
    }
    


# 定义函数计算模型的损失
def calculate_model_loss(degrees_list, target_degrees, target_degrees_unique, model_name):
    """
    计算模型的各种损失
    :param degrees_list: 模型生成的度列表
    :param target_degrees: 目标度分布
    :param model_name: 模型名称
    :return: 包含各种损失的字典
    """
    # 初始化直方图匹配损失函数


    hist_loss = HistogramMatchingLoss(num_bins=100, 
                                    use_wasserstein=True, 
                                    use_kl=True,
                                    use_mmd=True,
                                    kernel_type='rbf')
    if len(degrees_list) > 0:
        model_loss_all = []
        for degree_list in degrees_list:
            all_predicted_degrees = torch.tensor(degree_list, dtype=torch.float32).unsqueeze(0)
            all_target_degrees = target_degrees

            mae_mse_metrics = calculate_mae_mse(all_predicted_degrees, all_target_degrees, name = "degree")
            histogram_loss = hist_loss.forward(all_predicted_degrees, all_target_degrees, name = "degree")
            
            mae_mse_metrics_unique = calculate_mae_mse(all_predicted_degrees, target_degrees_unique, name = "unique")
            histogram_loss_unique = hist_loss.forward(all_predicted_degrees, target_degrees_unique, name = "unique")

            model_loss = {
                **mae_mse_metrics,
                **histogram_loss,
                **mae_mse_metrics_unique,
                **histogram_loss_unique
            }
            model_loss = {k: v.item() for k, v in model_loss.items()}
            model_loss_all.append(model_loss)
        # 对于model_loss_all中所有字典的相同键计算平均值
        model_loss_aggregate = {}
        for key in model_loss_all[0].keys():
            values = [d[key] for d in model_loss_all]
            model_loss_aggregate[key] = sum(values) / len(values)
        model_loss_aggregate['model'] = model_name
        return model_loss_aggregate
    return None


def read_metric_json(json_path, model="InformerDecoder", dataset="weibo_daily"):
    import json
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    # 将metrics字典转换为DataFrame
    metrics_df = pd.DataFrame([metrics])
    # 转置metrics字典，使InformerDecoder为key，metric为列
    metrics_transposed = {}
    for key, value in metrics.items():
        for metric_name, metric_value in value.items():
            if metric_name not in metrics_transposed:
                metrics_transposed[metric_name] = {}
            metrics_transposed[metric_name][key] = metric_value
        metrics_transposed["model"]={key: model}
        metrics_transposed["dataset"]={key: dataset}

    metrics_df = pd.DataFrame(metrics_transposed)
    return metrics_df


if __name__ == "__main__":
    
    # 计算tigger和dggen的损失
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from LLMGGen.utils.bwr_ctdg import BWRCTDGALLDataset
    import pandas as pd
    dataset = BWRCTDGALLDataset(
        root=os.path.join(args.data_root, args.data_name), 
        time_window=args.time_window,
        bwr=args.bwr,
        use_feature=args.use_feature,
        pred_ratio=args.pred_ratio,
        cm_order = args.cm_order
    )

    data_baseline_map = {
        "8days_dytag_large":[
            "/data/jiarui_ji/DGGen/results/synthetic_data/8days_dytag_large_uf_False_20250522_233700_b604.csv",
            "/data/jiarui_ji/tigger/models/8days_dytag_large/results/generated_edges.csv"
        ],
        "8days_dytag_small":[
            "/data/jiarui_ji/DGGen/results/synthetic_data/8days_dytag_small_uf_False_20250522_233641_186a.csv",
            "/data/jiarui_ji/tigger/models/8days_dytag_small/results/generated_edges.csv"
        ],
        "8days_dytag_small_text":[
            "/data/jiarui_ji/DGGen/results/synthetic_data/8days_dytag_small_textfno.csv",
            "/data/jiarui_ji/tigger/models/8days_dytag_small_text/results/generated_edges.csv"
        ],
         "8days_dytag_small_text_en":[
            "LLMGGen/baselines/DGGen/results/synthetic_data/8days_dytag_small_text_en.csv",
            "LLMGGen/baselines/tigger/models/8days_dytag_small_text_en/results/generated_edges.csv",
            "LLMGGen/baselines/DYMOND/8days_dytag_small_text_en/learned_parameters/generated_graph/results/generated_edges.csv"
        ],
        "weibo_daily":[
            "LLMGGen/baselines/DGGen/results/synthetic_data/weibo_daily.csv",
            "LLMGGen/baselines/tigger/models/weibo_daily/results/generated_edges.csv"
        ],
        "imdb":[
            "LLMGGen/baselines/DGGen/results/synthetic_data/imdb.csv",
            "LLMGGen/baselines/tigger/models/imdb/results/generated_edges.csv"
        ],
        "weibo_tech":[
            "LLMGGen/baselines/DGGen/results/synthetic_data/weibo_tech.csv",
            "LLMGGen/baselines/tigger/models/weibo_tech/results/generated_edges.csv"
        ]
    }
    
    test_data = dataset.test_data.ctdg.to(device)
   
    max_node_number = max(test_data.src.max(), test_data.dst.max())
    target_degrees, target_unique_degrees = get_ctdg_outdegrees(test_data, max_node_number, time_window= dataset.time_window)
    
    results = []
    import re
    # 计算degree和unique degree的损失
    # for degree_type, target_deg in [('degree', target_degrees[:,-1]), 
    #                                 ('unique', target_unique_degrees[:,-1])]:
    # tigger-I
    for baseline_path in data_baseline_map[args.data_name]:
        match = re.search(r'LLMGGen/baselines/([^/]+)/', baseline_path)
        baseline_name = match.group(1)  # 返回第一个捕获组（即斜杠之间的内
        out_degrees_list = get_graph_snapshots(baseline_path, max_node_number, time_window=1, undirected=True)
        loss = calculate_model_loss(out_degrees_list, target_degrees[:,-1], target_unique_degrees[:,-1], baseline_name)
        if  loss:
            # tigger_loss['type'] = degree_type
            loss['dataset'] = args.data_name
            results.append(loss)

    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 计算每个指标的排名
    metrics = ['mae_allbatches_degree', 'mse_allbatches_degree', 'histogram_wasserstein_distance_degree', 
              'histogram_kl_divergence_degree', 'histogram_mmd_distance_degree',
                'mae_allbatches_unique', 'mse_allbatches_unique', 'histogram_wasserstein_distance_unique',
                'histogram_kl_divergence_unique', 'histogram_mmd_distance_unique']
    

    
    # 读取原始结果文件
    root = f'saved_results_deg/InformerDecoder_seed0_bwr{args.bwr}_qm{args.quantile_mapping}_uf{args.use_feature}_cm{args.cm_order}_rescale{args.rescale}/{args.data_name}/'
    # original_df = pd.read_csv(f'/data/jiarui_ji/DTGB/{args.data_name}.csv')
    original_df = read_metric_json(os.path.join(root,'metrics.json'),
                                   model="InformerDecoder", 
                                   dataset=args.data_name)
    # 合并结果
    # Ensure columns used for merging have the same dtype
    for col in metrics:
        df[col] = df[col].astype(float)
        original_df[col] = original_df[col].astype(float)
    merged_df = pd.merge(df, original_df, on=[*metrics,"model","dataset"], how='outer')
    
    # 更新df
    df = merged_df
    # 保存为CSV
    output_path = os.path.join(root,f'result_{args.data_name}_baseline.csv')
    df.to_csv(output_path, index=False)
    
    # 打印markdown表格
    markdown_table = df.to_markdown(index=False)
    print("\n度分布损失比较:")
    print(markdown_table)


