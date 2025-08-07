


import pandas as pd
import networkx as nx
import numpy as np

import torch
from torch import nn
from argparse import ArgumentParser
import os
from torch_geometric.data import TemporalData

from .get_baseline_graph import get_baseline_graphs





def get_ctdg_outdegrees(datas:list[TemporalData], max_node_number):
    """
    从边文件构建时序图快照并计算度分布
    :param edge_file: 边文件路径
    :param max_node_number: 最大节点数
    :param time_window: 时间窗口大小
    :param undirected: 是否构建无向图
    :return: 出度列表和唯一出度列表
    """

    # 初始化存储列表
    out_degrees_list = []
    unique_out_degrees_list = []
    
    # 对每个时间窗口进行统计
    

    for data in datas:
        # 统计当前时间窗口内每个源节点的度
        unique_src, src_counts = np.unique(data.src, return_counts=True)
        src_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
        out_degrees_list.append(src_counts)
        
        # 计算唯一出度(去除重复边)
        edge_list = list(zip(data.src, data.dst))
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
    
    return out_degrees_array.T, unique_out_degrees_array.T


       
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
        mmd = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
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
    
    def forward(self, pred_degrees, target_degrees):
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
            loss["histogram_wasserstein_distance"] = torch.tensor(distance, device=pred_degrees.device)
            
        if self.use_kl:
            # 使用KL散度
            eps = 1e-10
            kl_div = np.sum(target_hist * np.log((target_hist + eps) / (pred_hist + eps)))
            loss["histogram_kl_divergence"] = torch.tensor(kl_div, device=pred_degrees.device)
            
        if self.use_mmd:
            mmd_distance = self.mmd_distance(pred_hist, target_hist)
            loss["histogram_mmd_distance"] = mmd_distance
            
        return loss



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
def calculate_mae_mse(predictions, targets):
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
        'mae_allbatches': mae,
        'mse_allbatches': mse
    }
    


# 定义函数计算模型的损失
def calculate_model_loss(degrees_list, target_degrees, model_name):
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
            model_degrees = torch.tensor(degree_list, dtype=torch.float32)
            mae_mse_loss = calculate_mae_mse(model_degrees.unsqueeze(0), target_degrees)
            # 计算直方图匹配损失
            histogram_loss = hist_loss.forward(model_degrees.unsqueeze(0), target_degrees)
            model_loss = {
                **mae_mse_loss,
                **histogram_loss,
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


    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--bwr', type=int, default=2048, help='BWR参数')
    parser.add_argument('--use_feature', type=str, default="no", help='是否使用特征')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    args = parser.parse_args()

    
    # 计算tigger和dggen的损失
    results = []
   
    test_data, tigger_data, dggen_data, max_node_number = get_baseline_graphs(args)
    
    target_degrees, target_unique_degrees = get_ctdg_outdegrees(test_data, max_node_number)
    tigger_out_degrees_list,tigger_unique_out_degrees_list = get_ctdg_outdegrees(tigger_data, max_node_number)
    dggen_out_degrees_list,dggen_unique_out_degrees_list = get_ctdg_outdegrees(dggen_data, max_node_number)

    results = []
    for baseline_name, baseline_out_degrees_list, baseline_unique_out_degrees_list in [('tigger', tigger_out_degrees_list, tigger_unique_out_degrees_list), 
                                                                                       ('dggen', dggen_out_degrees_list, dggen_unique_out_degrees_list)]:
        baseline_loss = calculate_model_loss(baseline_out_degrees_list, target_degrees, baseline_name)
        if baseline_loss:
            baseline_loss['type'] = 'degree'
            baseline_loss['dataset'] = args.data_name
            results.append(baseline_loss)
        
        baseline_loss = calculate_model_loss(baseline_unique_out_degrees_list, target_unique_degrees, baseline_name)
        if baseline_loss:
            baseline_loss['type'] = 'unique'
            baseline_loss['dataset'] = args.data_name
            results.append(baseline_loss)



    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 计算每个指标的排名
    metrics = ['mae_allbatches', 'mse_allbatches', 'histogram_wasserstein_distance', 
              'histogram_kl_divergence', 'histogram_mmd_distance']
    
    # 读取原始结果文件
    original_df = pd.read_csv(f'/data/jiarui_ji/DTGB/{args.data_name}.csv')
    
    # 合并结果
    merged_df = pd.merge(df, original_df, on=[*metrics,"type","dataset"], how='outer')
    
   
    # 更新df
    df = merged_df
    # 保存为CSV
    output_path = f'results/degree/result_{args.data_name}_baseline.csv'
    df.to_csv(output_path, index=False)
    
    # 打印markdown表格
    markdown_table = df.to_markdown(index=False)
    print("\n度分布损失比较:")
    print(markdown_table)
