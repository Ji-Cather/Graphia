# 从 node embedding中提取出 degree, query semantic id

import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import Dataset, DataLoader

from .models.informer.model import Informer, InformerTranverse, InformerDecoder
from .utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from .utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from .evaluate_models_utils import evaluate_model_node_regression_v2
from .utils.DataLoader import get_ctdg_generation_data
from .utils.bwr_ctdg import BWRCTDGDataset, custom_collate
from .utils.EarlyStopping import EarlyStopping
from .utils.load_configs import get_node_regression_args
def fit_daily_edge_count_distribution(train_data_ctdg):
    """
    对训练数据中每天的边数量进行高斯分布拟合
    
    Args:
        train_data_ctdg: 训练数据集
    
    Returns:
        dict: 高斯分布参数 (mu, sigma) 和每天的实际边数
    """
    ctdg = train_data_ctdg.ctdg
    
    # 按天统计边数量
    daily_edge_counts = {}
    
    for i in range(len(ctdg.src)):
        t = ctdg.t[i].item()
        day = int(t // 1)  # 假设时间是以天为单位，或者需要根据实际情况调整
        
        if day not in daily_edge_counts:
            daily_edge_counts[day] = 0
        daily_edge_counts[day] += 1
    
    # 提取每日边数量列表
    edge_counts = list(daily_edge_counts.values())
    
    # 拟合高斯分布
    if len(edge_counts) > 1:
        mu, sigma = np.mean(edge_counts), np.std(edge_counts)
    else:
        mu, sigma = edge_counts[0], 0.0
    
    return {
        'mu': float(mu),
        'sigma': float(sigma),
        'daily_counts': daily_edge_counts,
        'all_counts': edge_counts
    }

def sample_daily_edge_counts(gaussian_params, num_days=10, seed=None):
    """
    从高斯分布中采样未来几天的边数量
    
    Args:
        gaussian_params: 高斯分布参数
        num_days: 预测天数
        seed: 随机种子
    
    Returns:
        list: 每天采样的边数量
    """
    if seed is not None:
        np.random.seed(seed)
    
    mu, sigma = gaussian_params['mu'], gaussian_params['sigma']
    
    if sigma == 0:
        # 如果标准差为0，返回均值
        sampled_counts = [int(mu)] * num_days
    else:
        # 从高斯分布采样并确保为正整数
        samples = np.random.normal(mu, sigma, num_days)
        sampled_counts = [max(1, int(abs(s))) for s in samples]
    
    return sampled_counts

def determine_source_node_degree_from_edge_count(edge_count, avg_degree_per_node=2.0):
    """
    根据边数量确定源节点的度数分布
    假设平均每节点产生一定数量的边
    
    Args:
        edge_count: 边数量
        avg_degree_per_node: 每个节点平均产生的边数
    
    Returns:
        int: 源节点数量
    """
    # 简单模型：源节点数 = 边数 / 平均度数
    source_node_count = max(1, int(edge_count / avg_degree_per_node))
    return source_node_count
def generate_degree_distribution_for_prediction_days(train_data_ctdg, test_data_ctdg, prediction_days=None, seed=None):
    """
    为预测期生成每天的度数分布，并按照src_id作为索引
    
    Args:
        train_data_ctdg: 训练数据
        test_data_ctdg: 测试数据
        prediction_days: 预测天数，默认使用test_data_ctdg.pred_len
        seed: 随机种子
    
    Returns:
        dict: 包含度数分布和源节点索引的字典
    """
    if prediction_days is None:
        prediction_days = test_data_ctdg.pred_len
    
    # 1. 拟合每日边数的高斯分布
    edge_dist_params = fit_daily_edge_count_distribution(train_data_ctdg)
    
    # 2. 采样预测期每天的边数量
    predicted_edge_counts = sample_daily_edge_counts(
        edge_dist_params, 
        num_days=prediction_days, 
        seed=seed
    )
    
    # 3. 获取测试数据中的源节点索引
    unique_src_ids = np.unique(test_data_ctdg.ctdg.src.numpy())
    max_src_id = int(test_data_ctdg.ctdg.src.max().item())
    
    # 4. 为每个预测天生成度数分布
    prediction_degree_distribution = {}
    
    for day_idx, edge_count in enumerate(predicted_edge_counts):
        # 计算该天需要的源节点数量
        source_node_count = determine_source_node_degree_from_edge_count(edge_count)
        
        # 为每个源节点分配度数（简化处理：平均分配）
        avg_degree = max(1, edge_count // max(1, source_node_count))
        degrees = np.full(source_node_count, avg_degree)
        
        # 如果有余数，将余数分配给前面的节点
        remainder = edge_count % max(1, source_node_count)
        if remainder > 0:
            degrees[:remainder] += 1
            
        # 随机选择源节点（如果需要的节点数小于等于可用节点数）
        if source_node_count <= len(unique_src_ids):
            selected_src_ids = np.random.choice(unique_src_ids, source_node_count, replace=False)
        else:
            # 如果需要的节点数大于可用节点数，则重复使用节点
            repeated_src_ids = np.tile(unique_src_ids, (source_node_count // len(unique_src_ids)) + 1)
            selected_src_ids = repeated_src_ids[:source_node_count]
        
        prediction_degree_distribution[day_idx] = {
            'edge_count': edge_count,
            'source_node_count': source_node_count,
            'source_node_ids': selected_src_ids.astype(int).tolist(),
            'degrees': degrees.tolist(),
            'degree_per_node': edge_count / max(1, source_node_count) if source_node_count > 0 else 0
        }
    
    return {
        'edge_distribution_params': edge_dist_params,
        'predicted_days_degree': prediction_degree_distribution,
        'predicted_edge_counts': predicted_edge_counts,
        'unique_src_ids': unique_src_ids.astype(int).tolist(),
        'max_src_id': max_src_id
    }

def convert_to_numpy_degree_format(degree_distribution, max_src_nodes=None):
    """
    将度数分布转换为numpy数组格式
    格式: [num_src_nodes, num_days]，其中索引对应src_id
    
    Args:
        degree_distribution: 度数分布字典
        max_src_nodes: 最大源节点数，如果为None则自动计算
    
    Returns:
        np.ndarray: 形状为[max_src_nodes, num_days]的度数数组
    """
    predicted_days_degree = degree_distribution['predicted_days_degree']
    num_days = len(predicted_days_degree)
    
    if max_src_nodes is None:
        max_src_nodes = degree_distribution['max_src_id'] + 1
    
    # 初始化度数数组，索引对应src_id
    degree_array = np.zeros((max_src_nodes, num_days), dtype=np.int32)
    
    # 填充每天的度数
    for day_idx in range(num_days):
        day_info = predicted_days_degree[day_idx]
        source_node_ids = np.array(day_info['source_node_ids'])
        degrees = np.array(day_info['degrees'])
        
        # 将度数放置在对应src_id的位置
        for src_id, degree in zip(source_node_ids, degrees):
            if src_id < max_src_nodes:  # 确保src_id在有效范围内
                degree_array[src_id, day_idx] = degree
    
    return degree_array

# 在主函数中使用修改后的代码
import math
if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_regression_args(is_evaluation=False)
   
    train_data_ctdg, val_data_ctdg, test_data_ctdg = \
        get_ctdg_generation_data(args = args)

    # 生成度数分布，包含测试数据信息
    degree_distribution = generate_degree_distribution_for_prediction_days(
        train_data_ctdg, 
        test_data_ctdg,
        prediction_days=test_data_ctdg.pred_len,  # 预测天数
        seed=42
    )
    
    # 转换为numpy数组格式，索引对应src_id
    degree_array = convert_to_numpy_degree_format(degree_distribution)

    save_dir = f"saved_results_deg/seq_deg/{args.data_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存度数分布参数和numpy数组
    degree_save_path = os.path.join(save_dir, 'test_degree.pt')
    torch.save({
        'degree': torch.from_numpy(degree_array),  # 形状: [num_src_nodes, num_days]
        'unique_degree': torch.from_numpy(degree_array)  # 形状: [num_src_nodes, num_days]
    }, degree_save_path)
    
    print(f"Degree distribution saved to {degree_save_path}")
    print(f"Degree array shape: {degree_array.shape}")
    print(f"Max src id: {degree_distribution['max_src_id']}")
    
    # 显示一些示例数据
    print("\nSample degree data (first 5 src_ids, first 3 days):")
    print(degree_array[:5, :3])