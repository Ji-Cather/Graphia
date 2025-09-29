import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from .edge.tenc import Tenc,diffusion
from .informer.model import InformerTranverse


def quantile_mapping_func(generated_data, target_dis_quantile):
    """
    将生成的数据通过分位数映射到目标分布
    
    参数:
    generated_data: 生成的数据
    target_dis_quantile: N*T-1*1的目标度数矩阵
    
    返回:
    映射后的度数值，保持原始生成数据的形状
    """
    # 将目标分布展平为一维数组
    target_flat = target_dis_quantile.reshape(-1)
    generated_flat = generated_data.reshape(-1)
    
    # 获取生成数据的排序索引和排名
    generated_sorted, generated_indices = torch.sort(generated_flat)
    rank_indices = torch.zeros_like(generated_indices)
    rank_indices[generated_indices] = torch.arange(len(generated_flat), device=generated_data.device)
    generated_ranks = rank_indices.float() / (len(generated_flat) - 1)
    
    # 对目标分布进行排序
    sorted_target, _ = torch.sort(target_flat)
    
    # 计算插值的索引位置
    interp_indices = generated_ranks * (len(sorted_target) - 1)
    interp_indices_floor = torch.floor(interp_indices).long()
    interp_indices_ceil = torch.ceil(interp_indices).long()
    
    # 处理边界情况
    interp_indices_ceil = torch.clamp(interp_indices_ceil, max=len(sorted_target)-1)
    
    # 计算插值权重
    weights_ceil = interp_indices - interp_indices_floor.float()
    weights_floor = 1.0 - weights_ceil
    
    # 执行线性插值
    mapped_data = weights_floor * sorted_target[interp_indices_floor] + weights_ceil * sorted_target[interp_indices_ceil]
    
    # 保持原始生成数据的形状
    return mapped_data.reshape_as(generated_data).int()

def pad_edges(edges_list, max_len, padding_value = 0):
    """
    将边列表进行统一长度处理
    :param edges_list: 边列表
    :param max_len: 目标长度
    :return: 处理后的边列表和对应的长度矩阵
    """
    processed_edges = []
    edges_length = []
    for node_edges in edges_list:
        current_len = len(node_edges)
        edges_length.append(min(current_len, max_len))
        if current_len > max_len:
            # 如果当前序列长度超过目标长度，则截断
            processed_edges.append(node_edges[:max_len])
        else:
            # 如果当前序列长度小于目标长度，则补零
            padding = np.zeros((max_len - current_len, 3)) + padding_value
            processed_edges.append(np.vstack((node_edges + 1, padding)))
    return processed_edges, np.array(edges_length)

        
class GraphGenerator(nn.Module):
    def __init__(self, 
                 bwr,
                 src_node_feature_dim, 
                 dst_node_feature_dim, 
                 edge_feature_dim,
                 max_node_num, 
                 input_len, # input snapshot_len
                 node_raw_features,
                 edge_raw_features,
                 src_node_out_feature_dim: int = 1,
                 dropout = 0.1,
                 candidate_dst_num: int = 100,
                 device = torch.device('cuda:0')):
        super(GraphGenerator, self).__init__()
        # 输入feature，首位为degree
        # p(src,dst,e) = p(src)p(dst|src)p(e|src,dst)
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.device = device
        self.candidate_dst_num = candidate_dst_num
        self.src_node_generator = InformerTranverse(bwr, 
                                                    src_node_feature_dim,
                                                    input_len, 
                                                    src_node_out_feature_dim = src_node_out_feature_dim, 
                                                    device = device)
        self.dst_node_generator = EdgeGenerator(dst_node_feature_dim,
                                                max_node_num, 
                                                candidate_dst_num,
                                                device = device)
        self.edge_generator = EdgeGenerator(edge_feature_dim, max_node_num, input_len, device = device)
        
        
        
    def forward(self, 
                input_src_node_features,
                input_src_node_interact_times,
                input_src_node_pred_feature,
                input_src_node_pred_interact_times,
                input_edges,
                target_src_node_degrees,
                target_edges,
                ):
        # dst_ids: [B, Sum(Degree)]
        # dst_len: [B]
        # pred_dst_ids: [B, target_degree]
        # pred_dst_len: [B]
        # dst_interact_times: [B, Sum(Degree)]
        # pred_interact_times: [B, target_degree]

        
        src_candidates = self.src_node_generator(input_src_node_features, input_src_node_interact_times, 
                                                 input_src_node_pred_feature, input_src_node_pred_interact_times)
        src_loss = self.src_node_generator.cacu_loss(src_candidates, target_src_node_degrees, loss_type='l1')
        
        input_edges = [np.vstack(input_edges[node]) for node in range(len(input_edges))]
        target_edges = [np.vstack(target_edges[node]) for node in range(len(target_edges))]
        
        
        input_edges_select = []
        # 预先计算每个节点的边数量和目标边数量
        input_edges_sizes = np.array([edges.shape[0] for edges in input_edges])
        target_edges_sizes = np.array([edges.shape[0] for edges in target_edges])
        
        # 创建掩码标识哪些节点的边数量大于self.candidate_dst_num
        mask_large = input_edges_sizes > self.candidate_dst_num
        
        # 处理边数量大于self.candidate_dst_num的节点
        for i in np.where(mask_large)[0]:
            # 使用向量化操作一次性生成所有随机索引
            max_start = input_edges_sizes[i] - self.candidate_dst_num
            indices = np.random.randint(0, max_start, size=target_edges_sizes[i])
            if len(indices) > 0:
                # 使用numpy的高级索引直接构建结果
                selected = np.stack([input_edges[i][idx:idx+self.candidate_dst_num] for idx in indices])
                input_edges_select.extend(selected)
        
        # 处理边数量小于等于self.candidate_dst_num的节点
        for i in np.where(~mask_large)[0]:
            # 直接复制现有边到所有目标
            selected = np.array([input_edges[i]] * target_edges_sizes[i])
            input_edges_select.extend(selected)
            
        
        # 处理input_edges和pred_edges，使其长度统一为self.candidate_dst_num
        processed_input_edges, input_edges_length = pad_edges(input_edges_select, self.candidate_dst_num)
        
        # 将处理后的边转换为张量
        input_edges_tensor = ç[:,:,-1].int()
        input_edges_length = input_edges_length
        target_dst_ids_tensor = torch.tensor(np.concatenate(target_edges, axis=0), 
                                          device=self.device)[:,-1].int()
        
        
        dst_loss = self.dst_node_generator(input_edges_tensor, 
                                           input_edges_length, 
                                            target_dst_ids_tensor)
        
        
        
        
        
        edge_attr_loss = self.edge_generator(input_edges_tensor, 
                                        input_edges_length, 
                                        target_dst_ids_tensor)
        
        return {
            "src_loss": src_loss,
            "dst_loss": dst_loss,
            "edge_attr_loss": edge_attr_loss,
        }
        
    def sample_one_step(self,
                    src_ids,
                    input_src_node_features,
                    input_src_node_interact_times,
                    input_src_node_pred_feature,
                    input_src_node_pred_interact_times,
                    pred_edge_num: int = None,
                    src_quantile_mapping: bool = False,
                    all_history_src_node_degrees: torch.Tensor = None
                    ):
        src_scores = self.src_node_generator(input_src_node_features, input_src_node_interact_times, 
                                                 input_src_node_pred_feature, input_src_node_pred_interact_times)
        if src_quantile_mapping:
            src_scores = quantile_mapping_func(src_scores, all_history_src_node_degrees)
        
        # 将log1p转换的度量转换回原始度量
        all_predicted_degrees = np.expm1(torch.cat(all_predicted_degrees).squeeze(-1).transpose(1,0).numpy())
        all_target_degrees = np.expm1(torch.cat(all_target_degrees).squeeze(-1).transpose(1,0).numpy()) 
        
        all_predicted_degrees = torch.tensor(all_predicted_degrees)
        all_target_degrees = torch.tensor(all_target_degrees)
        
        if pred_edge_num is not None:
            src_degrees = src_scores / src_scores.sum(dim=0)
            
        else: 
            src_degrees = src_scores # [B]
        
        src_degrees = (src_degrees+0.5).int()    
        
        input_edges = [np.vstack(input_edges[node]) for node in range(len(input_edges))]
        target_edges = [np.vstack(target_edges[node]) for node in range(len(target_edges))]
        
        # 处理input_edges和pred_edges，使其长度统一为self.candidate_dst_num
        processed_input_edges, input_edges_length = pad_edges(input_edges, self.candidate_dst_num)
        processed_target_edges, target_edges_length = pad_edges(target_edges, self.candidate_dst_num)
        
        # 将处理后的边转换为张量
        input_edges_tensor = torch.tensor(np.stack(processed_input_edges), dtype=torch.float32)
        target_edges_tensor = torch.tensor(np.stack(processed_target_edges), dtype=torch.float32)
        edges = self.dst_node_generator.sample_edges(input_edges_tensor, input_edges_length, 
                                        src_degrees, src_ids)
        
        return edges
            
            
        
class EdgeGenerator(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 max_node_num,
                 input_len,
                 timesteps = 100,
                 beta_start = 0.0001,
                 beta_end = 0.02,
                 w = 0.0,
                 dropout = 0.1,
                 num_predictions = 3,  # 可以根据需要调整这个值
                 device = torch.device('cuda:0')):
        super(EdgeGenerator, self).__init__()
        self.num_predictions = max(num_predictions, 1)
        self.model = Tenc(feature_dim,
                        max_node_num,
                        input_len,
                        dropout=dropout,
                        diffuser_type='mlp1', 
                        device=device
                        )
        self.diff = diffusion(timesteps, beta_start, beta_end, w)
        
    def sample_edges(self,
                      input_states,
                      input_len,
                      src_degrees,
                      src_ids):
        
        
        max_degree = src_degrees.max().item()
        # 计算需要重复预测的次数
       
        scores_all = []
        
        # 进行多次预测
        for _ in range(self.num_predictions):
            # 计算得分
            scores = self.model.predict(input_states, input_len, self.diff)
            scores_all.append(scores)
            
            
        scores_all = torch.cat(scores_all, dim=1).mean(dim=1) # [B]
        # 对每个节点采样top_k个目标节点, predicted_dst_node_id落在 N+1， 0为padding
        _, predicted_dst_node_ids = torch.topk(scores_all, k=max_degree, dim=1)  # [B, max_degree]
        
        # 创建一个范围矩阵，用于比较
        batch_size = src_degrees.shape[0]
        range_matrix = torch.arange(max_degree, device=src_degrees.device).unsqueeze(0).expand(batch_size, -1)
        
        # 使用广播操作创建掩码，避免循环
        dst_mask = range_matrix < src_degrees.unsqueeze(1)
        # 获取所有有效的目标节点位置
        valid_indices = dst_mask.nonzero(as_tuple=True)
        # 根据有效索引获取对应的源节点ID和目标节点ID
        src_nodes = src_ids[valid_indices[0]]
        dst_nodes = predicted_dst_node_ids[valid_indices] - 1 
        
        # 直接构建所有有效边
        all_valid_edges = torch.stack([src_nodes, dst_nodes], dim=1)  # [E, 2]
        
        
        return all_valid_edges
    
    
        
    def forward(self,
                input_states,
                input_len,
                target_states):
    
        
        # 计算起始向量和隐藏状态
        x_start = self.model.cacu_x(target_states)
        h = self.model.cacu_h(input_states, input_len, p=0.0)
        
        # 随机选择时间步长
        batch_size_new = input_states.shape[0]
        n = torch.randint(0, self.diff.timesteps, (batch_size_new,), device=self.model.device).long()
        
        # 计算损失
        dst_loss, predicted_x = self.diff.p_losses(self.model, x_start, h, n, loss_type='l2')
    
        return dst_loss
    
    def cacu_hit_k(self,
                   candidate_x,
                   predicted_x, 
                   target_x,
                   k = 10):
        
        # 计算预测的度数
        predicted_degrees = torch.sum(predicted_x, dim=1)
        # 计算目标的度数
        target_degrees = torch.sum(target_x, dim=1)
        
        