import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from .models.EdgeBank import edge_bank_link_prediction
from .utils.metrics import get_link_prediction_metrics, get_edge_classification_metrics, get_retrival_metrics
from .utils.utils import set_random_seed
from .utils.utils import NegativeEdgeSampler, NeighborSampler
# from .utils.metrics import mmd_degree_metrics
from .utils.DataLoader import Data
from .models.loss import HistogramMatchingLoss

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


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics




def quantile_mapping_func(generated_data, target_dis_quantile):
    """
    将生成的数据通过分位数映射到目标分布
    
    参数:
    generated_data: 生成的数据
    target_dis_quantile: N*T-1*1的目标度数矩阵
    
    返回:
    映射后的度数值
    """
    import numpy as np
    from scipy.stats import rankdata
    
    # 将目标分布展平为一维数组
    target_flat = target_dis_quantile.reshape(-1).numpy()
    generated_data = generated_data.reshape(-1).numpy()
    # 对生成数据进行排序并获取排名
    generated_ranks = rankdata(generated_data) / len(generated_data)
    
    # 对目标分布进行排序
    sorted_target = np.sort(target_flat)
    
    # 通过线性插值将生成数据映射到目标分布
    # 根据生成数据的排名在排序后的目标分布中查找对应位置的值
    mapped_data = np.interp(generated_ranks, 
                           np.linspace(0, 1, len(sorted_target)), 
                           sorted_target)
    
    return torch.tensor(mapped_data, dtype=target_dis_quantile.dtype, 
                       device=target_dis_quantile.device).unsqueeze(0).int()



def evaluate_model_node_regression_v2(model_name: str, 
                                   model: nn.Module, 
                                   evaluate_data: any, 
                                   loss_func: nn.Module,
                                   args,
                                   mode: str = 'val', # val or test
                                   quantile_mapping: bool = False,
                                   convert_degree_int: bool = False,
                                   save_degree_path: str = None):
    """
    评估节点回归模型在验证数据上的性能
    """
    model.eval()
    # 计算MMD距离 - 使用与degree_bwr.py相同的方式
    from .utils.mmd_metrics import compute_mmd_rbf
    from .utils.bwr_ctdg import custom_collate
    pred_len = evaluate_data.pred_len
    with torch.no_grad():
        # 存储评估损失和指标
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader = DataLoader(evaluate_data, 
                                            batch_size=100, 
                                            shuffle=False,
                                            collate_fn=custom_collate
                                            )
        
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        # 存储所有批次的预测和目标值，用于计算MMD距离
        all_src_node_ids = []
        all_predicted_degrees = []
        all_target_degrees = []
        all_predicted_unique_degrees = []
        all_target_unique_degrees = []
        all_predict_history_degrees = []
        all_predict_history_unique_degrees = []
        
        for batch_idx, batch_data in enumerate(evaluate_idx_data_loader_tqdm):
            batch_loss = 0
            
            # 获取批次数据，与训练逻辑保持一致
            batch_src_node_feature = batch_data["src_node_feature"]
            batch_src_node_interact_times = batch_data["src_interact_times"]
            batch_src_pred_time = batch_data["src_pred_time"]
            batch_src_node_degree = batch_data["src_node_degree"]
            batch_src_node_unique_degree = batch_data["src_node_unique_degree"]
            
            # 转换为张量并移动到设备
            batch_src_node_feature = torch.tensor(batch_src_node_feature, dtype=torch.float32).to(args.device)
            batch_src_node_interact_times = torch.tensor(batch_src_node_interact_times, dtype=torch.float32).to(args.device)
            batch_src_pred_time = torch.tensor(batch_src_pred_time, dtype=torch.float32).to(args.device)
            batch_src_node_degree = torch.tensor(batch_src_node_degree, dtype=torch.float32).to(args.device)
            batch_src_node_unique_degree = torch.tensor(batch_src_node_unique_degree, dtype=torch.float32).to(args.device)
            
            # 使用与训练相同的模型输入参数
            pred_output = model.forward_encoder(batch_src_node_feature, 
                                                batch_src_node_interact_times, 
                                                batch_src_pred_time)
            
            all_src_node_ids.append(batch_data["src_node_ids"])
            # 计算损失
            loss_degree = loss_func(pred_output[:,:,:pred_len], torch.log1p(batch_src_node_degree))
            loss_unique = loss_func(pred_output[:,:,pred_len:], torch.log1p(batch_src_node_unique_degree))
            loss = loss_degree + loss_unique
            
            batch_loss += loss
            # 存储预测和目标值，用于计算MMD距离
            all_predicted_degrees.append(pred_output[:,:,:pred_len].cpu())
            all_target_degrees.append(torch.log1p(batch_src_node_degree).cpu())
            all_predicted_unique_degrees.append(pred_output[:,:,pred_len:].cpu())
            all_target_unique_degrees.append(torch.log1p(batch_src_node_unique_degree).cpu())
            
            all_predict_history_degrees.append(torch.expm1(batch_src_node_feature[:,:,:evaluate_data.input_len]).cpu())

            all_predict_history_unique_degrees.append(torch.expm1(batch_src_node_feature[:,:,
                                                        evaluate_data.input_len:2*evaluate_data.input_len]).cpu())
            
            loss = batch_loss
            
            evaluate_losses.append(loss.item())
            
            # 计算评估指标
            evaluate_metrics.append({
                'total_loss': loss.item()
            })

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')
            
    # 计算所有批次的损失和MMD距离
    evaluate_losses_allbatches = loss_func(torch.cat(all_predicted_degrees), 
                                           torch.cat(all_target_degrees))
    evaluate_losses_allbatches_unique = loss_func(torch.cat(all_predicted_unique_degrees), 
                                                  torch.cat(all_target_unique_degrees))
    
    # 将log1p转换的度量转换回原始度量
    all_src_node_ids = np.concatenate(all_src_node_ids).flatten()
    # src_first mask: select unique src id first occurrence
    _, src_first_indices = np.unique(all_src_node_ids, return_index=True)
    all_src_node_ids = all_src_node_ids[src_first_indices]
    assert len(all_src_node_ids) == evaluate_data.ctdg.src.max() + 1

    all_predicted_degrees = np.expm1(torch.cat(all_predicted_degrees).view(-1, pred_len)[src_first_indices].flatten().unsqueeze(0).numpy())
    all_target_degrees = np.expm1(torch.cat(all_target_degrees).view(-1, pred_len)[src_first_indices].flatten().unsqueeze(0).numpy())
    all_predicted_unique_degrees = np.expm1(torch.cat(all_predicted_unique_degrees).view(-1, pred_len)[src_first_indices].flatten().unsqueeze(0).numpy())
    all_target_unique_degrees = np.expm1(torch.cat(all_target_unique_degrees).view(-1, pred_len)[src_first_indices].flatten().unsqueeze(0).numpy())
    all_predicted_degrees = torch.tensor(all_predicted_degrees)
    all_target_degrees = torch.tensor(all_target_degrees)
    all_predicted_unique_degrees = torch.tensor(all_predicted_unique_degrees)
    all_target_unique_degrees = torch.tensor(all_target_unique_degrees)
    all_predict_history_degrees = torch.cat(all_predict_history_degrees)
    all_predict_history_unique_degrees = torch.cat(all_predict_history_unique_degrees)
    if quantile_mapping:
        all_predicted_degrees = quantile_mapping_func(all_predicted_degrees,all_predict_history_degrees) 
        all_predicted_unique_degrees = quantile_mapping_func(all_predicted_unique_degrees,all_predict_history_unique_degrees) 
    
    if args.rescale:
        rescale_factor = all_target_degrees.sum() / all_predicted_degrees.sum()
        all_predicted_degrees = all_predicted_degrees * rescale_factor

    if convert_degree_int:
        all_predicted_degrees = all_predicted_degrees.int()
        all_target_degrees = all_target_degrees.int()
        all_predicted_unique_degrees = all_predicted_unique_degrees.int()
        all_target_unique_degrees = all_target_unique_degrees.int()

    
        


    # 计算所有预测度和目标度之间的MAE和MSE
    mae_mse_metrics = calculate_mae_mse(all_predicted_degrees, all_target_degrees, name = "degree")
    mae_mse_metrics_unique = calculate_mae_mse(all_predicted_unique_degrees, all_target_unique_degrees, name = "unique")
    
    histogram = HistogramMatchingLoss(num_bins = 100, 
                                      use_wasserstein=True, 
                                      use_kl=True,
                                      use_mmd=True,
                                      kernel_type='rbf')
    histogram_loss = histogram.forward(all_predicted_degrees, all_target_degrees, name = "degree")
    histogram_loss_unique = histogram.forward(all_predicted_unique_degrees, all_target_unique_degrees, name = "unique")
    
    evaluate_metrics_all = {
        'total_loss_allbatches_degree': evaluate_losses_allbatches.item(),
        'total_loss_allbatches_unique': evaluate_losses_allbatches_unique.item(),
        **mae_mse_metrics,
        **mae_mse_metrics_unique,
        **histogram_loss,
        **histogram_loss_unique
    }
    if save_degree_path is not None:
        # 保存目标度数据到文件
        degrees_map = {
            "degree": all_predicted_degrees.reshape(-1, pred_len), 
            "unique_degree": all_predicted_unique_degrees.reshape(-1, pred_len)
        }
        torch.save(degrees_map, save_degree_path)
        
    return evaluate_losses, evaluate_metrics, evaluate_metrics_all
            
    
    
    
def evaluate_model_retrival(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                _, all_batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=100,
                                                                                batch_src_node_ids=batch_src_node_ids,
                                                                                batch_dst_node_ids=batch_dst_node_ids,
                                                                                current_batch_start_time=batch_node_interact_times[0],
                                                                                current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, all_batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=100)
            batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            all_batch_neg_src_node_embeddings, all_batch_neg_dst_node_embeddings = [], []
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                
                for i in range(len(all_batch_neg_dst_node_ids)):
                    batch_neg_dst_node_ids = all_batch_neg_dst_node_ids[i].repeat(len(batch_neg_src_node_ids)) # batch_size
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=num_neighbors)
                    all_batch_neg_src_node_embeddings.append(batch_neg_src_node_embeddings) # neg_size | batch_size*feat_dim
                    all_batch_neg_dst_node_embeddings.append(batch_neg_dst_node_embeddings)
                
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                for i in range(len(all_batch_neg_dst_node_ids)):
                    batch_neg_dst_node_ids = all_batch_neg_dst_node_ids[i].repeat(len(batch_neg_src_node_ids)) # batch_size
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        edge_ids=None,
                                                                        edges_are_positive=False,
                                                                        num_neighbors=num_neighbors)
                    all_batch_neg_src_node_embeddings.append(batch_neg_src_node_embeddings) # neg_size | batch_size*feat_dim
                    all_batch_neg_dst_node_embeddings.append(batch_neg_dst_node_embeddings)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                for i in range(len(all_batch_neg_dst_node_ids)):
                    batch_neg_dst_node_ids = all_batch_neg_dst_node_ids[i].repeat(len(batch_neg_src_node_ids)) # batch_size
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=num_neighbors,
                                                                        time_gap=time_gap)
                    all_batch_neg_src_node_embeddings.append(batch_neg_src_node_embeddings) # neg_size | batch_size*feat_dim
                    all_batch_neg_dst_node_embeddings.append(batch_neg_dst_node_embeddings)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                for i in range(len(all_batch_neg_dst_node_ids)):
                    batch_neg_dst_node_ids = all_batch_neg_dst_node_ids[i].repeat(len(batch_neg_src_node_ids)) # batch_size
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times)
                    all_batch_neg_src_node_embeddings.append(batch_neg_src_node_embeddings) # neg_size | batch_size*feat_dim
                    all_batch_neg_dst_node_embeddings.append(batch_neg_dst_node_embeddings)
                
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid() # batch_size
            negative_probabilities = [model[1](input_1=all_batch_neg_src_node_embeddings[i], input_2=all_batch_neg_dst_node_embeddings[i]).squeeze(dim=-1).sigmoid() for i in range(len(all_batch_neg_dst_node_embeddings))] # neg_size | batch_size

            predicts = torch.cat([positive_probabilities, negative_probabilities[0]], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities[0])], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_retrival_metrics(positive_probabilities, negative_probabilities))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_model_edge_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the edge classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            predicts = model[1](x_1=batch_src_node_embeddings, x_2 = batch_dst_node_embeddings, rel_embs = model[0].edge_raw_features)
            pred_labels = torch.max(predicts, dim=1)[1]
            labels = torch.from_numpy(batch_labels).int().type(torch.LongTensor).to(predicts.device)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(pred_labels)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_edge_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.data_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.data_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.data_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                _, all_batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=100,
                                                                            batch_src_node_ids=batch_src_node_ids,
                                                                            batch_dst_node_ids=batch_dst_node_ids,
                                                                            current_batch_start_time=batch_node_interact_times[0],
                                                                            current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, all_batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=100)
            batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = [(batch_neg_src_node_ids, all_batch_neg_dst_node_ids[i].repeat(len(batch_neg_src_node_ids))) for i in range(len(all_batch_neg_dst_node_ids))]

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            all_negtive_probabilities = []
            for i in range(len(negative_edges)):
                positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                        positive_edges=positive_edges,
                                                                                        negative_edges=negative_edges[i],
                                                                                        edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                        time_window_mode=args.time_window_mode,
                                                                                        time_window_proportion=args.test_ratio)
                all_negtive_probabilities.append(negative_probabilities)

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, all_negtive_probabilities[0]])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(all_negtive_probabilities[0]))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_retrival_metrics(positive_probabilities, all_negtive_probabilities))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.data_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
