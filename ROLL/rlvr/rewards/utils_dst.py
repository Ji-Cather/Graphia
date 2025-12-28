import re
import json
from typing import Optional, List, Union, Sequence, Dict
from abc import ABCMeta
import logging
import numpy as np
import torch.nn as nn
from roll.distributed.scheduler.protocol import DataProto

from LLMGGen.load_gnn_judger import compute_src_dsts_score
from LLMGGen.utils.bwr_ctdg import (BWRCTDGALLDataset, 
                                    BWRCTDGDataset,
                                    Dataset_Template)
from .utils_parser import *


# from roll.models.model_providers import default_tokenizer_provider, default_model_provider


        
def apply_positional_weighting(gnn_rewards, dx_src, weight_inside=1.0, weight_outside=0.3):
    """
    对 gnn_rewards 根据节点位置加权：
        - 前 dx_src 个节点：乘以 weight_inside（如 1.0）
        - 后面的节点：乘以 weight_outside（如 0.3）
    """
    num_can = len(gnn_rewards)
    weights = torch.ones(num_can, device=gnn_rewards.device)
    
    if dx_src > 0:
        weights[:dx_src] = weight_inside   # 高权重
    if dx_src < num_can:
        weights[dx_src:] = weight_outside # 低权重

    return gnn_rewards * weights




from typing import Iterable
def execute_search_dst_toolkit(
                           query_text: str,
                           dx_src: int,
                           dst_node_ids: np.ndarray,
                           src_id: int,
                           bert_embedder: BertEmbedder, 
                           environment_info: dict,
                           environment_data:BWRCTDGDataset,
                           logger: logging.Logger,
                           interaction_cache:dict = {},
                           filter_rule = None,
                           recall_common_neighbor:bool = False,
                           recall_inductive: bool = False,
                           recall_alpha:int = 3,
                           recall_topk:int = None # activate when set
                           ):
    if recall_topk is None:
        recall_number = recall_alpha*dx_src
    else:
        recall_number = int(recall_topk) if recall_topk > dx_src else int(dx_src)

    try:
        logger.debug(f"query_text: {query_text}")
        logger.debug(f"environment_data: {type(environment_data)}")
        logger.debug(f"embedder: {type(bert_embedder)}")
        query_embedding = bert_embedder.get_embedding(query_text)
        
        if filter_rule is not None and filter_rule != "None":
            # 对目标节点进行过滤
            try:
                filter_rule_func = eval(filter_rule)
                
                filtered_dst_ids = []
                for dst_id in dst_node_ids:
                    # 构建包含节点指标的字典
                    node_metrics = {
                        'SF': interaction_cache[dst_id]['frequency'],
                        'AFN': int(np.mean(list(interaction_cache[dst_id]['count'].values()))) if interaction_cache[dst_id]['count'] else 0,
                        'HI': interaction_cache[src_id]['count'][dst_id] if dst_id in interaction_cache[src_id]['count'] else 0,
                        'CN': len(interaction_cache[dst_id]['neighbors'])
                    }
                    
                    
                    # 如果满足过滤规则则保留该节点
                    for metric, eval_value in filter_rule_func.items():
                        try:
                            if eval(f"{node_metrics[metric]}{eval_value}"):
                                filtered_dst_ids.append(dst_id)
                                break
                        except Exception as e:
                            pass
                        
                # 更新目标节点列表为过滤后的节点
                if len(filtered_dst_ids) >= dx_src:
                    dst_node_ids = np.array(filtered_dst_ids)
                else:
                    pass
            except Exception as e:
                pass

        # 不考虑自环
        if src_id in dst_node_ids:
            # 使用布尔索引移除源节点
            mask = dst_node_ids != src_id
            dst_node_ids = dst_node_ids[mask]
            
        node_features = torch.tensor(environment_data.node_feature)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, node_features[dst_node_ids])
        top_k = min(recall_number, len(similarities))
        candidate_dst_r_ids = torch.topk(similarities, k=top_k).indices
        candidate_dst_ids = dst_node_ids[candidate_dst_r_ids].tolist()
        if isinstance(candidate_dst_ids, int):
            candidate_dst_ids = [candidate_dst_ids]

        if recall_common_neighbor:
            n_dst_node_ids = np.array(list(interaction_cache[src_id]['neighbors']))
            node_features = torch.tensor(environment_data.node_feature)
            n_similarities = torch.nn.functional.cosine_similarity(query_embedding, 
            node_features[n_dst_node_ids])
            n_top_k = min(recall_number, len(n_similarities))
            candidate_dst_r_n_ids = torch.topk(n_similarities, k=n_top_k).indices
            candidate_dst_n_ids = n_dst_node_ids[candidate_dst_r_n_ids.tolist()].tolist()
            # 保持相对顺序去重，先合并，再去重
            merged_ids = list(candidate_dst_n_ids) + list(candidate_dst_ids)
            seen = set()
            candidate_dst_ids = [x for x in merged_ids if not (x in seen or seen.add(x))]
            
        if recall_inductive:
            ind_dst_node_ids = np.where(environment_data.new_node_mask)[0]
            ind_node_features = torch.tensor(environment_data.node_feature[environment_data.new_node_mask])
            ind_similarities = torch.nn.functional.cosine_similarity(query_embedding, ind_node_features)
            ind_top_k = min(recall_number, len(ind_similarities))
            ind_candidate_dst_r_ids = torch.topk(ind_similarities, k=ind_top_k).indices
            ind_candidate_dst_ids = ind_dst_node_ids[ind_candidate_dst_r_ids.tolist()].tolist()
            # 保持相对顺序去重，先合并，再去重
            merged_ids = list(ind_candidate_dst_ids) + list(candidate_dst_ids)
            seen = set()
            candidate_dst_ids = [x for x in merged_ids if not (x in seen or seen.add(x))]

        if isinstance(candidate_dst_ids, Iterable) and len(candidate_dst_ids) > top_k:
            candidate_dst_ids = candidate_dst_ids[:top_k]
        
        candidate_dst_metrics = environment_data.get_dst_nodes_texts(
                                                src_id,
                                                candidate_dst_ids, 
                                                interaction_cache = interaction_cache)
        
        
        return {
            "dst_ids": candidate_dst_ids,
            "dst_metrics": candidate_dst_metrics
        }
        
    except Exception as e:
        logger.error(f"execution error: {e}")


def preprocess_candidate_set(candidate_dst_ids, ground_truth_dst_ids):
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidate_ids = []
    for id in candidate_dst_ids:
        if id not in seen:
            seen.add(id)
            unique_candidate_ids.append(id)
    
    # Cut off to the size of ground_truth_dst_ids
    cutoff_size = len(set(ground_truth_dst_ids))
    if cutoff_size < len(unique_candidate_ids):
        preprocessed_candidate_set = set(unique_candidate_ids[:cutoff_size])
    else:
        preprocessed_candidate_set = set(unique_candidate_ids)
    return preprocessed_candidate_set


def curriculum_reward(t, r_estimate, r_gt):
    """
    Args:
        t: 当前训练步数
        r_estimate: LLM估计的奖励值
        r_gt: 真实奖励值（可能为空）
        T1: 阶段1结束步数
        T2: 阶段2结束步数
    Returns:
        total_reward: 动态加权后的奖励值
    """
    # 阶段划分
    alpha_gt = max(min(0.01 * t, 5), 1)  # 线性增长，上限为5，下限为1

    # 如果GT奖励为空（如稀疏信号未触发），仅使用estimate_gt
    if r_gt is None:
        total_reward = r_estimate
    else:
        total_reward = r_estimate + alpha_gt * r_gt
    
    return total_reward

def curriculum_reward_reverse(t, r_estimate, r_gt):
    """
    Args:
        t: 当前训练步数
        r_estimate: LLM估计的奖励值
        r_gt: 真实奖励值（可能为空）
        T1: 阶段1结束步数
        T2: 阶段2结束步数
    Returns:
        total_reward: 动态加权后的奖励值
    """
    # 阶段划分
    alpha_gt = 6 - max(min(0.01 * t, 5), 1)  # 线性增长，上限为5，下限为1

    # 如果GT奖励为空（如稀疏信号未触发），仅使用estimate_gt
    if r_gt is None:
        total_reward = r_estimate
    else:
        total_reward = r_estimate + alpha_gt * r_gt
    
    return total_reward

class DstMapper:

    def __init__(self, 
                data_load_args,
                reward_args
                ):
        bwr_ctdg = BWRCTDGALLDataset(
                    pred_ratio=data_load_args.pred_ratio,
                    bwr=data_load_args.bwr,
                    time_window=data_load_args.time_window,
                    root=data_load_args.root,
                    use_feature=data_load_args.use_feature,
                    cm_order=data_load_args.cm_order,
                    force_reload=data_load_args.force_reload
                )
        self.environment_data = {
            "train":bwr_ctdg.train_data,
            "val": bwr_ctdg.val_data,
            "test": bwr_ctdg.test_data,
            "full": bwr_ctdg.data
        } # tag : tag environment data

        
        
        self.environment_info = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description'],
            'node_text_cols': Dataset_Template[bwr_ctdg.data_name]['node_text_cols'],
            'node_text_template': Dataset_Template[bwr_ctdg.data_name]['node_text_template'],
            'required_keys': [
                            *Dataset_Template[bwr_ctdg.data_name]['node_text_cols']]
            # loose version
            }

        self.res_parser:RegexTaggedContentParser = RegexTaggedContentParser(
            required_keys=self.environment_info["required_keys"])

        self.reward_args = reward_args

    def get_link_prediction_data(self):
        node_raw_features = self.environment_data["train"].node_feature
        
        edge_raw_features = self.environment_data["full"].edge_feature
        
        assert edge_raw_features.shape[0] == self.environment_data["full"].ctdg.edge_id_all.shape[0]


        full_data_temp = self.environment_data["full"].ctdg
        train_data_temp = self.environment_data["train"].ctdg
        val_data_temp = self.environment_data["val"].ctdg
        test_data_temp = self.environment_data["test"].ctdg

        
        full_data = Data(full_data_temp.src.cpu().numpy(), 
                        full_data_temp.dst.cpu().numpy(), 
                        full_data_temp.t.cpu().numpy(), 
                        full_data_temp.edge_id_all.cpu().numpy(), 
                        full_data_temp.label.cpu().numpy(), 
                        )
        edge_ids_seen = set(train_data_temp.edge_id_all.cpu().numpy().tolist())
        train_data = Data(
            train_data_temp.src.cpu().numpy(),
            train_data_temp.dst.cpu().numpy(),
            train_data_temp.t.cpu().numpy(),
            train_data_temp.edge_id_all.cpu().numpy(),
            train_data_temp.label.cpu().numpy(), 
        )
        val_edge_ids = val_data_temp.edge_id_all.cpu().numpy()
        edge_unseen_mask = torch.tensor(np.array([eid not in edge_ids_seen for eid in val_edge_ids]))

        val_data = Data(
            val_data_temp.src.cpu().numpy()[edge_unseen_mask],
            val_data_temp.dst.cpu().numpy()[edge_unseen_mask],
            val_data_temp.t.cpu().numpy()[edge_unseen_mask],
            val_data_temp.edge_id_all.cpu().numpy()[edge_unseen_mask],
            val_data_temp.label.cpu().numpy()[edge_unseen_mask],
        )

        new_node_val_data = Data(
            val_data_temp.src[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.dst[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.t[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.edge_id_all[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.label[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
        )
        edge_ids_seen = set(np.concatenate([train_data_temp.edge_id_all.cpu().numpy(),
                                            val_data_temp.edge_id_all.cpu().numpy()
                                            ]).tolist())
        test_edge_ids = test_data_temp.edge_id_all.cpu().numpy()
        edge_unseen_mask = torch.tensor(np.array([eid not in edge_ids_seen for eid in test_edge_ids]))

        test_data = Data(
            test_data_temp.src.cpu().numpy()[edge_unseen_mask],
            test_data_temp.dst.cpu().numpy()[edge_unseen_mask],
            test_data_temp.t.cpu().numpy()[edge_unseen_mask],
            test_data_temp.edge_id_all.cpu().numpy()[edge_unseen_mask], 
            test_data_temp.label.cpu().numpy()[edge_unseen_mask],
        )
        
        new_node_test_data = Data(test_data_temp.src[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),  
                        test_data_temp.dst[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.t[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.edge_id_all[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(), 
                        test_data_temp.label[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        )
        label_num = np.unique(full_data.labels).shape[0]
        return node_raw_features, edge_raw_features, full_data, train_data, \
        val_data, test_data, new_node_val_data, new_node_test_data, label_num

    def parse_response(self, 
                       text_response: str,
                       logger):
        reward_dict = {}
        parsed = {}
        try:
            parsed = self.res_parser.parse(ModelResponse(text_response)).parsed
            query_text = self.environment_info['node_text_template'].format_map(parsed)
            parsed["query_text"] = query_text
            parsed["filter_rule"] = parsed.get("filter_rule", None)
            reward_dict["format_reward"] = 1    
            try:
                candidate_dst_ids = eval(parsed.get("candidate_dst_ids","[]"))
                assert isinstance(candidate_dst_ids, list)
                candidate_dst_ids = [int(x) for x in candidate_dst_ids]
            except:
                candidate_dst_ids = []
            parsed["candidate_dst_ids"] = candidate_dst_ids
        except Exception as e:
            logger.error(f"parse error: {e}")
            parsed = {"query_text": "",
                        "filter_rule": None}
            reward_dict["format_reward"] = 0
        parsed["ori_text"] = text_response
        return parsed, reward_dict

    
    

    def _collect_reward(self, 
                        rewards: dict,
                        response: dict,
                        bert_embedder: BertEmbedder,
                        logger) -> dict:
        
        cols_text = self.environment_info['node_text_cols']
        if response["query_text"]!= "":
            # required
            
            query_text = response["query_text"] # dict
            filter_rule = response['filter_rule']
            src_idx = response["src_idx"]
            dx_src, gt_dst_idxs = response["gt_dx_src_unique"], response["gt_dst_idxs_unique"]
            tag = response["tag"]

            assert tag in ["train", "val", "test"], "tag should be within [train, val, test]"

            sim_mean, sim_std = self.environment_data[tag].cal_sim()
            dst_node_ids = np.arange(self.environment_info['dst_min'], 
                                    self.environment_info['dst_max'] + 1)
            
            
            execution_res = execute_search_dst_toolkit(
                    query_text = query_text, 
                    dx_src = dx_src, 
                    dst_node_ids = dst_node_ids, 
                    src_id = src_idx, 
                    bert_embedder=bert_embedder,
                    environment_info = self.environment_info,
                    environment_data = self.environment_data[tag], 
                    logger = logger,
                    interaction_cache = self.environment_data[tag].interaction_cache,
                    filter_rule=filter_rule,
                    recall_common_neighbor=True,
                    recall_alpha = self.reward_args.recall_alpha,
                    recall_topk=self.reward_args.recall_topk
                )
            
            candidate_dst_texts = execution_res["dst_metrics"]
            candidate_dst_ids = execution_res["dst_ids"]
            # llm_candidate_dst_ids = response["candidate_dst_ids"]
            # candidate_dst_ids = list(set(candidate_dst_ids) | set(llm_candidate_dst_ids))
            rewards_cal = calculate_reward_value(
                            candidate_dst_ids,
                            gt_dst_idxs,
                            self.environment_data[tag],
                            sim_mean,
                            sim_std,
                            self.reward_args.alpha,
                        )
            response["candidate_dst_ids"] = candidate_dst_ids
            response["candidate_dst_texts"] = candidate_dst_texts
            
        else:
            logger.error("_collect_reward")
            response["candidate_dst_ids"] = []
            response["candidate_dst_texts"] = ""
            rewards_cal = {
                "overlap_reward" : 0,
                "hit_reward" : 0,
                "weight_reward": 0,
                "score": 0,
            }
        rewards.update(rewards_cal)
        return rewards, response

    def _collect_gnn_reward(self, 
                        rewards: dict,
                        response: dict,
                        bert_embedder: BertEmbedder,
                        logger,
                        model_name: str,
                        gnn_model: nn.Module,
                        global_step: int,
                        ) -> dict:
        rewards, response = self._collect_reward(rewards,
                                                response,
                                                bert_embedder,
                                                logger,
                                                )
        tag = response["tag"]
        assert tag in ["train", "val", "test"], "tag should be within [train, val, test]"
        num_can = len(response["candidate_dst_ids"])

        try:
            if len(response["candidate_dst_ids"]) > 0:
                future_interact_times = np.repeat(
                                            self.environment_data[tag].unique_times.reshape(1, -1), 
                                            num_can, axis=0)[:,-1]
                src_idx = response["src_idx"]
                src_ids = int(src_idx) * np.ones(num_can, dtype=int)
                dst_ids = np.array(response["candidate_dst_ids"])
                dx_src = int(response["gt_dx_src_unique"])

                assert len(dst_ids) == num_can, f"Dst ids for src {src_idx} is not equal to num_can, please check the input data."
                gnn_rewards = compute_src_dsts_score(
                        src_ids=src_ids,
                        dst_ids=dst_ids,
                        interact_times=future_interact_times,
                        model_name=model_name,
                        model=gnn_model,
                        model_type = "lp"
                )

                rewards["gnn_reward"] = torch.sum(gnn_rewards).item() # sum or mean ?
                rewards["gnn_reward"] = apply_positional_weighting(
                                                        gnn_rewards,
                                                        dx_src=dx_src,
                                                        weight_inside=1.0,     # 前 dx_src 个保持原样或更高
                                                        weight_outside=0.2     # 后面的大幅降权
                                                    ).sum().item()
                                                    
                gnn_curriculum_reward = curriculum_reward(
                    global_step,
                    rewards["gnn_reward"],
                    rewards["overlap_reward"]
                )
                gnn_curriculum_reward_reverse = curriculum_reward_reverse(
                    global_step,
                    rewards["gnn_reward"],
                    rewards["overlap_reward"]
                )
                rewards["gnn_curriculum_reward"] = gnn_curriculum_reward
                rewards["gnn_curriculum_reward_reverse"] = gnn_curriculum_reward_reverse
                
        except Exception as e:
            logger.error(f"curriculum error {e}")
            rewards["gnn_reward"] = 0
            rewards["gnn_curriculum_reward"] = 0
            rewards["gnn_curriculum_reward_reverse"] = 0
        
        rewards["gnn_weight_reward"] = rewards.get("gnn_reward",0) + 5 * rewards.get("overlap_reward",0)
        return rewards, response

    def _collect_seq_reward(self, 
                        response: dict,
                        bert_embedder: BertEmbedder,
                        logger,
                        alpha = 0.5) -> dict:
        cols_text = self.environment_info['node_text_cols']
        dst_node_ids = np.arange(self.environment_info['dst_min'], 
                                    self.environment_info['dst_max'] + 1)
        try:
            if response["query_text"]!= "":
                # required
                
                query_text = response["query_text"] # dict
                src_idx = response["src_idx"]
                gt_dst_idx = response["gt_dst_idx"]
                query_embedding = bert_embedder.get_embedding(query_text)


                # only support training on train set now.
                data = self.environment_data["train"]
                node_features = torch.tensor(self.environment_data["train"].node_feature)
                similarities = torch.nn.functional.cosine_similarity(query_embedding, node_features[dst_node_ids])

                # recall 10, only consider 1 dst
                candidate_dst_r_ids = torch.topk(similarities, k=10).indices
                candidate_dst_ids = dst_node_ids[candidate_dst_r_ids].tolist()
                overlap_reward = 1 if gt_dst_idx in candidate_dst_ids else 0
                r_sim_value = torch.mean(torch.tensor(data.node_feature[list(candidate_dst_ids)]) \
                @ torch.tensor(data.node_feature[[gt_dst_idx]].T))
        
                r_value = (1-alpha) * overlap_reward + alpha * r_sim_value 
        
                rewards = {
                    "overlap_reward": overlap_reward,
                    "LIKR_reward": r_value,
                    "score": overlap_reward
                }
                
            else:
                logger.error("_collect_reward")
                rewards = {
                    "overlap_reward": 0,
                    "LIKR_reward": 0,
                    "score": 0,
                }
        except:
            logger.error("_collect_reward")
            rewards = {
                    "overlap_reward": 0,
                    "LIKR_reward": 0,
                    "score": 0,
                }
        return rewards, response
    

    def _collect_reference_obs(self,
                                src_idx,
                               gt_dst_idxs_unique,
                               tag):
        environment_data = self.environment_data[tag]
        interaction_cache = environment_data.interaction_cache
        gt_dst_metrcis = environment_data.get_dst_nodes_texts(
                                                src_idx,
                                                gt_dst_idxs_unique, 
                                                interaction_cache = interaction_cache)
        return gt_dst_metrcis



## hit validation 
def calculate_reward_value(
    candidate_dst_ids: List[int],
    ground_truth_dst_ids: List[int],
    data: BWRCTDGDataset,
    sim_mean,
    sim_std,
    alpha = 0.5
):
    """
   $$
    R_j=R_i^{G T}+\\alpha R_i^{S I M} 
    R_i^{G T}=1 \\text { or } o ; R_i^{S I M}=\sum_{j=0}^d\\left\\langle\\mathbf{e}_i, \\mathbf{e}_j\\right\\rangle
    $$
    """
    
    # 计算带位置权重的overlap ratio
    # candidate_set = set(candidate_dst_ids)
    # candidate_set = preprocess_candidate_set(candidate_dst_ids, ground_truth_dst_ids)
    candidate_set = set(candidate_dst_ids)
    gt_set = set(ground_truth_dst_ids)
    overlap = candidate_set & gt_set
    
    # 计算加权overlap ratio
    r_gt_overlap_ratio = len(overlap) / len(gt_set) if len(gt_set) > 0 else 0
    
    if not candidate_set or not ground_truth_dst_ids:
        r_sim_value = sim_mean
    else:
        r_sim_value = torch.mean(torch.tensor(data.node_feature[list(candidate_set)]) \
            @ torch.tensor(data.node_feature[ground_truth_dst_ids].T))
    
    r_value = ((1-alpha) * 20 * r_gt_overlap_ratio + alpha * 20*((r_sim_value - sim_mean)/sim_std))
    
    # acc_ratio = len(overlap)/len(gt_set)

    calculate_reward = {
        "overlap_reward" : int(len(overlap)),
        "hit_reward" : int(len(overlap)>0),
        "weight_reward": int(r_value),
        "score": int(len(overlap)>0),
        "5_overlap_reward": 5*int(len(overlap)),
    }

    return calculate_reward




