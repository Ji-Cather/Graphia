import re
import json
from typing import Optional, List, Union, Sequence, Dict,Iterable
from abc import ABCMeta
import logging
import numpy as np
import torch.nn as nn

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.rewards.utils_parser import *                         


from Graphia.load_gnn_judger import compute_src_dsts_score
from Graphia.utils.bwr_ctdg import (BWRCTDGALLDataset, 
                                    BWRCTDGDataset,
                                    Dataset_Template)

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
    alpha_gt = max(min(0.01 * t, 5), 1)  # 线性增长，上限为2，下限为1

    # 如果GT奖励为空（如稀疏信号未触发），仅使用estimate_gt
    if r_gt is None:
        total_reward = r_estimate
    else:
        total_reward = r_estimate + alpha_gt * r_gt
    
    return total_reward

def execute_search_edge_toolkit(
                           edge_text: str,
                           src_id: int,
                           bert_embedder: BertEmbedder, 
                           environment_info: dict,
                           environment_data:BWRCTDGDataset,
                           logger: logging.Logger,
                           interaction_cache:dict = {},
                           recall_common_neighbor:bool = False):
    try:
        logger.debug(f"edge_text: {edge_text}")
        if edge_text== "":
            return {}
        edge_embedding = bert_embedder.get_embedding(edge_text)
     
            
        edge_features = torch.tensor(environment_data.edge_feature)
        similarities = torch.nn.functional.cosine_similarity(edge_embedding, edge_features)
        top_k = 10
        edge_ids = environment_data.ctdg.edge_id
        candidate_edge_r_ids = torch.topk(similarities, k=top_k).indices
        candidate_edge_ids = edge_ids[candidate_edge_r_ids].tolist()
        if isinstance(candidate_edge_ids, int):
            candidate_edge_ids = [candidate_edge_ids]

        if recall_common_neighbor:
            n_edge_ids = environment_data.src_edge_infos([src_id])[0]
            n_edge_ids = torch.vstack(n_edge_ids).flatten()
            if n_edge_ids.shape[0] > 0:
                n_similarities = torch.nn.functional.cosine_similarity(edge_embedding, 
                edge_features[n_edge_ids])
                n_top_k = min(top_k, len(n_similarities))
                candidate_edge_r_n_ids = torch.topk(n_similarities, k=n_top_k).indices
                candidate_edge_n_ids = n_edge_ids[candidate_edge_r_n_ids.tolist()].tolist()
                candidate_edge_ids = list(set(candidate_edge_n_ids) | set(candidate_edge_ids))
            

        if isinstance(candidate_edge_ids, Iterable) and len(candidate_edge_ids) > top_k:
            candidate_edge_ids = candidate_edge_ids[:top_k]
        
        if not isinstance(candidate_edge_ids, Union[list, np.ndarray, torch.Tensor]):
            candidate_edge_ids = [candidate_edge_ids] 
        
        edge_labels = environment_data.ctdg.label[candidate_edge_ids]
        unique_labels, counts = torch.unique(edge_labels, return_counts=True)
        top_freq_idx = torch.argmax(counts)
        top1_f_edge_label = unique_labels[top_freq_idx].item()

        return {
            "label": edge_labels[0].item(),
            "top1_f_label": top1_f_edge_label,
            "recalled_labels": edge_labels.tolist(),
        }
        
    except Exception as e:
        logger.error(f"execution error: {e}")
        return {}
def execute_search_edge_label_toolkit(
                           edge_label_text: str,
                           src_id: int,
                           bert_embedder: BertEmbedder, 
                           environment_info: dict,
                           environment_data:BWRCTDGDataset,
                           logger: logging.Logger,
                           interaction_cache:dict = {},
                           recall_common_neighbor:bool = False):
    try:
        logger.debug(f"edge_label_text: {edge_label_text}")
        if edge_label_text== "":
            return {}
        edge_label_embedding = bert_embedder.get_embedding(edge_label_text)
     
            
        label_features = torch.tensor(environment_data.label_feature)
        similarities = torch.nn.functional.cosine_similarity(edge_label_embedding, label_features)
        top_k = min(5, len(label_features))
        edge_label_ids = torch.arange(len(label_features))
       # Identify exact matches where similarity is 1
        exact_match_indices = torch.where((1.0-similarities)<1e-5)[0]
        select_label = None
        if len(exact_match_indices) > 0:
            # Prioritize exact matches
            select_label = exact_match_indices.tolist()[0]
        candidate_edge_label_ids = edge_label_ids[torch.topk(similarities, k=top_k).indices].tolist()
                

        if isinstance(candidate_edge_label_ids, int):
            candidate_edge_label_ids = [candidate_edge_label_ids]

        if recall_common_neighbor and len(exact_match_indices) == 0:
            n_edge_ids = environment_data.src_edge_infos([src_id])[0]
            n_edge_ids = torch.vstack(n_edge_ids).flatten()
            if n_edge_ids.shape[0] > 0:
                n_label_ids = environment_data.ctdg.label[n_edge_ids]
                n_similarities = torch.nn.functional.cosine_similarity(edge_label_embedding, 
                label_features[n_label_ids])
                n_top_k = min(top_k, len(n_similarities))
                candidate_edge_label_r_n_ids = torch.topk(n_similarities, k=n_top_k).indices
                candidate_edge_label_n_ids = n_label_ids[candidate_edge_label_r_n_ids.tolist()].tolist()
                if len(candidate_edge_label_n_ids) > 0 and select_label is None:
                    select_label = candidate_edge_label_n_ids[0]
                candidate_edge_label_ids =  list(set(candidate_edge_label_n_ids)| set(candidate_edge_label_ids))
            

        if isinstance(candidate_edge_label_ids, Iterable) and len(candidate_edge_label_ids) > top_k:
            candidate_edge_label_ids = candidate_edge_label_ids[:top_k]
        
        if not isinstance(candidate_edge_label_ids, Union[list, np.ndarray, torch.Tensor]):
            candidate_edge_label_ids = [candidate_edge_label_ids] 
        
        if select_label is None:
            select_label = candidate_edge_label_ids[0]
        assert select_label is not None, "error selecting label"
        unique_labels, counts = torch.unique(torch.tensor(candidate_edge_label_ids), return_counts=True)
        top_freq_idx = torch.argmax(counts)
        top1_f_edge_label = unique_labels[top_freq_idx].item()

        return {
            "label": select_label,
            "top1_f_label": top1_f_edge_label,
            "recalled_labels": candidate_edge_label_ids,
        }
        
    except Exception as e:
        logger.error(f"execution error: {e}")


def calculate_reward_value(
    label: int,
    gt_label: int,
    recalled_labels: list,
):
    """
   $$
    R_j=R_i^{G T}+\\alpha R_i^{S I M} 
    R_i^{G T}=1 \\text { or } o ; R_i^{S I M}=\sum_{j=0}^d\\left\\langle\\mathbf{e}_i, \\mathbf{e}_j\\right\\rangle
    $$
    """
    
    # 计算带位置权重的overlap ratio
    gt_acc = label == gt_label # 1\0
   

    calculate_reward = {
        "gt_reward" : int(gt_acc),
        "score": int(bool(gt_label in recalled_labels))
    }

    return calculate_reward


class EdgeMapper:

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
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description'],
            'edge_text_cols': Dataset_Template[bwr_ctdg.data_name]['edge_text_cols'],
            'edge_text_template': Dataset_Template[bwr_ctdg.data_name]['edge_text_template'],
            'required_keys': [
                            *Dataset_Template[bwr_ctdg.data_name]['edge_text_cols']],
            'goal': Dataset_Template[bwr_ctdg.data_name]['goal']
            # loose version without think
            }
        
        self.res_parser:RegexTaggedContentParser = RegexTaggedContentParser(
            required_keys=self.environment_info["required_keys"])
        
        self.label_res_parser:RegexTaggedContentParser = RegexTaggedContentParser(
            required_keys=["label"])

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
            edge_text = self.environment_info['edge_text_template'].format_map(parsed)
            parsed["edge_text"] = edge_text     
            reward_dict["format_reward"] = 1       
            logger.info(f"parsed: {parsed}")
        except Exception as e:
            parsed["edge_text"] = text_response
            reward_dict["format_reward"] = 0
            logger.error(f"parse fail: {text_response}")

        try:
            label_parsed = self.label_res_parser.parse(ModelResponse(text_response)).parsed
            parsed.update(label_parsed)
            parsed["label_text"] = label_parsed.get("label","")
            reward_dict["format_reward"] += 1       
            logger.info(f"label parsed: {parsed}")
        except Exception as e:
            parsed["label_text"] = ""
            logger.error(f"label parse fail: {text_response}")
        return parsed, reward_dict

    def _collect_gnn_reward(self, 
                        response: dict,
                        bert_embedder: BertEmbedder, 
                        model_name: str,
                        gnn_model: nn.Module,
                        global_step: int,
                        logger,
                        recall_query:str="label_text"
                        ) -> dict:

        src_ids = np.array([response["src_idx"]])
        dst_ids = np.array([response["dst_idx"]])
        tag = response["tag"]

        future_interact_times = np.repeat(
                            self.environment_data[tag].unique_times.reshape(1, -1), 
        1, axis=0)[:,-1] # set interaction time as the end prediction time
        
        try:
            # edge text as query
            if recall_query=="edge_text":
                edge_text = response["edge_text"]
                execution_res = execute_search_edge_toolkit(
                        edge_text = edge_text, 
                        src_id = int(src_ids[0]), 
                        bert_embedder = bert_embedder,
                        environment_info = self.environment_info,
                        environment_data = self.environment_data[tag], 
                        logger = logger,
                        interaction_cache = self.environment_data[tag].interaction_cache,
                        recall_common_neighbor=True,
                    )

            elif recall_query=="label_text":
                # label text as query
                label_text = response["label_text"]
                execution_res = execute_search_edge_label_toolkit(
                    edge_label_text=label_text,
                    src_id=int(src_ids[0]), 
                    bert_embedder=bert_embedder,
                    environment_info=self.environment_info,
                    environment_data=self.environment_data[tag], 
                    logger=logger,
                    interaction_cache=self.environment_data[tag].interaction_cache,
                    recall_common_neighbor=True,
                )
            
            response.update(execution_res)
            
            rewards = calculate_reward_value(
                            execution_res.get("label",-1),
                            response["gt_edge_label"],
                            execution_res.get("recalled_labels",[]),
                        )
            logger.info({
                "llm_edge_label": execution_res.get("label",-1),
                "gt_edge_label":response["gt_edge_label"]
            })
            
            if execution_res.get("label") is not None:
                gnn_pred_prob = compute_src_dsts_score(
                        src_ids=src_ids,
                        dst_ids=dst_ids,
                        interact_times=future_interact_times,
                        model_name=model_name,
                        model=gnn_model,
                        model_type = "ec")
                rewards["gnn_reward"] = gnn_pred_prob[0,int(execution_res["label"])].item()
                gnn_curriculum_reward = curriculum_reward(
                    global_step,
                    rewards["gnn_reward"],
                    rewards["gt_reward"]
                )
                rewards["gnn_curriculum_reward"] = gnn_curriculum_reward 
            else:
                rewards["gnn_reward"] = 0
                rewards["gnn_curriculum_reward"] = 0
            
            logger.debug(f"curriculum: {rewards}")
        
        except Exception as e:
            logger.error(f"curriculum error: {e}")
            rewards = {
                "format_reward": 0,
                "gt_reward": 0,
                "gnn_reward": 0,
                "gnn_curriculum_reward": 0,
                "llm_reward": 0,
                "score":0
            }
        return rewards, response



if __name__ == "__main__":
    import logging
    import torch
    from dacite import from_dict
    # Setup logger
    logger = logging.getLogger("edge_toolkit_test")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Initialize dataset
    dataset = BWRCTDGALLDataset(
        pred_ratio=0.15,
        bwr=1980,
        time_window=86400,
        root="Graphia/data/8days_dytag_small_text_en",
        use_feature="bert",
        cm_order=True,
    )
    
    # Initialize BERT embedder
    from roll.configs import ModelArguments
    bert_args = {
        "model_name_or_path": "hf_cache/bert-tiny",
    }
    bert_args = from_dict(ModelArguments, bert_args)
    bert_embedder = BertEmbedder(bert_args, logger)
    
    # Create environment info similar to EdgeMapper
    environment_info = {
        'data_name': dataset.data_name,
        "description": "Test environment",
        'edge_text_cols': ["text"],  # Replace with actual columns
        'edge_text_template': "{text}",
        'required_keys': ["text"]
    }
    
    # Create interaction cache (simplified example)
    interaction_cache = {
        4: {'neighbors': [1, 2, 3]},
        1: {'neighbors': [0, 2]}
    }
    
    # Test cases
    test_cases = [
        {"text": "example edge text 1", "label": "Hotpot sa", "src_id": 4, "recall_neighbor": True},
        {"text": "another edge example", "label": "Hotpot sauce.","src_id": 5, "recall_neighbor": False}
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i+1}: {test_case['text']}")
        print(f"{'='*50}")
        
        result = execute_search_edge_toolkit(
            edge_text=test_case['text'],
            src_id=test_case['src_id'],
            bert_embedder=bert_embedder,
            environment_info=environment_info,
            environment_data=dataset.train_data,  # Use train split
            logger=logger,
            interaction_cache=interaction_cache,
            recall_common_neighbor=test_case['recall_neighbor']
        )

        label_result = execute_search_edge_label_toolkit(
            edge_label_text=test_case['label'],
            src_id=test_case['src_id'],
            bert_embedder=bert_embedder,
            environment_info=environment_info,
            environment_data=dataset.train_data,  # Use train split
            logger=logger,
            interaction_cache=interaction_cache,
            recall_common_neighbor=test_case['recall_neighbor']
        )
        
        print(f"\nRESULT FOR CASE {i+1}:")
        print(f"Label: {result.get('label', 'N/A')}")
        