import torch
import os
import pandas as pd

from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from transformers import BertTokenizer, BertModel


from .utils.bwr_ctdg import (BWRCTDGALLDataset, 
                            BWRCTDGDataset, 
                            Dataset_Template)
from .utils.utils_parser import RegexTaggedContentParser, ModelResponse
   
from .utils.utils import get_neighbor_sampler
from .utils.DataLoader import get_link_prediction_data
from .load_gnn_judger import create_link_prediction_model,compute_src_dsts_score, create_edge_classification_model

class BertEmbedder:
    def __init__(self, model_name = "prajjwal1/bert-tiny"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

# 使用单例模式确保只加载一次模型

THINK_PREFIX = "/no_think " # for qwen3 disable think/ able think "/think"

QUERY_SYS_PROMPT = THINK_PREFIX + """You should act as a src node in the network. You are given a list of dst nodes and their node texts.
Your task is to predict the profile of dst nodes. You should think about the dst nodes you are going to interact with.
** Objective **
You should maximize the chances to retrieve desired dst nodes with your query text.
"""
    
EDGE_ATTR_PROMPT = THINK_PREFIX+"""
You should generate the edge attributes for the edge (relation/action between src and dst node). 
You should think about the edge attribute.
** Objective **
You should first predict the edge LABEL. Then generate the edge TEXT, consistent with src node history edges.
"""

ACTOR_JUDGE = """
/no_think 
You are an expert judge evaluating the quality of a response to a given prompt. Please evaluate the role-playing ability of the ACTOR NODE based on its actor actions consistency with reference actions.

The social goal for the actor is:
    '{goal}'

[Prompt]:
{prompt}

[ACTOR Action]:
{response}

[Reference Action]:
{reference}

Scoring Logic  
- GOAL Fulfillment(GF):
1 (Frequent mismatches with the goal),3 (Mostly aligned, with minor inconsistencies),5 (Fully aligned with the goal) 
- Contextual Fidelity(CF):  
1 (Frequent inconsistencies),3 (Minor inconsistencies),5 (Deep contextual mastery)  
- Personality Depth(PD):
1 (Contradictory traits),3 (Occasional deviations),5 (Nuanced embodiment)  
- Dynamic Adaptability(DA):  
1 (Rigid responses),3 (Context-dependent adaptation),5 (Creative innovation)  
- Immersive Quality(IQ):  
1 (Disruptive inconsistencies),3 (Minor immersion breaks),5 (Seamless portrayal)  
- Content Richness(CR):  
1 (Superficial/output),3 (Adequate detail),5 (Rich, layered interactions)  


Your response must follow the format provided below. Please note that only when the content quality is extremely good can 5 Points be given.

[Response Format]:
GF: [1-5]
CF: [1-5]  
PD: [1-5]  
DA: [1-5]  
IQ: [1-5]  
CR: [1-5]  


[Response]:
"""

def select_to_last_period(s, upper_token = 4e3):
    upper_token = int(upper_token)
    if len(s) <= upper_token:
        return s
    if upper_token <=0:
        return ""
    s = s[-upper_token:]
    # 查找最后一个句号的位置
    last_period_index = s.rfind('.')
    # 如果找到句号，返回从开始到最后一个句号之前的部分
    if last_period_index != -1:
        return s[:last_period_index]
    else:
        # 如果没有找到句号，返回整个字符串
        return s

def assign_difficulty(df):
    lengths = []
    for idx, row in df.iterrows():
        lengths.append(row["gt_dx_src_unique"])

    # Compute quantiles
    quantiles = np.percentile(lengths, [30, 70])


    # Second pass: write updated content to temp file
    difficulties = []
    for idx, row in df.iterrows():
       
        length = row["gt_dx_src_unique"]
        if length <= quantiles[0]:
            difficulty = 3
        elif length <= quantiles[1]:
            difficulty = 2
        else:
            difficulty = 1
        difficulties.append(difficulty)

    df["difficulty"] = difficulties
    return df

def evaluate_retrieval(predictions, ground_truths, k_list=[10, 50, 100, None]):
    """
    对多标签 node retrieval 任务进行评估。
    
    Args:
        predictions: list of lists. 每个元素是模型返回的排序节点列表（按相关性降序）
                     e.g., [ ['n1','n2',...], ... ]
        ground_truths: list of sets/lists. 每个元素是该 src 的真实 dst 集合
                     e.g., [ {'a','b'}, {'c','d','e'} ]
        k_list: list of int. 要评估的 top-K 值，如 [10, 50, 100]

    Returns:
        results: dict. 格式如下：
                {
                  'Recall@10': 0.65,
                  'Precision@10': 0.43,
                  'F1@10': 0.51,
                  'HitRate@10': 0.82,
                  'Recall@50': ...,
                  ...
                }
    """
    assert len(predictions) == len(ground_truths)
    results = {}
    sum_correct = 0
    for K in k_list:
        recalls = []
        precisions = []
        f1s = []
        hit_rates = []

        for pred_list, gt_set in zip(predictions, ground_truths):
            gt_set = set(gt_set)
             
            # 转换类型并取前 K
            if K is None:
                pred_set_at_k = set(pred_list[:len(gt_set)])
            else:
                if len(pred_list) > K:
                    pred_set_at_k = set(pred_list[:K])
                else:
                    pred_set_at_k = set(pred_list)
            

            if len(gt_set) == 0:
                continue  # 忽略无标签样本

            correct = pred_set_at_k & gt_set  # 交集
            n_correct = len(correct)
            n_pred = len(pred_set_at_k)
            n_gt = len(gt_set)
            sum_correct += n_correct
            # Recall@K
            recall = n_correct / n_gt if n_gt > 0 else 0
            recalls.append(recall)

            # Precision@K
            precision = n_correct / n_pred if n_pred > 0 else 0
            precisions.append(precision)

            # F1@K
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1s.append(f1)

            # Hit Rate@K (至少命中一个)
            hit_rate = 1 if n_correct > 0 else 0
            hit_rates.append(hit_rate)

        # 平均所有 query 的结果
        results[f"Ncorrect@{K}"] = sum_correct
        results[f'Recall@{K}'] = float(np.mean(recalls))
        results[f'Precision@{K}'] = float(np.mean(precisions))
        results[f'F1@{K}'] = float(np.mean(f1s))
        results[f'HitRate@{K}'] = float(np.mean(hit_rates))

    results = pd.DataFrame([results])
    return results

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


def assign_difficulty_result(df, data_ctdg):
    lengths = []
    gt_dst_node_idxs_unique_list = []
    for idx, row in df.iterrows():
        src_id = int(float(row["src_idx"]))
        output_edge_ids = np.array(data_ctdg.output_edges_dict[src_id])
        if not isinstance(data_ctdg.get_dst_ids(output_edge_ids), torch.Tensor):
            gt_dst_node_ids = torch.tensor(data_ctdg.get_dst_ids(output_edge_ids))
        else:
            gt_dst_node_ids = data_ctdg.get_dst_ids(output_edge_ids)
        gt_dst_node_idxs_unique = gt_dst_node_ids.unique().cpu().tolist()
        gt_dst_node_idxs_unique_list.append(gt_dst_node_idxs_unique)
        lengths.append(len(gt_dst_node_idxs_unique ))

    # Compute quantiles
    df["gt_dst_idxs_unique"] = gt_dst_node_idxs_unique_list
    quantiles = np.percentile(lengths, [30, 70])

    # Second pass: write updated content to temp file
    difficulties = []
    for idx, row in df.iterrows():
       
        length = len(row["gt_dst_idxs_unique"])
        if length <= quantiles[0]:
            difficulty = 3
        elif length <= quantiles[1]:
            difficulty = 2
        else:
            difficulty = 1
        difficulties.append(difficulty)

    df["difficulty"] = difficulties
    return df

import random

def construct_few_shot_prompt(text_array, dst_ids_pool, k=2):
    """
    构造 few-shot 示例 prompt 片段，用于 destination node selection.
    
    Args:
        data: 包含 node_text 等属性的数据对象
        dst_ids_pool: 可选的目标节点 ID 列表（用于采样）
        k: 采样 few-shot 数量
    
    Returns:
        few_shot_text: 格式化后的字符串，可拼接到主 prompt
    """
    if k <= 0 or len(dst_ids_pool) == 0:
        return ""

    # 随机采样 k 个 dst_id
    sampled_dst_ids = random.sample(dst_ids_pool, min(k, len(dst_ids_pool)))

    examples = []
    for i, dst_id in enumerate(sampled_dst_ids):
        try:
            # 获取目标节点文本
            examples.append(text_array[dst_id])
        except Exception as e:
            continue  # 跳过异常节点

    # 拼接成完整的 few-shot block
    few_shot_text = "\n\n".join(examples)
    return few_shot_text

def predict_dst_given_dx(src_id, 
                        dx_src,
                        input_edge_ids,
                        data:BWRCTDGDataset,
                        environment_data:dict,
                        cold_start = False,
                        gt_dst_text = None,
                        interaction_cache:dict = {},
                        few_shot: int = 0
                        ):
    dx_src = int(dx_src)
    # dx:[outdegree, k] [src_ids, dst_identifiers]
    # get: [N, dx * k] [src_ids, dst_ids] -> TemporalData(test_edges)
    src_node_text = data.get_src_node_texts(src_id, interaction_cache)
    
    memory_dst_texts = data.get_memory_dst(src_id,
                                            input_edge_ids,
                                           interaction_cache)
    
    neighbor_dst_texts = data.get_neighbor_dst(src_id, interaction_cache)
   
    ### market 
    if cold_start:
        agent_text = (
        f"Your task is to depict node text of dst nodes for the src node {src_id}",
        f"You're about to interact with {dx_src} dst nodes in the network."
        f"{environment_data['description']}",
        f"[For src-node ({src_id}):]",
        f"{src_node_text}",
        
        f"Here's your interaction history with other destination nodes in the network",
        f"{select_to_last_period(memory_dst_texts, upper_token=4e3)}",
        
        f"Here's your friends' interaction history with other destination nodes in the network",
        f"{select_to_last_period(neighbor_dst_texts, upper_token=2e3)}",
        # f"Do not assume that you already know the preset results. Instead, you should obtain the results of the dst node based on the analysis of the src node.",

        # f"Here's the ground truth dst nodes for the src node to interact with in future time steps:",
        # f"{gt_dst_text}"
        )
        
    else:
        agent_text = (
        f"Your task is to depict node text of dst nodes for the src node {src_id}",
        f"You're about to interact with {dx_src} dst nodes in the network."

        f"{environment_data['description']}",
        f"[For src-node ({src_id}):]",
        f"{src_node_text}",
        
        f"Here's your interaction history with other destination nodes in the network:",
        f"{select_to_last_period(memory_dst_texts, upper_token=4e3)}",
        
        f"Here's your friends' interaction history with other destination nodes in the network:",
        f"{select_to_last_period(neighbor_dst_texts, upper_token=2e3)}",

    )
       
    if few_shot!=0:
        few_shot_texts = construct_few_shot_prompt(data.edge_text, np.arange(data.node_feature.shape[0]),
        few_shot)
        agent_text +=(
            f"Here's a few response examples:",
            f"{select_to_last_period(few_shot_texts, upper_token=2e3)}",
        )
   
    content_hint = f"""
    {chr(10).join([
        *[f"<{k}>{v}</{k}>" for k,v in Dataset_Template[environment_data['data_name']]['node_text_hint'].items()],
        "<filter_rule>Optionally, you can give python array code, default is None, Similar to {'SF': '>1', 'AFN': '<1', 'HI': '==0', 'CN': '>=0'}</filter_rule>"
    ])}
"""
    format_instruction = (
        "You should respond a xml object in a xml fenced code block as "
        f"follows:\n```xml\n{content_hint}\n```"
    )

    agent_parser = RegexTaggedContentParser(
        format_instruction=format_instruction,
        required_keys=[*Dataset_Template[environment_data['data_name']]['node_text_cols'], "filter_rule"],
    )
    
    return "\n".join(agent_text), agent_parser

def predict_edge(src_id, 
                dst_id,
                input_edge_ids,
                data:BWRCTDGDataset,
                environment_data:dict,
                interaction_cache:dict = {},
                few_shot: int = 0
                ):

    # dx:[outdegree, k] [src_ids, dst_identifiers]
    # get: [N, dx * k] [src_ids, dst_ids] -> TemporalData(test_edges)
    src_node_text = data.get_src_node_texts(src_id, interaction_cache)
    
    dst_node_text = data.get_src_node_texts(dst_id, interaction_cache)

    memory_edge_texts = data.get_history_dst_edge_texts(input_edge_ids,
                                                        dst_id=dst_id)
    
    ### market 
    agent_text = (
        
    f"[For src-node ({src_id}):]",
    f"{select_to_last_period(src_node_text,3e2)}",

    f"[For dst-node ({dst_id}):]",
    f"{select_to_last_period(dst_node_text,3e2)}",

    f"Src-node({src_id}) past edges:\n{select_to_last_period(memory_edge_texts, 4e3)}"
    )
       
    
    content_hint = f"""
    {chr(10).join([
        *[f"<{k}>{v}</{k}>" for k,v in Dataset_Template[environment_data['data_name']]['edge_text_hint'].items()]
    ])}
"""
    format_instruction = (
        "You should respond a xml object in a xml fenced code block as "
        f"follows:\n```xml\n{content_hint}\n```"
    )

    if few_shot!=0:
        few_shot_texts = construct_few_shot_prompt(data.node_text, np.arange(data.node_feature.shape[0]),
        few_shot)
        agent_text +=(
            f"Here's a few response examples:",
            f"{select_to_last_period(few_shot_texts, upper_token=2e3)}",
        )

    agent_parser = RegexTaggedContentParser(
        format_instruction=format_instruction,
        required_keys=[*Dataset_Template[environment_data['data_name']]['edge_text_cols']],
    )
    
    return "\n".join(agent_text), agent_parser




import random

from typing import Iterable
def execute_search_dst_toolkit(
                           query_text: str,
                           dx_src: int,
                           dst_node_ids: np.ndarray,
                           src_id: int,
                           bert_embedder: BertEmbedder, 
                           environment_info: dict,
                           environment_data:BWRCTDGDataset,
                           interaction_cache:dict = {},
                           filter_rule = None,
                           recall_common_neighbor:bool = False,
                           recall_inductive: bool = False,
                           recall_alpha:int = 3,
                           recall_topk:int = None, # activate when set
                           use_src_node_text:bool = False,
                           ):
    if recall_topk is None:
        recall_number = recall_alpha*dx_src
    else:
        recall_number = int(recall_topk) if recall_topk > dx_src else int(dx_src)

    try:
        query_embedding = bert_embedder.get_embedding(query_text)
        if query_text == "" and not use_src_node_text:
            return {
            "dst_ids": random.choices(dst_node_ids, k=recall_number),
            "dst_metrics": []
        }
        
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
        print(e)
        return {
            "dst_ids": random.choices(dst_node_ids, k=recall_number),
            "dst_metrics": []
        }
        
def aggregate_rewards(query_reward_pairs, dx_src):
    """聚合query_reward_pairs中的rewards并采样dst nodes
    
    Args:
        query_reward_pairs: 包含query和reward信息的列表
        dx_src: 需要采样的dst nodes数量
        
    Returns:
        dict: 包含selected_dst_ids和对应ts的字典
    """
    # 初始化dst_id到reward pairs的映射
    dst_reward_pairs = defaultdict(list)
    selected_dst_reward_pairs = defaultdict(list)
    # 对每个query-reward pair进行聚合
    for pair in query_reward_pairs:
        dst_ids = pair["candidate_dst_ids"]
        selected_dst_ids = pair["selected_dst_ids"]
        reward = pair["reward"]
        for selected_dst_id in selected_dst_ids:
            selected_dst_reward_pairs[selected_dst_id].append({"reward": reward})
        # 为每个dst_id累积reward
        for dst_id in dst_ids:
            dst_reward_pairs[dst_id].append({"reward": reward})
    
    # 对每个dst_id计算聚合reward
    dst_agg_rewards = defaultdict(int)
    for dst_id, reward_pairs in dst_reward_pairs.items():
        # 计算聚合reward
        agg_reward = sum(float(pair["reward"]) for pair in reward_pairs)
        dst_agg_rewards[dst_id] += agg_reward
    
    # 对selected dst id再添加一次
    for dst_id, reward_pairs in selected_dst_reward_pairs.items():
        # 计算聚合reward
        agg_reward = sum(float(pair["reward"]) for pair in selected_dst_reward_pairs[dst_id])
        dst_agg_rewards[dst_id] += agg_reward
    
    # 按reward加权采样
    dst_ids = list(dst_agg_rewards.keys())
    rewards = list(dst_agg_rewards.values())
    probs = torch.softmax(torch.tensor(rewards,dtype=torch.float32), dim=0)
    
    # 采样dx_src个dst nodes
    prob_indices = torch.multinomial(probs, num_samples=dx_src)
    topk_selected_dst_ids = [dst_ids[idx] for idx in prob_indices]
    
    return {
        "topk_selected_dst_ids": topk_selected_dst_ids,
        "dst_ids": dst_ids,
        "dst_rewards": rewards,
        "dx_src": dx_src
    }

    
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

def process_single_prediction(
    dx_src_all: int,
    src_id: int,
    environment_data: Dict,
    data: BWRCTDGDataset,
    args: Any,
    interaction_cache: Dict,
    input_edge_ids: List,
    few_shot =0
) -> Dict:
    query_examples = []
    
    query_agent_text, agent_parser = predict_dst_given_dx(
        src_id, dx_src_all, input_edge_ids, data, 
        environment_data, False, None, interaction_cache, few_shot
    )
    output_edge_ids = np.array(data.output_edges_dict[src_id])
    if not isinstance(data.get_dst_ids(output_edge_ids), torch.Tensor):
        gt_dst_node_ids = torch.tensor(data.get_dst_ids(output_edge_ids))
    else:
        gt_dst_node_ids = data.get_dst_ids(output_edge_ids)
        
    query_examples.append({
                "prompt": QUERY_SYS_PROMPT + "\n" + agent_parser.format_instruction + "\n" + query_agent_text,
                # "identifier": f"{src_id}_{pred_idx}",
                "src_idx": src_id,  
                # "pred_idx": pred_idx,
                "gt_dx_src_all": dx_src_all, # 可能存在duplicates
                "gt_dx_src_unique": len(gt_dst_node_ids.unique().cpu().tolist()), # 不存在duplicates
                "gt_dst_idxs_unique": gt_dst_node_ids.unique().cpu().tolist(),
                "gt_dst_idxs_all":gt_dst_node_ids.cpu().tolist()
                # "instruction": QUERY_SYS_PROMPT + "\n" + agent_parser.format_instruction,
                # "input": query_agent_text,
    })
    
 
    return query_examples

def process_single_edge_attr_prediction(
    src_id: int,
    dst_ids: List[int], # dst 
    edge_ids: List[int], # edge id
    environment_data: Dict,
    data: BWRCTDGDataset,
    args: Any,
    interaction_cache: Dict,
    input_edge_ids: List,
    type: str = "pred", # pred or gt
    few_shot = 0
) -> Dict:
    edge_examples = []
    
    for dst_id, edge_id in zip(dst_ids, edge_ids):
        try:
            dst_id = dst_id.item()
            edge_id = edge_id.item()
        except:
            pass
        agent_text, agent_parser = predict_edge(
            src_id,
            dst_id,
            input_edge_ids,
            data,
            environment_data,
            interaction_cache,
            few_shot
        )
        if type == "pred":
            edge_examples.append({
                        "prompt": EDGE_ATTR_PROMPT + "\n" + agent_parser.format_instruction + "\n" + agent_text,
                        # "pred_edge_id": 
                        "src_idx": src_id,
                        "dst_idx": dst_id,
                        "edge_id": edge_id
                        
            })
        elif type == "gt":
            gt_edge_text = np.array(data.edge_text[edge_id])
            gt_label = int(data.ctdg.label[edge_id].item())
            gt_t = data.ctdg.t[edge_id].item()
            edge_examples.append({
                        "prompt": EDGE_ATTR_PROMPT+ "\n" + agent_parser.format_instruction + "\n" + agent_text,
                        "instruction": EDGE_ATTR_PROMPT+ "\n" + agent_parser.format_instruction,
                        "input": agent_text,
                        # "pred_edge_id": 
                        "src_idx": src_id,  
                        "dst_idx": dst_id,
                        "output": gt_edge_text,
                        "gt_label": gt_label,
                        "edge_id": edge_id,
                        "t": gt_t
            })
    
 
    return edge_examples


    

def collect_src_prediction_cold_start(
    pred_idx: int,
    dx_src: int,
    src_id: int,
    batch_data: Dict,
    environment_data: Dict,
    data: Any,
    args: Any,
    bert_embedder: Any,
    interaction_cache: Dict,
    input_edge_ids: List,
    output_edge_ids: List,
    dst_node_ids: List[int],
    sim_mean:float,
    sim_std:float
) -> Tuple[List, List]:
    # return instruction, input, output
    # return len: 2d_u query_examples
    # return len: 2d_u reward_examples
    
    
    query_positive_examples = []
    
    
    assert isinstance(data, BWRCTDGDataset)
    gt_dst_node_ids = data.get_dst_ids(output_edge_ids).unique().cpu().tolist()
    reward_gt_metrics = data.get_dst_nodes_texts(src_id,
                                                gt_dst_node_ids,
                                                interaction_cache=interaction_cache)
    dst_node_texts = [data.node_text[dst_id] for dst_id in gt_dst_node_ids]

    dst_text_to_ids = {}
    for dst_id, dst_text in zip(gt_dst_node_ids, dst_node_texts):
        if dst_text not in dst_text_to_ids:
            dst_text_to_ids[dst_text] = []
        dst_text_to_ids[dst_text].append(dst_id)
    
    dst_unique_node_texts = list(dst_text_to_ids.keys())
    gt_dst_node_ids_list = list(dst_text_to_ids.values())
    query_num = len(dst_unique_node_texts)
    
    # positive_samples
    for i in range(query_num):
        gt_dst_text = data.get_dst_nodes_texts(src_id, 
                                               gt_dst_node_ids_list[i], 
                                               interaction_cache=interaction_cache)
        
        query_agent_text, agent_parser = predict_dst_given_dx(
            src_id, dx_src, input_edge_ids,  data, 
            environment_data, True, gt_dst_text, interaction_cache
        )
        
        output = data.node_text[gt_dst_node_ids_list[i]][0]

        query_positive_examples.append({
                    "prompt": QUERY_SYS_PROMPT + "\n" + agent_parser.format_instruction + "\n" + query_agent_text,
                    "instruction": QUERY_SYS_PROMPT + "\n" + agent_parser.format_instruction,
                    "input": query_agent_text,
                    "output":output,
                    "type": "positive"
        })
        
    return query_positive_examples




from torch_geometric.data import TemporalData

def get_gen_data(src_dsts) -> TemporalData:
    # 初始化边列表
    edges = []
    
    # 遍历每个源节点及其预测边
    for src, pred_edges in src_dsts.items():
        # 遍历每个预测时间步
        for pred_idx, query_reward_pair in pred_edges.items():
            # 遍历每个查询-奖励对
            # 获取候选目标节点ID
            dst_ids = query_reward_pair["topk_selected_dst_ids"]
            ts = query_reward_pair["ts"]
            msg = query_reward_pair["msg"]
            # 为每个目标节点创建边
            for dst_id, t, msg in zip(dst_ids, ts, msg):
                edge = {
                    "src_idx": src,
                    "dst_idx": dst_id,
                    "t": t,
                    "msg": msg
                }
                edges.append(edge)
    
    # 将边数据转换为张量
    src = torch.tensor([edge["src_idx"] for edge in edges], dtype=torch.int64)
    dst = torch.tensor([edge["dst_idx"] for edge in edges], dtype=torch.int64)
    t = torch.tensor([edge["t"] for edge in edges], dtype=torch.int64)
    msg = torch.stack([edge["msg"] for edge in edges],dim=0).to(torch.float32)
    # 创建并返回时序数据对象
    return TemporalData(src=src, dst=dst, t=t, msg=msg)


   
    


def main_infer_edge(query_result_path, dx_src_path: str = None):
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    print("Loading degree predictor results from:", dx_src_path)
    data_ctdg.load_degree_predictor_results(dx_src_path)

    query_examples_all_result = pd.read_csv(query_result_path)
    prompt_dir = os.path.join(args.save_root,f'prompts/{args.model_config_name}/{args.data_name}/{args.split}/inference')

    if "seq_deg" in dx_src_path:
        prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/seq_inference')
    os.makedirs(prompt_dir, exist_ok=True)

    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )

    edge_text_examples_all = pd.DataFrame()
    pred_src_ids = []
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader), "predicting edges"):
        non_zero_indices = np.where(batch_data['src_model_pred_degree']>0)
        for batch_inter_idx, bwr_idx, pred_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1], 
                                                    non_zero_indices[2]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            if src_id in pred_src_ids:
                continue

            pred_src_ids.append(src_id)
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dst_ids = query_examples_all_result[query_examples_all_result["src_idx"] == src_id]['dst_idx'].tolist()
            output_edge_ids = query_examples_all_result[query_examples_all_result["src_idx"] == src_id]['edge_id'].tolist()
            pred_degree = np.sum(data_ctdg.pred_src_dx[src_id])
            # assert np.sum(data_ctdg.pred_src_dx[src_id]) == len(dst_ids), f"{pred_degree}, {len(dst_ids)}: {src_id}"
            edge_text_examples = process_single_edge_attr_prediction(
                src_id,
                dst_ids,
                output_edge_ids,
                environment_data,
                data_ctdg,
                args,
                data_ctdg.interaction_cache,
                input_edge_ids,
                type = "pred"
            )
            edge_text_examples_all = pd.concat([edge_text_examples_all, 
                                                pd.DataFrame(edge_text_examples)], 
                                                ignore_index=True)

    edge_text_examples_all.drop_duplicates(subset=["edge_id"], inplace=True)
    columns_to_drop = ["instruction", "input"]
   
    edge_text_examples_all.drop(columns=[col for col in columns_to_drop if col in edge_text_examples_all.columns], 
            inplace=True)
    
    edge_text_examples_all['tag'] = args.split
    edge_text_examples_all.to_csv(os.path.join(prompt_dir, 'edge_text_examples.csv'), index=False)
    
    print(f"Edge examples prompt mean length: {edge_text_examples_all['prompt'].str.len().mean():.2f}")
    print(f"Edge examples prompt max length: {edge_text_examples_all['prompt'].str.len().max()}")

def main_infer_dst(dx_src_path: str = None):
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")

    data_ctdg.load_degree_predictor_results(dx_src_path)
    
    sim = (data_ctdg.node_feature @ data_ctdg.node_feature.T)
    sim_mean = sim.mean()
    sim_std = sim.std()
    
    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/inference')
    if "seq_deg" in dx_src_path:
        prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/seq_inference')


    os.makedirs(prompt_dir, exist_ok=True)
    
    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )
    
    query_all_examples = pd.DataFrame()
    
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader),
    "predicting edges"):
        dx_src_all_batch = np.sum(batch_data['src_model_pred_degree'], axis = -1)
        non_zero_indices = np.where(dx_src_all_batch>0)
        dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
    
        for batch_inter_idx, bwr_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dx_src_all = dx_src_all_batch[batch_inter_idx][bwr_idx] # 某些dst可能重复

            query_examples = process_single_prediction(
                dx_src_all, src_id,  environment_data,
                data_ctdg, args,  data_ctdg.interaction_cache, 
                input_edge_ids, args.few_shot
            )
            
            query_all_examples = pd.concat([query_all_examples, 
                                            pd.DataFrame(query_examples)], 
                                            ignore_index=True)
            
    query_all_examples.drop_duplicates(subset=['src_idx'], inplace=True)
    
    columns_to_drop = ["instruction", "input"]
    query_all_examples.drop(columns=[col for col in columns_to_drop if col in query_all_examples.columns], 
            inplace=True)
   
    
    query_all_examples['tag'] = args.split
    query_all_examples['domain'] = 'dst_rule'

    filtered_examples = query_all_examples[query_all_examples['gt_dx_src_unique'] != 0]
    query_all_examples = assign_difficulty(filtered_examples)

    query_all_examples.to_csv(os.path.join(prompt_dir, 'query_examples.csv'), index=False)

    print(f"Query examples prompt mean length: {query_all_examples['prompt'].str.len().mean():.2f}")
    print(f"Query examples prompt max length: {query_all_examples['prompt'].str.len().max()}")
    

def main():
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")

    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/teacher_forcing')
    result_dir = os.path.join(args.save_root,f'results/{args.model_config_name}/{args.data_name}/{args.split}/teacher_forcing')
   
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )
    
    query_file_name = f"query_examples.csv"
    query_all_examples = pd.DataFrame()
    if os.path.exists(os.path.join(result_dir, query_file_name)):
        query_result_path = os.path.join(result_dir, query_file_name)
        process_query_result(environment_data, 
                             data_ctdg,
                             True,
                             result_path=query_result_path,
                             interaction_cache=data_ctdg.interaction_cache,
                             recall_common_neighbor=True,
                             recall_inductive=False
                             )
        
        
    
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader),
    "predicting edges"):

        batch_data["src_model_pred_degree"] = batch_data["src_node_degree"]    
        dx_src_all_batch = np.sum(batch_data['src_model_pred_degree'], axis = -1)
        non_zero_indices = np.where(dx_src_all_batch>0)
        dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
    
        for batch_inter_idx, bwr_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dx_src_all = dx_src_all_batch[batch_inter_idx][bwr_idx] # 某些dst可能重复

            query_examples = process_single_prediction(
                dx_src_all, src_id,  environment_data,
                data_ctdg, args,  data_ctdg.interaction_cache, 
                input_edge_ids, args.few_shot
            )
            
            query_all_examples = pd.concat([query_all_examples, 
                                            pd.DataFrame(query_examples)], 
                                            ignore_index=True)
            

    edge_text_examples_all = pd.DataFrame()
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader), "predicting edges"):
        ## 训练过程 teacher forcing 
        batch_data["src_model_pred_degree"] = batch_data["src_node_degree"]        
        non_zero_indices = np.where(batch_data['src_model_pred_degree']>0)
        for batch_inter_idx, bwr_idx, pred_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1], 
                                                    non_zero_indices[2]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dx_src = batch_data['src_model_pred_degree'][batch_inter_idx][bwr_idx][pred_idx]
            pred_dx_src_sum = batch_data['src_model_pred_degree'][batch_inter_idx][bwr_idx][:pred_idx].sum()
            output_edge_ids = batch_data['output_edge_ids'][batch_inter_idx][bwr_idx][pred_dx_src_sum:pred_dx_src_sum+dx_src]
            dst_ids = data_ctdg.get_dst_ids(output_edge_ids)
            edge_text_examples = process_single_edge_attr_prediction(
                src_id,
                dst_ids,
                output_edge_ids,
                environment_data,
                data_ctdg,
                args,
                data_ctdg.interaction_cache,
                input_edge_ids,
                type = "gt"
            )
            
            edge_text_examples_all = pd.concat([edge_text_examples_all, 
                                                pd.DataFrame(edge_text_examples)], 
                                                ignore_index=True)

    edge_text_examples_all.drop_duplicates(subset=["edge_id"], inplace=True)
    query_all_examples.drop_duplicates(subset=['src_idx'], inplace=True)
    
    columns_to_drop = ["instruction", "input"]
    query_all_examples.drop(columns=[col for col in columns_to_drop if col in query_all_examples.columns], 
            inplace=True)
    edge_text_examples_all.drop(columns=[col for col in columns_to_drop if col in edge_text_examples_all.columns], 
            inplace=True)

    assert np.sum(np.sum(data_ctdg.ctdg_src_node_degree[:,data_ctdg.input_len:],axis=-1)>0) == \
            query_all_examples.shape[0], "query_all_examples should be equal to the number of src nodes with degree > 0"
    assert np.where(data_ctdg.ctdg_src_node_degree > 0, data_ctdg.ctdg_src_node_degree, 0)[:,data_ctdg.input_len:].sum() == \
        edge_text_examples_all.shape[0], "edge_text_examples_all should be equal to the number of degree sum"
    


    query_all_examples['tag'] = args.split
    query_all_examples['domain'] = 'dst_rule'
    query_all_examples = assign_difficulty(query_all_examples)
    
    edge_text_examples_all['tag'] = args.split
    edge_text_examples_all['domain'] = 'edge_rule'
    indices = edge_text_examples_all.sample(frac=0.5, random_state=42).index
    edge_text_examples_all.loc[indices, 'domain'] = 'edge_text_rule'
    edge_text_examples_all["difficulty"] = 1


    combined_df = pd.concat([query_all_examples, edge_text_examples_all], ignore_index=True)
    query_sub = query_all_examples[query_all_examples["difficulty"] < 3]
    edge_sub_index = edge_text_examples_all.sample(frac=query_sub.shape[0]/edge_text_examples_all.shape[0], random_state=0).index
    edge_sub = edge_text_examples_all.loc[edge_sub_index]
    assert edge_sub[edge_sub["domain"]== 'edge_rule'].shape[0] != 0

    sub_df = pd.concat([query_sub, edge_sub], ignore_index=True)

    # for multi domain rl
    combined_df.to_csv(os.path.join(prompt_dir, 'combined_examples.csv'), index=False)
    if args.split == 'val' and args.data_name != "8days_dytag_small_text_en" and sub_df.shape[0] >300:
    # 随机采样 300 条，不重复，样本数不超过总数
        sub_df = sub_df.sample(n=300, replace=False, random_state=42)  # random_state 可选，保证可复现
        
    sub_df.to_csv(os.path.join(prompt_dir, 'combined_easy_examples.csv'), index=False)
    
    # for single-domain rl
    query_all_examples.to_csv(os.path.join(prompt_dir, 'query_examples.csv'), index=False)
    edge_text_examples_all.to_csv(os.path.join(prompt_dir, 'edge_text_examples.csv'), index=False)
    
    print(f"Query examples prompt mean length: {query_all_examples['prompt'].str.len().mean():.2f}")
    print(f"Query examples prompt max length: {query_all_examples['prompt'].str.len().max()}")
    
    print(f"Edge examples prompt mean length: {edge_text_examples_all['prompt'].str.len().mean():.2f}")
    print(f"Edge examples prompt max length: {edge_text_examples_all['prompt'].str.len().max()}")
    
    
def main_idgg(dx_src_path: str = None):
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")

    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/idgg')
   
    os.makedirs(prompt_dir, exist_ok=True)
    
    data_ctdg.load_degree_predictor_results(dx_src_path)

    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )
    
    
        
        
    
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader),
    "predicting edges"):

        # batch_data["src_model_pred_degree"] = batch_data["src_node_degree"]    
        dx_src_all_batch_gt = np.sum(batch_data['src_node_degree'], axis = -1)
        dx_src_all_batch = np.sum(batch_data['src_model_pred_degree'], axis = -1)
        non_zero_indices = np.where(dx_src_all_batch_gt>0)
    
        for batch_inter_idx, bwr_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dx_src_all = dx_src_all_batch[batch_inter_idx][bwr_idx] # 某些dst可能重复
            dx_src_all_gt = dx_src_all_batch_gt[batch_inter_idx][bwr_idx]
            
            if dx_src_all_gt > 0 and dx_src_all > 0:
                query_examples = process_single_prediction(
                    dx_src_all, src_id,  environment_data,
                    data_ctdg, args,  data_ctdg.interaction_cache, 
                    input_edge_ids, args.few_shot
                )
                query_all_examples = pd.concat([query_all_examples, 
                                                pd.DataFrame(query_examples)], 
                                                ignore_index=True)
            

    edge_text_examples_all = pd.DataFrame()
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader), "predicting edges"):
        ## 训练过程 teacher forcing 
        batch_data["src_model_pred_degree"] = batch_data["src_node_degree"]        
        non_zero_indices = np.where(batch_data['src_model_pred_degree']>0)
        for batch_inter_idx, bwr_idx, pred_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1], 
                                                    non_zero_indices[2]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dx_src = batch_data['src_model_pred_degree'][batch_inter_idx][bwr_idx][pred_idx]
            pred_dx_src_sum = batch_data['src_model_pred_degree'][batch_inter_idx][bwr_idx][:pred_idx].sum()
            output_edge_ids = batch_data['output_edge_ids'][batch_inter_idx][bwr_idx][pred_dx_src_sum:pred_dx_src_sum+dx_src]
            dst_ids = data_ctdg.get_dst_ids(output_edge_ids)
            edge_text_examples = process_single_edge_attr_prediction(
                src_id,
                dst_ids,
                output_edge_ids,
                environment_data,
                data_ctdg,
                args,
                data_ctdg.interaction_cache,
                input_edge_ids,
                type = "gt",
                few_shot = args.few_shot
            )
            
            edge_text_examples_all = pd.concat([edge_text_examples_all, 
                                                pd.DataFrame(edge_text_examples)], 
                                                ignore_index=True)

    edge_text_examples_all.drop_duplicates(subset=["edge_id"], inplace=True)
    query_all_examples.drop_duplicates(subset=['src_idx'], inplace=True)
    
    columns_to_drop = ["instruction", "input"]
    query_all_examples.drop(columns=[col for col in columns_to_drop if col in query_all_examples.columns], 
            inplace=True)
    edge_text_examples_all.drop(columns=[col for col in columns_to_drop if col in edge_text_examples_all.columns], 
            inplace=True)


    query_all_examples['tag'] = args.split
    query_all_examples['domain'] = 'dst_rule'
    # Step 1: 过滤掉 "gt_dx_src_unique" == 0 的行
    filtered_examples = query_all_examples[query_all_examples['gt_dx_src_unique'] != 0]

    # Step 2: 在过滤后的数据上应用 assign_difficulty 函数
    query_all_examples = assign_difficulty(filtered_examples)
    
    edge_text_examples_all['tag'] = args.split
    edge_text_examples_all['domain'] = 'edge_rule'
    indices = edge_text_examples_all.sample(frac=0.5, random_state=42).index
    edge_text_examples_all.loc[indices, 'domain'] = 'edge_text_rule'
    edge_text_examples_all["difficulty"] = 1


    combined_df = pd.concat([query_all_examples, edge_text_examples_all], ignore_index=True)
    query_sub = query_all_examples[query_all_examples["difficulty"] < 3]
    edge_sub_index = edge_text_examples_all.sample(frac=query_sub.shape[0]/edge_text_examples_all.shape[0], random_state=0).index
    edge_sub = edge_text_examples_all.loc[edge_sub_index]
    assert edge_sub[edge_sub["domain"]== 'edge_rule'].shape[0] != 0

    sub_df = pd.concat([query_sub, edge_sub], ignore_index=True)

    # for multi domain rl
    combined_df.to_csv(os.path.join(prompt_dir, 'combined_examples.csv'), index=False)
    
    if args.split == 'val' and args.data_name != "8days_dytag_small_text_en" and sub_df.shape[0] >300:
    # 随机采样 300 条，不重复，样本数不超过总数
        sub_df = sub_df.sample(n=300, replace=False, random_state=42)  # random_state 可选，保证可复现
    sub_df.to_csv(os.path.join(prompt_dir, 'combined_easy_examples.csv'), index=False)
    
    # for single-domain rl
    query_all_examples.to_csv(os.path.join(prompt_dir, 'query_examples.csv'), index=False)
    edge_text_examples_all.to_csv(os.path.join(prompt_dir, 'edge_text_examples.csv'), index=False)
    
    print(f"Query examples prompt mean length: {query_all_examples['prompt'].str.len().mean():.2f}")
    print(f"Query examples prompt max length: {query_all_examples['prompt'].str.len().max()}")
    
    print(f"Edge examples prompt mean length: {edge_text_examples_all['prompt'].str.len().mean():.2f}")
    print(f"Edge examples prompt max length: {edge_text_examples_all['prompt'].str.len().max()}")
    

    

def main_inference_offline_cold_start():
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    else:
        raise ValueError(f"Invalid split for cold start: {args.split}")
    
    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/cold_start')
    
    os.makedirs(prompt_dir, exist_ok=True)

    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    sim = (data_ctdg.node_feature @ data_ctdg.node_feature.T)
    sim_mean = sim.mean()
    sim_std = sim.std()
    
    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )
        
    query_positive_examples_all = pd.DataFrame()   
    edge_text_examples_all = pd.DataFrame()

    bert_embedder = BertEmbedder()
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader),
    "predicting edges"):
        ## 训练过程 teacher forcing 
        
        non_zero_mask = batch_data['src_node_degree'] > 0
        non_zero_indices = np.where(non_zero_mask)
        for batch_inter_idx, bwr_idx, pred_idx in zip(non_zero_indices[0], 
                                                  non_zero_indices[1], 
                                                  non_zero_indices[2]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dx_src = batch_data['src_node_degree'][batch_inter_idx][bwr_idx][pred_idx]
            pred_dx_src_sum = batch_data['src_node_degree'][batch_inter_idx][bwr_idx][:pred_idx].sum()
            output_edge_ids = batch_data['output_edge_ids'][batch_inter_idx][bwr_idx][pred_dx_src_sum:pred_dx_src_sum+dx_src]
            
            
            """处理单个预测的异步函数"""
            dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
            query_positive_examples = collect_src_prediction_cold_start(
                    pred_idx=pred_idx,
                    dx_src=dx_src,
                    src_id=src_id,
                    batch_data=batch_data,
                    environment_data=environment_data,
                    data=data_ctdg,
                    args=args,
                    bert_embedder=bert_embedder,
                    interaction_cache=data_ctdg.interaction_cache,
                    input_edge_ids=input_edge_ids,
                    output_edge_ids=output_edge_ids,
                    dst_node_ids=dst_node_ids,
                    sim_mean=sim_mean,
                    sim_std=sim_std
                )
            query_positive_examples_all = pd.concat([query_positive_examples_all, 
                                                     pd.DataFrame(query_positive_examples)], 
                                                     ignore_index=True)
            
            
            dst_ids = data_ctdg.get_dst_ids(output_edge_ids)
            edge_text_examples = process_single_edge_attr_prediction(
                src_id,
                dst_ids,
                output_edge_ids,
                environment_data,
                data_ctdg,
                args,
                data_ctdg.interaction_cache,
                input_edge_ids,
                type = "gt"
            )
            edge_text_examples_all = pd.concat([edge_text_examples_all, pd.DataFrame(edge_text_examples)], ignore_index=True)


    print(f"Query positive examples prompt mean length: {query_positive_examples_all['prompt'].str.len().mean():.2f}")
    print(f"Query positive examples prompt max length: {query_positive_examples_all['prompt'].str.len().max()}")
    
    print(f"Edge examples prompt mean length: {edge_text_examples_all['prompt'].str.len().mean():.2f}")
    print(f"Edge examples prompt max length: {edge_text_examples_all['prompt'].str.len().max()}")            
    saved_cols = ["input", "instruction","output"]
    query_positive_examples_all = query_positive_examples_all[saved_cols]
    edge_text_examples_all =edge_text_examples_all[saved_cols]    
    
    # query_positive_examples_all.to_csv(os.path.join(prompt_dir, 'query_positive_examples.csv'), index=False)
    # edge_text_examples_all.to_csv(os.path.join(prompt_dir, 'edge_text_examples.csv'), index=False)
    combined_df = pd.concat([query_positive_examples_all, edge_text_examples_all], ignore_index=True)
    combined_df.to_csv(os.path.join(prompt_dir, 'combined_examples.csv'), index=False)





    
 
def strip_prefix(prefix:str,
                 strings:List[str]):
    # 定义要去掉的前缀
    
    # 处理每个字符串，去掉以该前缀开头的部分
    processed_input = []
    for s in strings:
        if s.startswith(prefix):
            # 去掉前缀部分，保留其余内容
            new_s = s[len(prefix):].strip()
            processed_input.append(new_s)
        else:
            processed_input.append(s)

    # 输出处理后的结果
    return strings

   

class DstReward:
    def __init__(self,
                 args):
        
        self.reward_sel = args.reward_sel
        if args.reward_sel is None: return
        
        if args.reward_sel == "gnn":
            node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num = get_link_prediction_data(args)
                
            full_neighbor_sampler = get_neighbor_sampler(
                    data=full_data, 
                    sample_neighbor_strategy=args.sample_neighbor_strategy,
                    time_scaling_factor=args.time_scaling_factor, 
                    seed=args.seed)
            
            self.model_name = args.gnn_model_name
            self.model_type = "lp"
            self.gnn_judger = create_link_prediction_model(
                model_name=args.gnn_model_name,
                save_model_path=args.gnn_save_model_path,
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                data=full_data,
                neighbor_sampler=full_neighbor_sampler
            )
        
    def reward(self,
                src_ids:np.ndarray,
                dst_ids:np.ndarray,
                interact_times:np.ndarray):
        if self.reward_sel is None: return np.zeros(len(src_ids))
        
        if self.reward_sel == "gnn":
            return self.gnn_reward_src_query_data(src_ids, dst_ids, interact_times)
        
    
    def gnn_reward_src_query_data(self,
                                  src_ids:np.ndarray,
                                  dst_ids:np.ndarray,
                                  interact_times:np.ndarray):
        
        return compute_src_dsts_score(
            src_ids,
            dst_ids,
            interact_times,
            self.model_name,
            self.gnn_judger,
            model_type=self.model_type)    
        
class EdgeReward:
    def __init__(self,
                 args):
        
        self.reward_sel = args.reward_sel
        if args.reward_sel is None: return
        
        if args.reward_sel == "gnn":
            node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num = get_link_prediction_data(args)
                
            full_neighbor_sampler = get_neighbor_sampler(
                    data=full_data, 
                    sample_neighbor_strategy=args.sample_neighbor_strategy,
                    time_scaling_factor=args.time_scaling_factor, 
                    seed=args.seed)
            
            self.model_name = args.gnn_model_name
            self.model_type = "ec"
            self.gnn_judger = create_edge_classification_model(
                model_name=args.gnn_model_name,
                save_model_path=args.gnn_save_model_path,
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                data=full_data,
                neighbor_sampler=full_neighbor_sampler
            )
        
    def reward(self,
                src_ids:np.ndarray,
                dst_ids:np.ndarray,
                interact_times:np.ndarray,
                pred_edge_id: int):
        if self.reward_sel is None: return 0
        
        if self.reward_sel == "gnn":
            return self.gnn_reward_src_query_data(src_ids, dst_ids, interact_times)[0,pred_edge_id].item()
        
    
    def gnn_reward_src_query_data(self,
                                  src_ids:np.ndarray,
                                  dst_ids:np.ndarray,
                                  interact_times:np.ndarray):
        
        return compute_src_dsts_score(
            src_ids,
            dst_ids,
            interact_times,
            self.model_name,
            self.gnn_judger,
            model_type=self.model_type)    
    
    
def process_query_result_idgg(
                 teacher_forcing:bool,
                 args,
                 gen_col:str = "generate_results",
                 recall_common_neighbor: bool = True,
                 recall_inductive: bool = False,
                 ):
    

    query_examples_all_result = pd.read_csv(args.query_save_path)
    assert gen_col in query_examples_all_result.columns and "src_idx" in query_examples_all_result.columns, f"gen_col {gen_col} not in query_examples_all_result"
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    if not teacher_forcing:
        data_ctdg.load_degree_predictor_results(args.dx_src_path)

    
    bert_embedder = BertEmbedder()
    query_parser = RegexTaggedContentParser(
        required_keys=Dataset_Template[environment_data['data_name']]['node_text_cols'],
    )
    
    
    identifier_map = {}
    data_ctdg_loader = DataLoader(data_ctdg, 
                            batch_size=1, 
                            shuffle=False,
                            collate_fn=custom_collate
                            )
    dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
    for batch_idx, batch_data in enumerate(data_ctdg_loader):
        if teacher_forcing:
            batch_data["src_model_pred_degree"] = batch_data["src_node_degree"]    

        dx_src_all_batch = np.sum(batch_data['src_model_pred_degree'], axis = -1)
        non_zero_indices = np.where(dx_src_all_batch>0)
        
        for batch_inter_idx, bwr_idx in zip(non_zero_indices[0], 
                                            non_zero_indices[1]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            pred_ids = np.where(batch_data['src_model_pred_degree'][batch_inter_idx][bwr_idx]>0)[0]
            identifier_map[src_id] = {       
                "dx_src_list": batch_data['src_model_pred_degree'][batch_inter_idx][bwr_idx].tolist(),
                "dst_node_ids": dst_node_ids,
                "pred_ids": pred_ids.tolist()
            }
    
    
    parsed_results = []
    for idx, row in query_examples_all_result.iterrows():
        try:
            parsed_results.append({
                "parsed": query_parser.parse(ModelResponse(row[gen_col])).parsed,
                "success": True
            } )
        except Exception as e:
            parsed_results.append({
                "parsed": None,
                "success": False
            })
            
    fail_count = sum(1 for result in parsed_results if not result["success"])
    print(f"解析失败的数量: {fail_count}")

    query_examples_all_result["success"] = [result["success"] for result in parsed_results]
    for col in Dataset_Template[environment_data['data_name']]['node_text_cols']:
        query_examples_all_result[col] = [result["parsed"].get(col, None) if result["success"] else None for result in parsed_results]
    
    
    rewarder = DstReward(args)
    query_edges_src_all = []
    
    cols_text = Dataset_Template[environment_data['data_name']]['node_text_cols']
    grouped_df = query_examples_all_result.groupby('src_idx')

    assert len(list(filter(lambda x: x not in identifier_map.keys(),
                           query_examples_all_result["src_idx"])))==0
    for src_id, group in tqdm(grouped_df, "processing query examples"):
        src_id = int(float(src_id))
        candidate_dst_ids_all = []
        query_edges = pd.DataFrame(columns=["src_idx", "dst_idx", "t"])
        for _, row in group.iterrows():
            
            dx_src_list = identifier_map[src_id]["dx_src_list"]
            dst_node_ids = identifier_map[src_id]["dst_node_ids"]
            if row["success"]:
                query_text = Dataset_Template[environment_data['data_name']]['node_text_template'].format_map(row[cols_text].to_dict())
                filter_rule = row.get("filter_rule")
            else:
                # query_text = data_ctdg.node_text[src_id]
                query_text = ""
                filter_rule = None
            
            
            for pred_idx in identifier_map[src_id]["pred_ids"]:
                candidate_dst_ids = execute_search_dst_toolkit(query_text,
                                                                    np.sum(dx_src_list),
                                                                    dst_node_ids,
                                                                    src_id,
                                                                    bert_embedder,
                                                                    environment_data,
                                                                    data_ctdg,
                                                                    data_ctdg.interaction_cache,
                                                                    filter_rule,
                                                                    recall_common_neighbor=recall_common_neighbor,
                                                                    # recall_inductive=recall_inductive,
                                                                    # recall_topk=100
                                                                    recall_alpha=3,
                                                                    # use_src_node_text = True
                                                                )
                t = data_ctdg.unique_times[data_ctdg.input_len + int(pred_idx)]
                times = [t] * int(dx_src_list[pred_idx])

                ## get candidate dst ids (candidate len)
                gt_dst_idxs_unique = row["gt_dst_idxs_unique"]
                for dst_id, t in zip(candidate_dst_ids["dst_ids"][:int(dx_src_list[pred_idx])], times):
                    query_edges.loc[len(query_edges)] = {
                        "src_idx": src_id,
                        "dst_idx": dst_id,
                        "t": int(t)
                    }
        assert np.array(dx_src_list).sum() == query_edges.shape[0], f"error {src_id}"
        query_edges_src_all.append(query_edges)
            # preprocess
            # candidate_dst_ids["gt_dst_idxs_unique"] = preprocess_candidate_set(
            #                                                 candidate_dst_ids["dst_ids"],
            #                                                 gt_dst_idxs_unique)

            ## get choosen dst ids (degree len)
            # candidate_dst_ids["dst_ids"] = candidate_dst_ids["dst_ids"][:np.sum(dx_src_list)]
            # 为每个 pred_idx 分配对应的时间，并根据 dx_src_list[pred_idx] 的数量重复该时间
            # times = []
            # for pred_idx in identifier_map[row["src_idx"]]["pred_ids"]:
            #     t = data_ctdg.unique_times[data_ctdg.input_len + int(pred_idx)]
            #     times.extend([t] * int(dx_src_list[pred_idx]))
                
            
            # for dst_id, t in zip(candidate_dst_ids["dst_ids"][:np.sum(dx_src_list)], times):
            #     query_edges.loc[len(query_edges)] = {
            #         "src_idx": src_id,
            #         "dst_idx": dst_id,
            #         "t": int(t)
            #     }
            
            # score = rewarder.reward(query_edges["src_idx"].values,
            #                         query_edges["dst_idx"].values,
            #                         query_edges["t"].values)
            
            # candidate_dst_ids_all.append((query_edges,score,candidate_dst_ids["dst_ids"], gt_dst_idxs_unique))
        
        # score candidates
        # 按照score对candidate_dst_ids_all进行排序，由高到低
        
        
        ## eval candidate recall acccuracy
        # best_can_dst_ids = candidate_dst_ids_all[0][2]
        # best_gt_dst_idxs_unique = candidate_dst_ids_all[0][3]
        # retrival_can_dst_list.append(best_can_dst_ids)
        # retrival_gt_dst_list.append(best_gt_dst_idxs_unique)
        # hub_mask.append(1 if src_id in hub_src_ids else 0)
        
        ### test time scaling: prove to be not effective
        # # 将 candidate_dst_ids_all 的 query_edges concat 并且按照频率从高到低排序，选择 iloc[:dx_src] 的边
        # # 1. 合并所有 query_edges
        # all_edges = pd.concat([item[0] for item in candidate_dst_ids_all], ignore_index=True)
        # # 2. 统计每个 (src_idx, dst_idx, t) 的出现次数
        # edge_counts = all_edges.value_counts(subset=["src_idx", "dst_idx", "t"]).reset_index(name="count")
        # # 3. 按照频率从高到低排序
        # edge_counts = edge_counts.sort_values("count", ascending=False)
        # # 4. 选择 iloc[:dx_src] 的边
        # dx_src = int(np.sum(identifier_map[src_id]["dx_src_list"]))
        # selected_edges = edge_counts.iloc[:dx_src][["src_idx", "dst_idx", "t"]].copy()
        # # 5. 添加到 query_edges_src_all
        # query_edges_src_all.append(selected_edges)
        
    

    query_edges_src_all = pd.concat(query_edges_src_all, ignore_index=True) # 含有src,dst,t, edge_id 四列
    # 直接使用 np.arange 设置 edge_id 列
    query_edges_src_all["edge_id"] = np.arange(len(query_edges_src_all))
    query_edges_src_all.to_csv(args.query_result_path, index=False)

def process_query_result(
                 teacher_forcing:bool,
                 args,
                 gen_col:str = "generate_results",
                 recall_common_neighbor: bool = True,
                 recall_inductive: bool = False,
                 ):
    

    query_examples_all_result = pd.read_csv(args.query_save_path)
    assert gen_col in query_examples_all_result.columns and "src_idx" in query_examples_all_result.columns, f"gen_col {gen_col} not in query_examples_all_result"
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    if not teacher_forcing:
        data_ctdg.load_degree_predictor_results(args.dx_src_path)

    
    bert_embedder = BertEmbedder()
    query_parser = RegexTaggedContentParser(
        required_keys=Dataset_Template[environment_data['data_name']]['node_text_cols'],
    )
    rewarder = DstReward(args)
    
    dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
    
    
    parsed_results = []
    for idx, row in query_examples_all_result.iterrows():
        try:
            parsed_results.append({
                "parsed": query_parser.parse(ModelResponse(row[gen_col])).parsed,
                "success": True
            } )
        except Exception as e:
            parsed_results.append({
                "parsed": None,
                "success": False
            })
            
    fail_count = sum(1 for result in parsed_results if not result["success"])
    print(f"解析失败的数量: {fail_count}")

    query_examples_all_result["success"] = [result["success"] for result in parsed_results]
    for col in Dataset_Template[environment_data['data_name']]['node_text_cols']:
        query_examples_all_result[col] = [result["parsed"].get(col, None) if result["success"] else None for result in parsed_results]
    
    

    query_examples_all_result = assign_difficulty(query_examples_all_result)
    cols_text = Dataset_Template[environment_data['data_name']]['node_text_cols']

    dst_node_ids = np.arange(environment_data['dst_min'], 
                             environment_data['dst_max'] + 1)
    final_pred_time = data_ctdg.unique_times[-1]
    # 初始化存储结构
    results_list = []

    # 所有要统计的指标和 topk
    metrics = ['recall', 'hit']
    ks = [3, 10, 50, 100]
    if rewarder.reward_sel == "gnn":
        metrics.extend(['gnn_recall','gnn_hit'])

    for idx, row in tqdm(query_examples_all_result.iterrows(), 
                         "processing query"):
        src_idx = int(float(row["src_idx"]))
        dx_src, gt_dst_idxs = row["gt_dx_src_unique"], row["gt_dst_idxs_unique"]
        gt_dst_idxs = set(eval(gt_dst_idxs))
        is_hub = row["difficulty"] < 3
        result = {
            'src_idx': src_idx,
            'is_hub': is_hub
        }
        
        if row["success"]:
            query_text = Dataset_Template[environment_data['data_name']]['node_text_template'].format_map(row[cols_text].to_dict())
            filter_rule = row.get("filter_rule")
        else:
            # query_text = data_ctdg.node_text[src_id]
            for k in [10, 50, 100]:
                result[f'recall@{k}'] = 0.0
                result[f'hit@{k}'] = 0
            for k2 in [3,10,50]:
                result[f'gnn_recall@{k2}'] = 0.0
                result[f'gnn_hit@{k2}'] = 0
            continue
        
        for k in [10, 50, 100]:
            candidate_dst_ids = execute_search_dst_toolkit(query_text,
                                                            dx_src, 
                                                            dst_node_ids,
                                                            src_idx,
                                                            bert_embedder,
                                                            environment_data,
                                                            data_ctdg,
                                                            data_ctdg.interaction_cache,
                                                            filter_rule,
                                                            recall_common_neighbor=recall_common_neighbor,
                                                            # recall_inductive=recall_inductive,
                                                            recall_topk=k,
                                                            use_src_node_text = args.use_src_node_text
                                                            # recall_alpha=3
                                                        )
            if k == 100 and rewarder.reward_sel == "gnn":
                # 获取每个候选 dst 的 GNN 得分
                scores = rewarder.reward(
                    src_idx * np.ones_like(candidate_dst_ids["dst_ids"]),
                    np.array(candidate_dst_ids["dst_ids"]),
                    final_pred_time * np.ones_like(candidate_dst_ids["dst_ids"])
                )
                scores = scores.detach().numpy()
                # 按得分从高到低排序（假设越高越合理）
                sorted_indices = np.argsort(-scores)  # 降序排列
                reranked_dst_ids = [candidate_dst_ids["dst_ids"][i] for i in sorted_indices]
                
                # 保存重排序结果
                candidate_dst_ids["dst_ids2"] = reranked_dst_ids
                for k2 in [3, 10, 50]:
                    pred_ids = set(reranked_dst_ids[:k2])
                    inter = pred_ids & gt_dst_idxs
                    recall = len(inter) / len(gt_dst_idxs) if gt_dst_idxs else 0.0
                    hit = int(len(inter) > 0)

                    result[f'gnn_recall@{k2}'] = recall
                    result[f'gnn_hit@{k2}'] = hit
                


            pred_set = set(candidate_dst_ids["dst_ids"])
            inter = pred_set & gt_dst_idxs
            recall = len(inter) / len(gt_dst_idxs) if gt_dst_idxs else 0.0
            hit = int(len(inter) > 0)

            result[f'recall@{k}'] = recall
            result[f'hit@{k}'] = hit
        # 添加到总列表
        results_list.append(result)


    df_results = pd.DataFrame(results_list)
    groups = {
        'Hub': df_results['is_hub'],
        'Normal': ~df_results['is_hub'],
        'All': pd.Series([True] * len(df_results))  # 全部样本
    }

    

    # 构建结果字典
    summary_data = {}

    for group_name, mask in groups.items():
        group_data = {}
        for metric in metrics:
            for k in ks:
                col = f"{metric}@{k}"
                try:
                    values = df_results[mask][col]
                    group_data[col] = values.mean() if len(values) > 0 else 0.0
                except:
                    continue
        summary_data[group_name] = group_data

    # 转为 DataFrame，并排序列：Recall@10, Recall@50, ..., Hit@10, ...
    df_summary = pd.DataFrame(summary_data).T  # 转置：group 为行
    df_summary = df_summary.round(4)
    

    # 可选：按列名排序（Recall 在前，Hit 在后，按 k 排序）
    sorted_cols = sorted(df_summary.columns, key=lambda x: (x.split('@')[0], int(x.split('@')[1])))
    df_summary = df_summary[sorted_cols]
    # 设置索引名
    df_summary.index.name = 'Group'
    df_summary["format_rate"] = round(1 - (fail_count / query_examples_all_result.shape[0]), 4)
    result_dir = os.path.dirname(args.query_result_path).replace("LLMGGen/results", "LLMGGen/reports")
    os.makedirs(result_dir,exist_ok=True)
    
    if not args.use_src_node_text:
        df_summary.to_csv(os.path.join(result_dir, "dst_retrival_matrix_raw.csv"))
    else:
        # with filter pipeline
        df_summary.to_csv(os.path.join(result_dir, "dst_retrival_matrix.csv"))


def process_query_result_group(
                 teacher_forcing:bool,
                 args,
                 gen_col:str = "generate_results",
                 recall_common_neighbor: bool = True,
                 recall_inductive: bool = False,
                 ):
    

    query_examples_all_result = pd.read_csv(args.query_save_path)
    assert gen_col in query_examples_all_result.columns and "src_idx" in query_examples_all_result.columns, f"gen_col {gen_col} not in query_examples_all_result"
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=True
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    if not teacher_forcing:
        data_ctdg.load_degree_predictor_results(args.dx_src_path)

    
    bert_embedder = BertEmbedder()
    query_parser = RegexTaggedContentParser(
        required_keys=Dataset_Template[environment_data['data_name']]['node_text_cols'],
    )
    rewarder = DstReward(args)
    
    dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
    query_examples_all_result = assign_difficulty(query_examples_all_result)
    
    parsed_results = []
    for idx, row in query_examples_all_result.iterrows():
        try:
            parsed_results.append({
                "parsed": query_parser.parse(ModelResponse(row[gen_col])).parsed,
                "success": True
            } )
        except Exception as e:
            parsed_results.append({
                "parsed": None,
                "success": False
            })
            
    fail_count = sum(1 for result in parsed_results if not result["success"])
    print(f"解析失败的数量: {fail_count}")

    query_examples_all_result["success"] = [result["success"] for result in parsed_results]
    for col in Dataset_Template[environment_data['data_name']]['node_text_cols']:
        query_examples_all_result[col] = [result["parsed"].get(col, None) if result["success"] else None for result in parsed_results]
    
    

    query_examples_all_result = assign_difficulty(query_examples_all_result)
    cols_text = Dataset_Template[environment_data['data_name']]['node_text_cols']

    dst_node_ids = np.arange(environment_data['dst_min'], 
                             environment_data['dst_max'] + 1)
    final_pred_time = data_ctdg.unique_times[-1]
    # 初始化存储结构
    results_list = []

    # 所有要统计的指标和 topk
    metrics = ['recall', 'hit']
    ks = [3, 10, 50, 100]
    if rewarder.reward_sel == "gnn":
        metrics.extend(['gnn_recall','gnn_hit'])

    # Group by src_idx and process each group
    grouped_df = query_examples_all_result.groupby('src_idx')
    
    
    
    print("Collecting all candidate edges...")
    for src_idx, group in tqdm(grouped_df, "processing src groups"):
        src_idx = int(float(src_idx))
        is_hub = row["difficulty"] < 3
        # Store all edges for frequency analysis
        result = {
            'src_idx': src_idx,
            'is_hub': is_hub
        }
        
        # Collect all candidate edges for this src_idx
        for k in ks:
            all_edges = []
            for idx, row in group.iterrows():
                dx_src, gt_dst_idxs = row["gt_dx_src_unique"], row["gt_dst_idxs_unique"]
                gt_dst_idxs = set(eval(gt_dst_idxs))
                
                if row["success"]:
                    query_text = Dataset_Template[environment_data['data_name']]['node_text_template'].format_map(row[cols_text].to_dict())
                    filter_rule = row.get("filter_rule")
                else:
                    # For failed parsing, use empty query
                    query_text = ""
                    filter_rule = None
                # Get candidate destinations
                candidate_dst_ids = execute_search_dst_toolkit(
                    query_text,
                    dx_src, 
                    dst_node_ids,
                    src_idx,
                    bert_embedder,
                    environment_data,
                    data_ctdg,
                    data_ctdg.interaction_cache,
                    filter_rule,
                    recall_common_neighbor=recall_common_neighbor,
                    recall_topk=k  # Use max K to get enough candidates
                )
            
                # Add edges to all_edges list
                for dst_id in candidate_dst_ids["dst_ids"]:
                    all_edges.append((src_idx, dst_id))
        
            # Count frequency of each edge
            from collections import Counter
            edge_counts = Counter(all_edges)
            
            # Sort edges by frequency (descending)
            # Sort edges by frequency (descending)
            sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Extract sorted dst_ids based on frequency
            sorted_dst_ids = [edge[0][1] for edge in sorted_edges]  # Get dst_id from (src_id, dst_id)
            
            pred_ids = set(sorted_dst_ids[:k]) if len(sorted_dst_ids) >= k else set(sorted_dst_ids)
            inter = pred_ids & gt_dst_idxs
            recall = len(inter) / len(gt_dst_idxs) if gt_dst_idxs else 0.0
            hit = int(len(inter) > 0)
            
            result[f'recall@{k}'] = recall
            result[f'hit@{k}'] = hit
            
            results_list.append(result)
    
    print("Calculating metrics for each src group...")
    # Process each group with frequency-based sorting
    
    df_results = pd.DataFrame(results_list)
    groups = {
        'Hub': df_results['is_hub'],
        'Normal': ~df_results['is_hub'],
        'All': pd.Series([True] * len(df_results))  # 全部样本
    }

    

    # 构建结果字典
    summary_data = {}

    for group_name, mask in groups.items():
        group_data = {}
        for metric in metrics:
            for k in ks:
                col = f"{metric}@{k}"
                try:
                    values = df_results[mask][col]
                    group_data[col] = values.mean() if len(values) > 0 else 0.0
                except:
                    continue
        summary_data[group_name] = group_data

    # 转为 DataFrame，并排序列：Recall@10, Recall@50, ..., Hit@10, ...
    df_summary = pd.DataFrame(summary_data).T  # 转置：group 为行
    df_summary = df_summary.round(4)

    # 可选：按列名排序（Recall 在前，Hit 在后，按 k 排序）
    sorted_cols = sorted(df_summary.columns, key=lambda x: (x.split('@')[0], int(x.split('@')[1])))
    df_summary = df_summary[sorted_cols]

    # 设置索引名
    df_summary.index.name = 'Group'
    result_dir = os.path.dirname(args.query_result_path).replace("LLMGGen/results", "LLMGGen/reports")
    os.makedirs(result_dir,exist_ok=True)
    df_summary.to_csv(os.path.join(result_dir, "dst_retrival_matrix_group.csv"))





def execute_search_edge_label_toolkit(
                           edge_label_text: str,
                           src_id: int,
                           bert_embedder: BertEmbedder, 
                           environment_info: dict,
                           environment_data:BWRCTDGDataset,
                           interaction_cache:dict = {},
                           recall_common_neighbor:bool = True,
                           recall_alpha:int = 5):
    try:
        
        edge_label_embedding = bert_embedder.get_embedding(edge_label_text)
     
            
        label_features = torch.tensor(environment_data.label_feature)
        similarities = torch.nn.functional.cosine_similarity(edge_label_embedding, label_features)
        top_k = min(recall_alpha, len(label_features))
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
                # 保持相对顺序去重，先合并，再去重
                merged_ids = list(candidate_edge_label_n_ids) + list(candidate_edge_label_ids)
                seen = set()
                candidate_edge_label_ids = [x for x in merged_ids if not (x in seen or seen.add(x))]
                
        if isinstance(candidate_edge_label_ids, Iterable) and len(candidate_edge_label_ids) > top_k:
            candidate_edge_label_ids = candidate_edge_label_ids[:top_k]
        
        if not isinstance(candidate_edge_label_ids, (list, np.ndarray, torch.Tensor)):
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
        
        print("error in execute_search_edge_label_toolkit", e)
        return {}
        
        
def get_eval_edge_text_prompt(
        edge_examples_all_result:pd.DataFrame,
        dataset_name
    ):
    eval_prompts = []
    for idx, row in edge_examples_all_result.iterrows():
        eval_prompt = ACTOR_JUDGE.format(
            goal = Dataset_Template[dataset_name]["goal"],
            prompt=select_to_last_period(row["prompt"], upper_token=2048),
            response=select_to_last_period(row["edge_text"], upper_token=512),
            reference=select_to_last_period(row["gt_text"], upper_token=512),
        )
        eval_prompts.append({
            "prompt": eval_prompt,
            "edge_id": row["edge_id"],
        })
    eval_prompts = pd.DataFrame(eval_prompts)
    return eval_prompts
        
        
def process_edge_result(args,
                        teacher_forcing=True,
                        gen_col = "predict"
                        ):
    edge_examples_all_result = pd.read_csv(args.edge_save_path)
    assert gen_col in edge_examples_all_result.columns and "src_idx" in edge_examples_all_result.columns, f"gen_col {gen_col} not in edge_examples_all_result"
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    bert_embedder = BertEmbedder()
    edge_parser = RegexTaggedContentParser(
        required_keys=Dataset_Template[environment_data['data_name']]['edge_text_cols'],
    )
    
    
    parsed_results = []
    for idx, row in edge_examples_all_result.iterrows():
        try:
            parsed_results.append({
                "parsed": edge_parser.parse(ModelResponse(row[gen_col])).parsed,
                "success": True
            } )
        except Exception as e:
            parsed_results.append({
                "parsed": None,
                "success": False
            })
            
    fail_count = sum(1 for result in parsed_results if not result["success"])
    print(f"解析失败的数量: {fail_count}")

    edge_examples_all_result["success"] = [result["success"] for result in parsed_results]
    for col in Dataset_Template[environment_data['data_name']]['edge_text_cols']:
        edge_examples_all_result[col] = [result["parsed"].get(col, None) if result["success"] else None for result in parsed_results]
    
    
    rewarder = EdgeReward(args)
    edges_all = []
    
    grouped_df = edge_examples_all_result.groupby('edge_id')
    
    # debug
    if "t" not in edge_examples_all_result.columns:
        edge_examples_all_result["t"] = np.repeat(
            data_ctdg.unique_times[-1], 
            edge_examples_all_result.shape[0]
        )
    
    if "ts_str" in edge_examples_all_result.columns and not teacher_forcing:
        # 尝试将ts_str列转回t（时间戳），如果失败则保留原t数值
        def safe_str_to_timestamp(row):
            try:
                return int(datetime.strptime(row["ts_str"], "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                return row["t"]
        edge_examples_all_result["t"] = edge_examples_all_result.apply(safe_str_to_timestamp, axis=1)
        
    if teacher_forcing:
        # append_cols = ["gt_label", "gt_text"]
        edge_examples_all_result["gt_label"] = edge_examples_all_result["gt_label"].map(lambda x: [int(x)])
        edge_examples_all_result.rename(columns={"output": "gt_text"}, inplace=True)
    else:
        for idx, row in edge_examples_all_result.iterrows():
            src_idx = int(float(row["src_idx"]))
            src_future_labels = data_ctdg.ctdg.label[data_ctdg.ctdg.src==src_idx]
            src_future_edge_idxs = data_ctdg.ctdg.edge_id[data_ctdg.ctdg.src==src_idx]
            dst_idx = int(float(row["dst_idx"]))
            dst_future_labels = data_ctdg.ctdg.label[data_ctdg.ctdg.dst==dst_idx]
            dst_future_edge_idxs = data_ctdg.ctdg.edge_id[data_ctdg.ctdg.dst==dst_idx]
            future_edge_idxs = [*src_future_edge_idxs, *dst_future_edge_idxs]
            if len(future_edge_idxs) > 3:
                future_edge_idxs = future_edge_idxs[:3]
            future_edge_labels = [*src_future_labels, *dst_future_labels]
            if len(future_edge_labels) > 3:
                future_edge_labels = future_edge_labels[:3]
            gt_edge_text_ref = "\n".join([f"{data_ctdg.ctdg.edge_text[edge_idx]}" for edge_idx in future_edge_idxs])
            gt_edge_lable_ref = future_edge_labels
            edge_examples_all_result.loc[idx, "gt_label"] = gt_edge_lable_ref
            edge_examples_all_result.loc[idx, "gt_text"] = gt_edge_text_ref
        
        
    for edge_id, group in tqdm(grouped_df, "processing edge examples"):
        edge_id = int(float(edge_id))
        candidate_edge_labels_all = []
        
        for _, row in group.iterrows():
            
            if row["success"]:
                label_text = row["label"]
            else:
                label_text = ""
                
            candidate_edge_labels = execute_search_edge_label_toolkit(label_text,
                                                                row["src_idx"],
                                                                bert_embedder,
                                                                environment_data,
                                                                data_ctdg,
                                                                data_ctdg.interaction_cache,
                                                                recall_common_neighbor=True,
                                                                recall_alpha=5
                                                            )
            
            row["edge_label"] = candidate_edge_labels["label"]
            row["edge_text"] = Dataset_Template[environment_data['data_name']]['edge_text_template'].format_map(row.to_dict())
            row = row[["src_idx", "dst_idx", "t", "prompt", "edge_id", "edge_label", "edge_text", "gt_label", "gt_text"]]

           
            score = rewarder.reward(np.array([row["src_idx"]]),
                                    np.array([row["dst_idx"]]),
                                    np.array([row["t"]]),
                                    int(row["edge_id"]))
            
            candidate_edge_labels_all.append((pd.DataFrame([row]),score))
        # 按照score对candidate_dst_ids_all进行排序，由高到低
        candidate_edge_labels_all.sort(key=lambda x: x[1], reverse=True)
        edges_all.append(candidate_edge_labels_all[0][0])
    
    
    edges_all = pd.concat(edges_all,ignore_index=True) # 含有src,dst,t,edge_id,edge_label,edge_text
    
    if teacher_forcing:
        eval_prompts = get_eval_edge_text_prompt(edges_all, dataset_name=bwr_ctdg.data_name)
        prompt_dir = os.path.dirname(args.edge_save_path).replace("results", "prompts")
        os.makedirs(prompt_dir, exist_ok=True)


        eval_prompts.to_csv(os.path.join(prompt_dir, 'edge_text_eval_prompt.csv'), index=False)
        edges_all.to_csv(args.edge_result_path, index=False)
        
        print(f"Edge text examples prompt mean length: {eval_prompts['prompt'].str.len().mean():.2f}")
        print(f"Edge text examples prompt max length: {eval_prompts['prompt'].str.len().max()}")


def process_edge_result_idgg(args,
                        teacher_forcing=True,
                        gen_col = "predict"
                        ):
    edge_examples_all_result = pd.read_csv(args.edge_save_path)
    assert gen_col in edge_examples_all_result.columns and "src_idx" in edge_examples_all_result.columns, f"gen_col {gen_col} not in edge_examples_all_result"
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root,args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
    )
    
    environment_data = {
            'dst_min': bwr_ctdg.dst_min,
            'dst_max': bwr_ctdg.dst_max,
            'bwr': bwr_ctdg.bwr,
            'data_name': bwr_ctdg.data_name,
            "description":Dataset_Template[bwr_ctdg.data_name]['description']
        }
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    bert_embedder = BertEmbedder()
    edge_parser = RegexTaggedContentParser(
        required_keys=Dataset_Template[environment_data['data_name']]['edge_text_cols'],
    )
    
    
    parsed_results = []
    for idx, row in edge_examples_all_result.iterrows():
        try:
            parsed_results.append({
                "parsed": edge_parser.parse(ModelResponse(row[gen_col])).parsed,
                "success": True
            } )
        except Exception as e:
            parsed_results.append({
                "parsed": None,
                "success": False
            })
            
    fail_count = sum(1 for result in parsed_results if not result["success"])
    print(f"解析失败的数量: {fail_count}")

    edge_examples_all_result["success"] = [result["success"] for result in parsed_results]
    for col in Dataset_Template[environment_data['data_name']]['edge_text_cols']:
        edge_examples_all_result[col] = [result["parsed"].get(col, None) if result["success"] else None for result in parsed_results]
    
    
    rewarder = EdgeReward(args)
    edges_all = []
    
    grouped_df = edge_examples_all_result.groupby('edge_id')
    
    # debug
    if "t" not in edge_examples_all_result.columns:
        edge_examples_all_result["t"] = np.repeat(
            data_ctdg.unique_times[-1], 
            edge_examples_all_result.shape[0]
        )
    
    if "ts_str" in edge_examples_all_result.columns and not teacher_forcing:
        # 尝试将ts_str列转回t（时间戳），如果失败则保留原t数值
        def safe_str_to_timestamp(row):
            try:
                return int(datetime.strptime(row["ts_str"], "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                return row["t"]
        edge_examples_all_result["t"] = edge_examples_all_result.apply(safe_str_to_timestamp, axis=1)
        
    if teacher_forcing:
        # append_cols = ["gt_label", "gt_text"]
        edge_examples_all_result["gt_label"] = edge_examples_all_result["gt_label"].map(lambda x: [int(x)])
        edge_examples_all_result.rename(columns={"output": "gt_text"}, inplace=True)
    else:
        # for idx, row in edge_examples_all_result.iterrows():
        #     src_idx = int(float(row["src_idx"]))
        #     src_future_labels = data_ctdg.ctdg.label[data_ctdg.ctdg.src==src_idx]
        #     src_future_edge_idxs = data_ctdg.ctdg.edge_id[data_ctdg.ctdg.src==src_idx]
        #     dst_idx = int(float(row["dst_idx"]))
        #     dst_future_labels = data_ctdg.ctdg.label[data_ctdg.ctdg.dst==dst_idx]
        #     dst_future_edge_idxs = data_ctdg.ctdg.edge_id[data_ctdg.ctdg.dst==dst_idx]
        #     future_edge_idxs = [*src_future_edge_idxs, *dst_future_edge_idxs]
        #     if len(future_edge_idxs) > 3:
        #         future_edge_idxs = future_edge_idxs[:3]
        #     future_edge_labels = [*src_future_labels, *dst_future_labels]
        #     if len(future_edge_labels) > 3:
        #         future_edge_labels = future_edge_labels[:3]
        #     gt_edge_text_ref = "\n".join([f"{data_ctdg.edge_text[edge_idx]}" for edge_idx in future_edge_idxs])
        #     gt_edge_lable_ref = future_edge_labels
        #     edge_examples_all_result.loc[idx, "gt_label"] = list(int(x) for x in gt_edge_lable_ref)
        #     edge_examples_all_result.loc[idx, "gt_text"] = gt_edge_text_ref

        pass
        
        
    for edge_id, group in tqdm(grouped_df, "processing edge examples"):
        edge_id = int(float(edge_id))
        candidate_edge_labels_all = []
        
        for _, row in group.iterrows():
            
            if row["success"]:
                label_text = row["label"]
            else:
                label_text = ""
                
            candidate_edge_labels = execute_search_edge_label_toolkit(label_text,
                                                                row["src_idx"],
                                                                bert_embedder,
                                                                environment_data,
                                                                data_ctdg,
                                                                data_ctdg.interaction_cache,
                                                                recall_common_neighbor=True,
                                                                recall_alpha=5
                                                            )
            
            row["edge_label"] = candidate_edge_labels["label"]
            row["edge_text"] = Dataset_Template[environment_data['data_name']]['edge_text_template'].format_map(row.to_dict())
            try:
                row = row[["src_idx", "dst_idx", "t", "prompt", "edge_id", "edge_label", "edge_text", "gt_label", "gt_text"]]
            except:
                row = row[["src_idx", "dst_idx", "t", "prompt", "edge_id", "edge_label", "edge_text"]]
           
            score = rewarder.reward(np.array([row["src_idx"]]),
                                    np.array([row["dst_idx"]]),
                                    np.array([row["t"]]),
                                    int(row["edge_id"]))
            
            candidate_edge_labels_all.append((pd.DataFrame([row]),score))
        # 按照score对candidate_dst_ids_all进行排序，由高到低
        candidate_edge_labels_all.sort(key=lambda x: x[1], reverse=True)
        edges_all.append(candidate_edge_labels_all[0][0])
    edges_all = pd.concat(edges_all,ignore_index=True) # 含有src,dst,t,edge_id,edge_label,edge_text
    edges_all.to_csv(args.edge_result_path, index=False)
    
    if teacher_forcing:    
        eval_prompts = get_eval_edge_text_prompt(edges_all, dataset_name=bwr_ctdg.data_name)
        prompt_dir = os.path.dirname(args.edge_save_path).replace("results", "prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        eval_prompts.to_csv(os.path.join(prompt_dir, 'edge_text_eval_prompt.csv'), index=False)
        
        print(f"Edge text examples prompt mean length: {eval_prompts['prompt'].str.len().mean():.2f}")
        print(f"Edge text examples prompt max length: {eval_prompts['prompt'].str.len().max()}")
    


def main_seq_dst(args):
    """
    生成用于sequential recommendation格式的数据
    数据格式: [{'prompt': ..., 'src_id': ..., 'dst_id': ...}, ...]
    """
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root, args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
    )
    
    environment_data = {
        'dst_min': bwr_ctdg.dst_min,
        'dst_max': bwr_ctdg.dst_max,
        'bwr': bwr_ctdg.bwr,
        'data_name': bwr_ctdg.data_name,
        "description": Dataset_Template[bwr_ctdg.data_name]['description']
    }
    
    # 根据split参数选择数据集
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")

    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )
    
    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/seq')
    os.makedirs(prompt_dir, exist_ok=True)
    query_all_examples = []
    
    # 遍历数据集中的每个batch
    for batch_idx, batch_data in enumerate(data_ctdg_loader):
        non_zero_mask = batch_data['src_node_degree'] > 0
        non_zero_indices = np.where(non_zero_mask)
       
        # 处理每个非零degree的源节点
        for batch_inter_idx, bwr_idx, pred_idx in zip(non_zero_indices[0], 
                                                  non_zero_indices[1], 
                                                  non_zero_indices[2]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
           
            # 生成sequential recommendation格式的数据
            seq_dst_examples = process_single_seq_dst(
                src_id, environment_data,
                data_ctdg
            )
            
            query_all_examples.extend(seq_dst_examples)
    
    # 保存结果
    query_all_examples = pd.DataFrame(query_all_examples) 
    query_all_examples['domain'] = 'dst_rule'
    query_all_examples.to_csv(os.path.join(prompt_dir, 'seq_dst.csv'), index=False)


def main_seq_edge(args):
    """
    生成用于edge generation的sequential数据
    数据格式: [{'prompt': ..., 'gt_edge_text': ...}, ...]
    """
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root, args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
    )
    
    environment_data = {
        'dst_min': bwr_ctdg.dst_min,
        'dst_max': bwr_ctdg.dst_max,
        'bwr': bwr_ctdg.bwr,
        'data_name': bwr_ctdg.data_name,
        "description": Dataset_Template[bwr_ctdg.data_name]['description']
    }
    
    # 根据split参数选择数据集
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/seq')
    os.makedirs(prompt_dir, exist_ok=True)
    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )
    query_all_examples = []
    
    # 遍历数据集中的每个batch
    for batch_idx, batch_data in enumerate(data_ctdg_loader):
        
        # 处理每个源节点
        for batch_inter_idx in range(len(batch_data['src_node_ids'])):
            for bwr_idx in range(len(batch_data['src_node_ids'][batch_inter_idx])):
                src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
                input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
                
                # 获取输出边信息
                output_edge_ids = np.array(data_ctdg.output_edges_dict[src_id])
                if len(output_edge_ids) == 0:
                    continue
                    
                dst_ids = data_ctdg.get_dst_ids(output_edge_ids)
                if not isinstance(dst_ids, torch.Tensor):
                    dst_ids = torch.tensor(dst_ids)
                
                # 生成sequential edge格式的数据
                seq_edge_examples = process_single_seq_edge(
                    src_id, dst_ids.tolist(), output_edge_ids.tolist(),
                    environment_data, data_ctdg, args,
                    data_ctdg.interaction_cache, input_edge_ids, "gt", args.few_shot
                )
                
                query_all_examples.extend(seq_edge_examples)
    
    

    # 保存结果
    query_all_examples = pd.DataFrame(query_all_examples) 
    query_all_examples['domain'] = 'edge_text_rule'
    query_all_examples.to_csv(os.path.join(prompt_dir, 'seq_edge.csv'), index=False)


def process_single_seq_dst(
    src_id: int,
    environment_data: Dict,
    data: BWRCTDGDataset,
) -> List[Dict]:
    """
    生成sequential recommendation格式的单个样本
    """
    # 获取源节点文本
    src_node_text = data.get_src_node_texts(src_id,good_metric = [])
    content_hint = f"""
    {chr(10).join([
        *[f"<{k}>{v}</{k}>" for k,v in Dataset_Template[environment_data['data_name']]['node_text_hint'].items()],
        
    ])}
"""
    format_instruction = (
        "You should respond a xml object in a xml fenced code block as "
        f"follows:\n```xml\n{content_hint}\n```"
    )
    # 获取真实的目标节点ID
    output_edge_ids = np.array(data.output_edges_dict[src_id])
    gt_dst_node_ids = data.get_dst_ids(output_edge_ids).unique().cpu().tolist()
    
    examples = []
    for dst_id in gt_dst_node_ids:    # 获取历史交互的目标节点文本
        memory_dst_texts = data.get_memory_seq_dst(src_id, dst_id)
        
        # 构建prompt，不包含邻居信息
        agent_text = (
            f"Your task is to depict node text of dst nodes for the src node {src_id}",
            f"You're about to interact with dst nodes in the network.",
            f"{environment_data['description']}",
            f"[For src-node ({src_id}):]",
            f"{src_node_text}",
            f"Here's your interaction history with other destination nodes in the network:",
            f"{select_to_last_period(memory_dst_texts, upper_token=4e3)}",
        )

        
        
        prompt = QUERY_SYS_PROMPT + "\n" +  format_instruction +  "\n".join(agent_text)
        examples.append({
            "prompt": prompt,
            "src_id": src_id,
            "dst_id": dst_id,
            "domain": "dst_rule"
        })
    
    return examples


def process_single_seq_edge(
    src_id: int,
    dst_ids: List[int],
    edge_ids: List[int],
    environment_data: Dict,
    data: BWRCTDGDataset,
    args: Any,
    interaction_cache: Dict,
    input_edge_ids: List,
    type: str = "gt",
    few_shot: int = 0
) -> List[Dict]:
    """
    生成sequential edge格式的单个样本
    """
    examples = []
    
    for dst_id, edge_id in zip(dst_ids, edge_ids):
        try:
            dst_id = dst_id.item()
            edge_id = edge_id.item()
        except:
            pass
            
        # 获取源节点和目标节点文本（只包含自身信息）
        src_node_text = data.get_src_node_texts(src_id, good_metric = [])
        dst_node_text = data.get_src_node_texts(dst_id, good_metric = [])
        
        # 获取历史边文本
        memory_edge_texts = data.get_history_dst_edge_texts(input_edge_ids, dst_id=dst_id)
        
        # 构建prompt，只包含src和dst的node_text，但保留历史交互信息
        agent_text = (
            f"[For src-node ({src_id}):]",
            f"{select_to_last_period(src_node_text, 3e2)}",
            f"[For dst-node ({dst_id}):]",
            f"{select_to_last_period(dst_node_text, 3e2)}",
            f"Here's your interaction history with other destination nodes in the network:",
            f"{select_to_last_period(memory_edge_texts, upper_token=4e3)}",
        )
        
        content_hint = f"""
    {chr(10).join([
        *[f"<{k}>{v}</{k}>" for k,v in Dataset_Template[environment_data['data_name']]['edge_text_hint'].items()]
    ])}
"""
        format_instruction = (
            "You should respond a xml object in a xml fenced code block as "
            f"follows:\n```xml\n{content_hint}\n```"
        )
        
        prompt = EDGE_ATTR_PROMPT + "\n" + format_instruction + "\n".join(agent_text)
        
        gt_edge_text = np.array(data.edge_text[edge_id])
        examples.append({
            "prompt": prompt,
            "gt_edge_text": gt_edge_text
        })
    
    return examples


    


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from .utils.bwr_ctdg import  custom_collate
   
    
    import argparse
    import os
    from datetime import datetime
    import re

    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--data_root', type=str, default="./data", help='data root dir')
    parser.add_argument('--data_name', type=str, default='8days_dytag_small_text', help='数据集名称')
    parser.add_argument('--pred_ratio', type=float, default=0.15, help='预测比例')
    parser.add_argument('--bwr', type=int, default=2048, help='BWR大小')
    parser.add_argument('--time_window', type=int, default=24*60*60, help='时间窗口大小')
    parser.add_argument('--recall_bwr', type=int, default=2048, help='召回BWR大小')
    # parser.add_argument('--query_num', type=int, default=1, help='查询数量')
    parser.add_argument('--split', type=str, default='test', help='数据集分割')
    parser.add_argument('--use_feature', type=str, default='bert', help='whether to use text embeddings as feature') # or Bert
    parser.add_argument('--cm_order', type=bool, default=True, help='是否使用cm_order')

    # evaluation args
    parser.add_argument('--node_msg', type=bool, default=False, help='是否使用节点消息 in graph embedding metric')
    parser.add_argument('--edge_msg', type=bool, default=False, help='是否使用边消息 in graph embedding metric')
    
    # gen graph args
    parser.add_argument('--model_config_name', type=str, default='default', help='模型配置名称')
    parser.add_argument('--rl', action= "store_true", help = "generate rl data")
    parser.add_argument('--idgg_rl', action= "store_true", help = "generate rl data")
    
    parser.add_argument('--infer_dst', action= "store_true", help = "generate infer dst data")
    parser.add_argument('--dx_src_path', type=str, default=None, help='评估查询图的路径')

    parser.add_argument('--infer_edge', action= "store_true", help = "generate infer dst data")
    
    
    # process query result    
    parser.add_argument('--process_query_result', action="store_true", help='process query result for llm generated data') # dst result
    parser.add_argument('--query_save_path', type=str, default=None, help='llm generated query result for dst node selection')
    parser.add_argument('--query_result_path', type=str, default=None, help='processed query result for dst node selection')
    parser.add_argument('--use_src_node_text', action = "store_true", help='query prompt path')
    
    # process edge result    
    parser.add_argument('--process_edge_result', action="store_true", help='process edge result for llm generated data') # dst result
    parser.add_argument('--edge_save_path', type=str, default=None, help='llm generated edge result for edge selection')
    parser.add_argument('--edge_result_path', type=str, default=None, help='processed edge result for edge selection')
    
    # reward selection args
    parser.add_argument('--reward_sel', type=str, default=None, help="奖励选择方式，默认无")

    # gnn reward args
    parser.add_argument('--gnn_model_name', type=str, default=None, help="GNN模型名称")
    parser.add_argument('--gnn_save_model_path', type=str, default=None, help="GNN模型保存路径")
    parser.add_argument('--gen_col', type=str, default="predict", help="llm generated column name")
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--save_root',type=str, default=".", help="save_root")

    ## few_shot prompting
    parser.add_argument('--few_shot', type=int, default=0, help='few shot number')

    ## for sequential data training
    parser.add_argument('--sft', action= "store_true", help = "generate seq data for sft.")
    parser.add_argument('--dst_seq', action= "store_true", help = "generate dst seq data for rl.")
    parser.add_argument('--edge_seq', action= "store_true", help = "generate edge seq data for rl.")


    args = parser.parse_args()

    # # 运行异步主函数
    # # expect, get instruction, inputs, 
    # for rl, inference 
    if args.rl:
        main() # O(N)
    
    if args.idgg_rl:
        main_idgg(dx_src_path = args.dx_src_path) # O(N)
    
    if args.infer_dst:
        assert args.dx_src_path is not None, "must pass degree prediction data for infer"
        main_infer_dst(dx_src_path = args.dx_src_path) # O(N)

    if args.process_query_result:
        if "teacher_forcing" in args.query_save_path:
            if "query_examples.csv" == os.path.basename(args.query_save_path):
                process_query_result(
                                    teacher_forcing=True,
                                    args = args,
                                    gen_col = args.gen_col,
                                    recall_common_neighbor = True,
                                    recall_inductive = False,
                                    )
            else:
                process_query_result_group(args = args,
                                    teacher_forcing=True,
                                    gen_col = args.gen_col,
                                    recall_common_neighbor = True,
                                    recall_inductive = False,
                                    )
        else:
            process_query_result_idgg(args = args,
                             teacher_forcing=False,
                             gen_col = args.gen_col,
                             recall_common_neighbor = True,
                             recall_inductive = False,
                             )
    if args.process_edge_result:
        if "teacher_forcing" in args.edge_save_path:
            process_edge_result(args = args,
                                 teacher_forcing=True,
                                 gen_col = args.gen_col
                                 )
        else:
            process_edge_result_idgg(args = args,
                             teacher_forcing=False,
                             gen_col = args.gen_col
                             )
       
        
    if args.infer_edge:
        assert args.query_result_path is not None, "must pass degree prediction data for infer"
        main_infer_edge(query_result_path = args.query_result_path,
                        dx_src_path=args.dx_src_path) # O(N)


    if args.sft:
        main_inference_offline_cold_start()

    # 添加新的函数调用接口
    if args.dst_seq:
        main_seq_dst(args)

    if args.edge_seq:
        main_seq_edge(args)
