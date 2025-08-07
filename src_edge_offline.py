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

[Prompt]:
{prompt}

[ACTOR Action]:
{response}

[Reference Action]:
{reference}

Scoring Logic  
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


Your resposne must follow the format provided below. Please note that only when the content quality is extremely good can 5 Points be given.

[Response Format]:
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

def predict_dst_given_dx(src_id, 
                        dx_src,
                        input_edge_ids,
                        data:BWRCTDGDataset,
                        environment_data:dict,
                        cold_start = False,
                        gt_dst_text = None,
                        interaction_cache:dict = {}
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
                interaction_cache:dict = {}
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

    agent_parser = RegexTaggedContentParser(
        format_instruction=format_instruction,
        required_keys=[*Dataset_Template[environment_data['data_name']]['edge_text_cols']],
    )
    
    return "\n".join(agent_text), agent_parser






from typing import Iterable
def execute_search_dst_toolkit(
                           query_text: str,
                           dx_src: int,
                           dst_node_ids: np.ndarray,
                           src_id: int,
                           bert_embedder: BertEmbedder, 
                           environment_data: dict,
                           data:BWRCTDGDataset,
                           interaction_cache:dict = {},
                           filter_rule = None,
                           recall_common_neighbor:bool = False,
                           recall_inductive: bool = False,
                           recall_alpha:int = 3,
                           append_candidate_ids: list = []
                           ):
    try:
        
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
            
        node_features = torch.tensor(data.node_feature)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, node_features[dst_node_ids])
        top_k = min(recall_alpha*dx_src, len(similarities))
        candidate_dst_r_ids = torch.topk(similarities, k=top_k).indices
        candidate_dst_ids = dst_node_ids[candidate_dst_r_ids].tolist()
        if isinstance(candidate_dst_ids, int):
            candidate_dst_ids = [candidate_dst_ids]

        if recall_common_neighbor:
            n_dst_node_ids = np.array(list(interaction_cache[src_id]['neighbors']))
            node_features = torch.tensor(data.node_feature)
            n_similarities = torch.nn.functional.cosine_similarity(query_embedding, node_features[n_dst_node_ids])
            n_top_k = min(recall_alpha*dx_src, len(n_similarities))
            candidate_dst_r_n_ids = torch.topk(n_similarities, k=n_top_k).indices
            candidate_dst_n_ids = n_dst_node_ids[candidate_dst_r_n_ids.tolist()]
            # 保持相对顺序去重，先合并，再去重
            merged_ids = list(candidate_dst_n_ids) + list(candidate_dst_ids)
            seen = set()
            candidate_dst_ids = [x for x in merged_ids if not (x in seen or seen.add(x))]
            
        if recall_inductive:
            ind_dst_node_ids = np.where(data.new_node_mask)[0]
            ind_node_features = torch.tensor(data.node_feature[data.new_node_mask])
            ind_similarities = torch.nn.functional.cosine_similarity(query_embedding, ind_node_features)
            ind_top_k = min(recall_alpha*dx_src, len(ind_similarities))
            ind_candidate_dst_r_ids = torch.topk(ind_similarities, k=ind_top_k).indices
            ind_candidate_dst_ids = ind_dst_node_ids[ind_candidate_dst_r_ids.tolist()]
            # 保持相对顺序去重，先合并，再去重
            merged_ids = list(ind_candidate_dst_ids) + list(candidate_dst_ids)
            seen = set()
            candidate_dst_ids = [x for x in merged_ids if not (x in seen or seen.add(x))]
        
        

        if isinstance(candidate_dst_ids, Iterable) and len(candidate_dst_ids) > top_k:
            candidate_dst_ids = candidate_dst_ids[:top_k]
        
        candidate_dst_metrics = data.get_dst_nodes_texts(
                                                src_id,
                                                candidate_dst_ids, 
                                                interaction_cache = interaction_cache)
        
        
        return {
            "dst_ids": candidate_dst_ids,
            "dst_metrics": candidate_dst_metrics
        }
        
    except Exception as e:
        return {}
        
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
) -> Dict:
    query_examples = []
    
    query_agent_text, agent_parser = predict_dst_given_dx(
        src_id, dx_src_all, input_edge_ids, data, 
        environment_data, False, None, interaction_cache
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
) -> Dict:
    edge_examples = []
    
    for dst_id, edge_id in zip(dst_ids, edge_ids):
        dst_id = dst_id.item()
        edge_id = edge_id.item()
        agent_text, agent_parser = predict_edge(
            src_id,
            dst_id,
            input_edge_ids,
            data,
            environment_data,
            interaction_cache
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


   
    


def main_infer_edge(query_result_path):
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

    query_examples_all_result = pd.read_csv(query_result_path)
    prompt_dir = os.path.join(args.save_root,f'prompts/{args.data_name}/{args.split}/inference')
    result_dir = os.path.join(args.save_root,f'results/{args.model_config_name}/{args.data_name}/{args.split}/inference')

    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    data_ctdg_loader = DataLoader(data_ctdg, 
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=custom_collate
                                )

    edge_text_examples_all = pd.DataFrame()
    for batch_idx, batch_data in tqdm(enumerate(data_ctdg_loader), "predicting edges"):
        non_zero_indices = np.where(batch_data['src_model_pred_degree']>0)
        for batch_inter_idx, bwr_idx, pred_idx in zip(non_zero_indices[0], 
                                                    non_zero_indices[1], 
                                                    non_zero_indices[2]):
            src_id = batch_data['src_node_ids'][batch_inter_idx][bwr_idx]
            input_edge_ids = batch_data['input_edge_ids'][batch_inter_idx][bwr_idx]
            dst_ids = query_examples_all_result[query_examples_all_result["src_idx"] == src_id]['dst'].tolist()
            output_edge_ids = query_examples_all_result[query_examples_all_result["src_idx"] == src_id]['edge_id'].tolist()
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
    edge_text_examples_all.to_csv(os.path.join(prompt_dir, ' edge_text_examples.csv'), index=False)
    
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
    result_dir = os.path.join(args.save_root,f'results/{args.model_config_name}/{args.data_name}/{args.split}/inference')
   

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
                             False,
                             result_path=query_result_path,
                             interaction_cache=data_ctdg.interaction_cache,
                             recall_common_neighbor=True,
                             recall_inductive=False
                             )
    
    
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
                input_edge_ids
            )
            
            query_all_examples = pd.concat([query_all_examples, 
                                            pd.DataFrame(query_examples)], 
                                            ignore_index=True)
            
    query_all_examples.drop_duplicates(subset=['src_idx'], inplace=True)
    
    columns_to_drop = ["instruction", "input"]
    query_all_examples.drop(columns=[col for col in columns_to_drop if col in query_all_examples.columns], 
            inplace=True)
   
    query_file_name = "query_examples.csv"
    query_all_examples['tag'] = args.split
    query_all_examples['domain'] = 'dst_rule'
    query_all_examples = assign_difficulty(query_all_examples)
    
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
                input_edge_ids
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
    

    query_file_name = "query_examples.csv"
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
    result_dir = os.path.join(args.save_root,f'results/{args.model_config_name}/{args.data_name}/{args.split}/idgg')
   
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    data_ctdg.load_degree_predictor_results(dx_src_path)

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

        # batch_data["src_model_pred_degree"] = batch_data["src_node_degree"]    
        dx_src_all_batch_gt = np.sum(batch_data['src_node_degree'], axis = -1)
        dx_src_all_batch = np.sum(batch_data['src_model_pred_degree'], axis = -1)
        non_zero_indices = np.where(dx_src_all_batch_gt>0)
        dst_node_ids = np.arange(environment_data['dst_min'], environment_data['dst_max'] + 1)
    
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
                    input_edge_ids
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


    query_file_name = "query_examples.csv"
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
    result_dir = os.path.join(args.save_root,f'results/{args.model_config_name}/{args.data_name}/{args.split}/cold_start')

    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
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

    bert_embedder = BertEmbedder("/data/oss_bucket_0/jjr/hf_cache/bert-tiny/")
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
        if self.reward_sel is None: return 0
        
        if self.reward_sel == "gnn":
            return self.gnn_reward_src_query_data(src_ids, dst_ids, interact_times).sum().item()
        
    
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

    
    bert_embedder = BertEmbedder("/data/oss_bucket_0/jjr/hf_cache/bert-tiny/")
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
            pred_ids = np.where(batch_data['src_node_degree'][batch_inter_idx][bwr_idx]>0)[0]
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
    for src_id, group in tqdm(grouped_df, "processing query examples"):
        src_id = int(float(src_id))
        candidate_dst_ids_all = []
        
        for _, row in group.iterrows():
            
            dx_src_list = identifier_map[src_id]["dx_src_list"]
            dst_node_ids = identifier_map[src_id]["dst_node_ids"]
            if row["success"]:
                query_text = Dataset_Template[environment_data['data_name']]['node_text_template'].format_map(row[cols_text].to_dict())
                filter_rule = row.get("filter_rule")
            else:
                query_text = data_ctdg.node_text[src_id]
                filter_rule = None
                
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
                                                                recall_inductive=recall_inductive,
                                                                recall_alpha=3
                                                            )
            candidate_dst_ids["dst_ids"] = candidate_dst_ids["dst_ids"][:np.sum(dx_src_list)]
            # 为每个 pred_idx 分配对应的时间，并根据 dx_src_list[pred_idx] 的数量重复该时间
            times = []
            for pred_idx in identifier_map[row["src_idx"]]["pred_ids"]:
                t = data_ctdg.unique_times[data_ctdg.input_len + int(pred_idx)]
                times.extend([t] * int(dx_src_list[pred_idx]))
                
            query_edges = pd.DataFrame(columns=["src_idx", "dst_idx", "t"])
            for dst_id, t in zip(candidate_dst_ids["dst_ids"], times):
                query_edges.loc[len(query_edges)] = {
                    "src_idx": src_id,
                    "dst_idx": dst_id,
                    "t": int(t)
                }
            
            score = rewarder.reward(query_edges["src_idx"].values,
                                    query_edges["dst_idx"].values,
                                    query_edges["t"].values)
            
            candidate_dst_ids_all.append((query_edges,score))
        
        # score candidates
        # 按照score对candidate_dst_ids_all进行排序，由高到低
        candidate_dst_ids_all.sort(key=lambda x: x[1], reverse=True)
        query_edges_src_all.append(candidate_dst_ids_all[0][0])
        
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
    ):
    eval_prompts = []
    for idx, row in edge_examples_all_result.iterrows():
        eval_prompt = ACTOR_JUDGE.format(
            prompt=row["edge_text"],
            response=row["edge_text"],
            reference=row["gt_text"]
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
    
    bert_embedder = BertEmbedder("/data/oss_bucket_0/jjr/hf_cache/bert-tiny/")
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
            row = row[["src_idx", "dst_idx", "t", "edge_id", "edge_label", "edge_text", "gt_label", "gt_text"]]

           
            score = rewarder.reward(np.array([row["src_idx"]]),
                                    np.array([row["dst_idx"]]),
                                    np.array([row["t"]]),
                                    int(row["edge_id"]))
            
            candidate_edge_labels_all.append((pd.DataFrame([row]),score))
        # 按照score对candidate_dst_ids_all进行排序，由高到低
        candidate_edge_labels_all.sort(key=lambda x: x[1], reverse=True)
        edges_all.append(candidate_edge_labels_all[0][0])
    
    
    edges_all = pd.concat(edges_all,ignore_index=True) # 含有src,dst,t,edge_id,edge_label,edge_text
    eval_prompts = get_eval_edge_text_prompt(edges_all)
    prompt_dir = os.path.dirname(args.edge_save_path)
    eval_prompts.to_csv(os.path.join(prompt_dir, 'edge_text_eval_prompt.csv'), index=False)
    edges_all.to_csv(args.edge_result_path, index=False)
    
    print(f"Edge text examples prompt mean length: {eval_prompts['prompt'].str.len().mean():.2f}")
    print(f"Edge text examples prompt max length: {eval_prompts['prompt'].str.len().max()}")
    
    


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from .utils.bwr_ctdg import  custom_collate
   
    
    import argparse
    import os
    from datetime import datetime

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
    parser.add_argument('--sft', action= "store_true", help = "generate sft data")
    parser.add_argument('--rl', action= "store_true", help = "generate rl data")
    parser.add_argument('--idgg_rl', action= "store_true", help = "generate rl data")
    
    parser.add_argument('--infer_dst', action= "store_true", help = "generate infer dst data")
    parser.add_argument('--dx_src_path', type=str, default=None, help='评估查询图的路径')

    parser.add_argument('--infer_edge', action= "store_true", help = "generate infer dst data")
    
    
    # process query result    
    parser.add_argument('--process_query_result', action="store_true", help='process query result for llm generated data') # dst result
    parser.add_argument('--query_save_path', type=str, default=None, help='llm generated query result for dst node selection')
    parser.add_argument('--query_result_path', type=str, default=None, help='processed query result for dst node selection')
    
    
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
            process_query_result(args = args,
                                 teacher_forcing=True,
                                 gen_col = args.gen_col,
                                 recall_common_neighbor = True,
                                 recall_inductive = False,
                                 )
        else:
            process_query_result(args = args,
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
            process_edge_result(args = args,
                             teacher_forcing=False,
                             gen_col = args.gen_col
                             )
       
        
    if args.infer_edge:
        assert args.query_result_path is not None, "must pass degree prediction data for infer"
        main_infer_edge(query_result_path = args.query_result_path) # O(N)

    #  for sft
    if args.sft:
        main_inference_offline_cold_start() 