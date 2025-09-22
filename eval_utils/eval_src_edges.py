import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter

import pandas as pd
import networkx as nx
import numpy as np

import torch
from torch import nn
from argparse import ArgumentParser
import os
from torch_geometric.data import TemporalData
import copy
# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import re


from .eval_graph_metric import evaluate_graph_metric
from ..jl_metric import JLEvaluator
from ..GraphEmbedding_metric import GraphEmbeddingEvaluator

def compute_metrics(pred_dsts, gt_dsts):
    """
    计算AUC和Precision指标
    
    参数:
    pred_dsts: [N] array, 预测的二值化数组，1表示预测为正，0表示预测为负
    gt_dsts: [N] array, 真实标签的二值化数组，1表示正样本，0表示负样本
    
    返回:
    auc: float, AUC分数
    precision: float, Precision分数
    """
    pred_dsts = np.array(pred_dsts)
    gt_dsts = np.array(gt_dsts)
    pred_dsts = (pred_dsts > 0).astype(int)
    gt_dsts = (gt_dsts > 0).astype(int)
    
    # 计算AUC
    auc = roc_auc_score(gt_dsts, pred_dsts)
    
    # 计算Precision
    # 预测为正的样本数
    pred_pos = np.sum(pred_dsts)
    if pred_pos == 0:
        precision = 0.0
        f1 = 0.0
    else:
        # 预测正确的正样本数
        true_pos = np.sum((pred_dsts == 1) & (gt_dsts == 1))
        precision = true_pos / pred_pos
        
        # 计算召回率
        gt_pos = np.sum(gt_dsts)
        recall = true_pos / gt_pos if gt_pos > 0 else 0.0
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return auc, precision, f1


def dcg_at_k(scores, k):
    return np.sum((np.power(2, scores) - 1) / np.log2(np.arange(2, k + 2)))


def ndcg_at_k(pred_scores, gt_scores, k):
    if len(pred_scores) < k:
        pred_scores = np.concatenate([pred_scores[:k], [0] * (k - len(pred_scores))])
    else:
        pred_scores = pred_scores[:k]
    if len(gt_scores) < k:
        gt_scores = np.concatenate([gt_scores[:k], [0] * (k - len(gt_scores))])
    else:
        gt_scores = gt_scores[:k]
    
    # 计算 DCG 和 IDCG
    dcg = dcg_at_k(pred_scores, k)
    idcg = dcg_at_k(sorted(gt_scores, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0





def evaluate_src(gt_dsts, pred_dsts, k = 20):
    auc, precision, f1 = compute_metrics(pred_dsts, gt_dsts)

    # 获取非零预测值的索引
    non_zero_mask = pred_dsts > 0
    
    # 获取非零预测值及其对应的索引
    non_zero_values = pred_dsts[non_zero_mask]
    non_zero_indices = np.where(non_zero_mask)[0]
    
    # 按预测值降序排序
    sorted_indices = non_zero_indices[np.argsort(-non_zero_values)][:k]

    gt_scores = gt_dsts[sorted_indices]
    pred_scores = pred_dsts[sorted_indices]
    ndcg = ndcg_at_k(pred_scores, gt_scores, k)

    return {
        "precision": precision,
        "f1": f1,
        "auc": auc,
    }
    
    
    
    
def evaluate_src_text(gt_dsts, pred_dsts,):
    pass

def evaluate_hubs(gt_edge_matrix, pred_edge_matrix, k=20):
    """
    评估预测的边矩阵与真实边矩阵之间hub detection的性能指标。
    参数:
    gt_edge_matrix: [N, M] array, 真实边矩阵
    pred_edge_matrix: [N, M] array, 预测边矩阵
    k: int, 用于计算NDCG的前k个节点
    
    返回:
    dict, 包含precision、f1、auc的评估结果
    """
    
    # 计算每行的hub分数
    gt_hub_scores = np.sum(gt_edge_matrix, axis=1)
    pred_hub_scores = np.sum(pred_edge_matrix, axis=1)
    
    gt_hubs = np.argsort(-gt_hub_scores)[:k]
    pred_hubs = np.argsort(-pred_hub_scores)[:k]

    # 创建gt_hubs和pred_hubs的0，1向量
    gt_hubs_vector = np.zeros(gt_edge_matrix.shape[0])
    pred_hubs_vector = np.zeros(pred_edge_matrix.shape[0])
    gt_hubs_vector[gt_hubs] = 1
    pred_hubs_vector[pred_hubs] = 1
    
    # 计算AUC
    auc, f1, precision = compute_metrics(pred_hubs_vector, gt_hubs_vector)
    
    # # 计算NDCG
    # ndcg = ndcg_at_k(pred_hub_scores[gt_hubs], gt_hub_scores[gt_hubs], k)


    return {
        f"precision": precision,
        f"f1": f1,
        f"auc": auc
    }





def jl_all_graph(reference_graph: TemporalData, 
                 generated_graph: TemporalData,
                 max_events: float = 1e6):
    """
    计算JL度量，如果超过max_events则进行截断
    
    参数:
    reference_graph: 参考图
    generated_graph: 生成图
    max_events: 最大事件数阈值
    
    返回:
    dict: 评估结果
    """
    
    # Initialize the evaluator
    evaluator = JLEvaluator(max_events=max_events)
    
    reference_graph = copy.deepcopy(reference_graph)
    generated_graph = copy.deepcopy(generated_graph)
    
    # 检查是否需要截断参考图
    if len(reference_graph.src) > max_events:
        # 随机采样max_events个事件
        indices = np.random.choice(len(reference_graph.src), int(max_events), replace=False)
        indices = np.sort(indices)  # 保持时间顺序
        reference_graph = TemporalData(
            src=reference_graph.src[indices].clone().detach(),
            dst=reference_graph.dst[indices].clone().detach(),
            t=reference_graph.t[indices].clone().detach(),
            msg=reference_graph.msg[indices].clone().detach() if hasattr(reference_graph, 'msg') and reference_graph.msg is not None else torch.zeros((len(indices), 1), dtype=torch.double)
        )
    
    # 检查是否需要截断生成图
    if len(generated_graph.src) > max_events:
        # 随机采样max_events个事件
        indices = np.random.choice(len(generated_graph.src), int(max_events), replace=False)
        indices = np.sort(indices)  # 保持时间顺序
        generated_graph = TemporalData(
            src=generated_graph.src[indices].clone().detach(),
            dst=generated_graph.dst[indices].clone().detach(),
            t=generated_graph.t[indices].clone().detach(),
            msg=generated_graph.msg[indices].clone().detach() if hasattr(generated_graph, 'msg') and generated_graph.msg is not None else torch.zeros((len(indices), 1), dtype=torch.double)
        )
    
    if 'msg' not in reference_graph.keys() and 'msg' not in generated_graph.keys():
        reference_graph.msg = torch.zeros((reference_graph.src.shape[0], 1), dtype=torch.double)
        generated_graph.msg = torch.zeros((generated_graph.src.shape[0], 1), dtype=torch.double)
    
    reference_graph = TemporalData(
        src=reference_graph.src.clone().detach(), 
        dst=reference_graph.dst.clone().detach(), 
        t=reference_graph.t.clone().detach(),
        msg=reference_graph.msg.clone().detach()
    )
    generated_graph = TemporalData(
        src=generated_graph.src.clone().detach(), 
        dst=generated_graph.dst.clone().detach(), 
        t=generated_graph.t.clone().detach(),
        msg=generated_graph.msg.clone().detach()
    )
    
    # Create input dictionary
    input_dict = {
        'reference': reference_graph,
        'generated': generated_graph
    }

    # Evaluate and get results
    try:
        result_dict = evaluator.eval(input_dict)
        return result_dict
    except Exception as e:
        print(f"Error in JL evaluation: {e}")
        return {}



def graph_embedding_all_graph(reference_graph, 
                              generated_graph,
                              node_feature:np.ndarray=None,
                              max_events: float = 1e6):
    """
    计算图嵌入度量，如果超过max_events则进行截断
    
    参数:
    reference_graph: 参考图
    generated_graph: 生成图
    node_feature: 节点特征
    max_events: 最大事件数阈值
    
    返回:
    dict: 评估结果
    """
    
    evaluator = GraphEmbeddingEvaluator(max_events=max_events)
    reference_graph = copy.deepcopy(reference_graph)
    generated_graph = copy.deepcopy(generated_graph)
    
    # 检查是否需要截断参考图
    if len(reference_graph.src) > max_events:
        # 随机采样max_events个事件
        indices = np.random.choice(len(reference_graph.src), int(max_events), replace=False)
        indices = np.sort(indices)  # 保持时间顺序
        reference_graph = TemporalData(
            src=reference_graph.src[indices].clone().detach(),
            dst=reference_graph.dst[indices].clone().detach(),
            t=reference_graph.t[indices].clone().detach(),
            msg=reference_graph.msg[indices].clone().detach() if hasattr(reference_graph, 'msg') and reference_graph.msg is not None else torch.zeros((len(indices), 1), dtype=torch.double)
        )
    
    # 检查是否需要截断生成图
    if len(generated_graph.src) > max_events:
        # 随机采样max_events个事件
        indices = np.random.choice(len(generated_graph.src), int(max_events), replace=False)
        indices = np.sort(indices)  # 保持时间顺序
        generated_graph = TemporalData(
            src=generated_graph.src[indices].clone().detach(),
            dst=generated_graph.dst[indices].clone().detach(),
            t=generated_graph.t[indices].clone().detach(),
            msg=generated_graph.msg[indices].clone().detach() if hasattr(generated_graph, 'msg') and generated_graph.msg is not None else torch.zeros((len(indices), 1), dtype=torch.double)
        )
    
    if 'msg' not in reference_graph.keys() and 'msg' not in generated_graph.keys():
        reference_graph.msg = torch.zeros((reference_graph.src.shape[0], 1), dtype=torch.double)
        generated_graph.msg = torch.zeros((generated_graph.src.shape[0], 1), dtype=torch.double)
    
    reference_graph = TemporalData(
        src=reference_graph.src.clone().detach(), 
        dst=reference_graph.dst.clone().detach(), 
        t=reference_graph.t.clone().detach(),
        msg=reference_graph.msg.clone().detach()
    )
    generated_graph = TemporalData(
        src=generated_graph.src.clone().detach(), 
        dst=generated_graph.dst.clone().detach(), 
        t=generated_graph.t.clone().detach(),
        msg=generated_graph.msg.clone().detach()
    )
    
    input_dict = {
        'reference': reference_graph,
        'reference_node': node_feature,
        'generated': generated_graph,
        'generated_node': node_feature
    }
    
    try:
        result_dict = evaluator.eval(input_dict)
        return result_dict
    except Exception as e:
        print(f"Error in Graph Embedding evaluation: {e}")
        return {}


def evaluate_graphs(gt_edge_matrix, 
                    pred_edge_matrix, 
                    gt_graph, 
                    pred_graph,
                    node_feature:np.ndarray=None):

    
    
    graph_metrics = {}
   
    abs_result = evaluate_graph_metric(gt_edge_matrix, pred_edge_matrix)
    
    pred_result = evaluate_hubs(gt_edge_matrix, pred_edge_matrix, 
                               k=int(0.2 * gt_edge_matrix.shape[0]))

    graph_metrics.update({f"{k}_hub": v for k, v in pred_result.items()})

    pred_result = evaluate_hubs(gt_edge_matrix, pred_edge_matrix, 
                               k=100)

    graph_metrics.update({f"{k}_hub": v for k, v in pred_result.items()})

    edge_overlap = np.sum((gt_edge_matrix > 0) & (pred_edge_matrix > 0)) / np.sum(gt_edge_matrix > 0)
    graph_metrics['edge_overlap'] = edge_overlap

    if "msg" in pred_graph.keys():
        try:
            jl_result = jl_all_graph(gt_graph, pred_graph)
            graph_embedding_result = graph_embedding_all_graph(gt_graph, pred_graph, node_feature)
            graph_metrics.update(jl_result)
            graph_metrics.update(graph_embedding_result)
        except Exception as e:
            print(e)
            pass
    
    graph_metrics.update(abs_result)

    return graph_metrics


def evaluate_graph_snapshots(gt_graphs, pred_graphs):
    from .mmd import evaluate_sampled_graphs
    
    # Filter out pred_graphs that have no edges and no nodes
    filtered_pred_graphs = []
    
    
    for i, ( pred_graph) in enumerate(pred_graphs):
        # Check if pred_graph has any nodes or edges
        if pred_graph.number_of_nodes() > 0 and pred_graph.number_of_edges() > 0:
            filtered_pred_graphs.append(pred_graph)
        else:
            # Optionally print which graphs are being filtered out
            # print(f"Filtering out graph {i} with {pred_graph.number_of_nodes()} nodes and {pred_graph.number_of_edges()} edges")
            pass
    
    # If all graphs were filtered out, return empty results
    if len(filtered_pred_graphs) == 0:
        print("Warning: All predicted graphs were filtered out (no nodes or edges)")
        return {}
    
    # If some graphs were filtered out, print info
    print(f"{len(gt_graphs)},{len(filtered_pred_graphs)}")
    
    mmd_results = evaluate_sampled_graphs(gt_graphs, filtered_pred_graphs)
    return mmd_results

import networkx as nx
import numpy as np

def split_temporal_graph_to_digraph_list(ctdg, unique_times):
    """
    Split temporal graph data into a list of NetworkX DiGraphs, one for each time window.
    
    Args:
        ctdg: TemporalData object containing src, dst, t, and edge_id
        unique_times: array of unique time points
        
    Returns:
        list: List of NetworkX DiGraphs, one for each time window
    """
    
    graph_list = []
    
    # Process each time window
    for t in range(len(unique_times)):
        # Create a DiGraph for this time window
        graph = nx.DiGraph()
        
        # Get edge indices for current time window using the same logic as in the original code
        if t == len(unique_times) - 1:
            mask = (ctdg.t >= unique_times[t])
        elif t == 0:
            mask = (ctdg.t < unique_times[t+1])
        else:
            mask = (ctdg.t >= unique_times[t]) & (ctdg.t < unique_times[t+1])
        
        # Extract edges for this time window
        src_nodes = ctdg.src[mask]
        dst_nodes = ctdg.dst[mask]
        timestamps = ctdg.t[mask]
        
        # Add nodes and edges to the graph
        for i in range(len(src_nodes)):
            src = src_nodes[i].item()
            dst = dst_nodes[i].item()
            timestamp = timestamps[i].item()
            
            # Add nodes if they don't exist
            if not graph.has_node(src):
                graph.add_node(src)
            if not graph.has_node(dst):
                graph.add_node(dst)
            
            # Add edge with attributes
            graph.add_edge(src, dst, timestamp=timestamp)
        
        # Add the graph to our list
        graph_list.append(graph)
    
    return graph_list


    

def evaluate_nodes(gt_edge_matrix, 
                   pred_edge_matrix,
                    test_src_indices:np.ndarray=None,
                    node_text:np.ndarray=None):
    
    src_results = []
    if test_src_indices is None:
        test_src_indices = np.where(gt_edge_matrix.sum(axis=1) > 0)[0]
    # if node_text is not None:
    #     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # 优化文本评估，批量处理所有节点的文本，减少bert_score和ROUGE的调用次数
    # if node_text is not None:
    #     # 先批量收集所有节点的预测和参考文本
    #     all_preds = []
    #     all_refs = []
    #     node_indices = []
    #     for i in test_src_indices:
    #         preds = node_text[pred_edge_matrix[i, :] > 0].astype(str).tolist()
    #         refs = node_text[gt_edge_matrix[i, :] > 0].astype(str).tolist()
    #         all_preds.append("\n".join(preds))
    #         all_refs.append("\n".join(refs))
    #         node_indices.append(i)
    #     # 批量计算ROUGE_L
    #     rouge_scores = [calc_rouge_l(scorer, pred, ref) for pred, ref in zip(all_preds, all_refs)]
    #     # 批量计算BERTScore
    #     P, R, F1 = bert_score(all_preds, all_refs, lang='en', rescale_with_baseline=True)
    #     F1 = F1.cpu().numpy() if hasattr(F1, 'cpu') else F1  # 兼容tensor
    #     # 再遍历节点，填充node_matrix
    #     for idx, i in enumerate(test_src_indices):
    #         node_matrix = {}
    #         node_matrix['ROUGE_L'] = rouge_scores[idx]
    #         node_matrix[f"BERTScore_F1"] = float(F1[idx])
    #         node_matrix.update(evaluate_src(gt_edge_matrix[i, :], pred_edge_matrix[i, :], k=10))
    #         src_results.append(node_matrix)
    # else:
    for i in test_src_indices:
        node_matrix = evaluate_src(gt_edge_matrix[i, :], pred_edge_matrix[i, :], k=10)
        src_results.append(node_matrix)

    src_agg = {
        f"{k}_node": np.mean([result[k] for result in src_results])
        for k in src_results[0].keys()
    }
    
    return src_agg
    


# nltk.download('punkt')
# def calc_bleu(pred, ref):
#     pred_tokens = nltk.word_tokenize(pred)
#     ref_tokens = [nltk.word_tokenize(ref)]  # BLEU expects a list of reference sequences
#     return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1)

def calc_rouge_l(scorer, pred, ref):
    score = scorer.score(ref, pred)
    return score['rougeL'].fmeasure

def evaluate_edge_text(df):
    # df['BLEU'] = df.apply(lambda row: calc_bleu(str(row['edge_text']), str(row['gt_text'])), axis=1)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    df['ROUGE_L'] = df.apply(lambda row: calc_rouge_l(scorer, str(row['edge_text']), str(row['gt_text'])), axis=1)
    preds = df['edge_text'].astype(str).tolist()
    refs = df['gt_text'].astype(str).tolist()
    P, R, F1 = bert_score(preds, refs, lang='en', rescale_with_baseline=True)
    df['BERTScore_F1'] = F1.tolist()
    return df   



    
def extract_score_v3(llm_output: str):
        # 修正正则表达式：允许方括号完全缺失
        pattern = r"(GF|CF|PD|DA|IQ|CR):\s*\[?\s*(\d+)\s*\]?.*?"
        
        matches = re.findall(pattern, llm_output, re.IGNORECASE)
        
        # 初始化默认值（全部设为1）
        scores = {'GF': 0, 'CF': 0, 'PD': 0, 'DA': 0, 'IQ': 0, 'CR': 0}
        
        # 更新匹配到的键值
        for key, value in matches:
            scores[key.upper()] = int(value)  # 转换为大写并存储为整数
        
        # 计算平均分
        total = sum(scores.values())
        average = total / (5*len(scores)) # 0-1
        scores.update({"average": average})
        return scores



def evaluate_edges(
                gen_graph_df:pd.DataFrame,
                gen_eval_result_df:pd.DataFrame=None): # save text_eval_prompt.df
    edge_matrix = pd.DataFrame()
    
    
    if gen_eval_result_df is not None:
        assert gen_graph_df.shape[0] == gen_eval_result_df.shape[0], "gen_graph_df and gen_eval_result_df must have the same number of rows"
        # 对每一行提取score字典，并将每个key作为单独的列
        scores = []
        for idx, row in gen_eval_result_df.iterrows():
            score_dict = extract_score_v3(row["predict"])
            scores.append(score_dict)
        scores_df = pd.DataFrame(scores)
    else:
        scores_df = pd.DataFrame()
   
    
    # 计算label_acc
    def calc_label_acc(row):
        try:
            gt_label = eval(row["gt_label"]) if isinstance(row["gt_label"], str) else row["gt_label"]
        except:
            gt_label = row["gt_label"]
        if not isinstance(gt_label, (list, tuple)):
            gt_label = [gt_label]
        return int(row["edge_label"] in gt_label)
    
    if "edge_label" in gen_graph_df.columns and "gt_label" in gen_graph_df.columns:
        gen_graph_df = evaluate_edge_text(gen_graph_df)
        gen_graph_df["label_acc"] = gen_graph_df.apply(calc_label_acc, axis=1)
        # debug
        print(f"标签准确率(label_acc): {gen_graph_df['label_acc'].mean():.4f}")
        edge_matrix = {}
        for col in ["label_acc", "ROUGE_L", "BERTScore_F1"]:
            if col in gen_graph_df.columns:
                edge_matrix[col] = np.mean(gen_graph_df[col])
        

    edge_matrix = {
        **{k: np.mean(scores_df[k]) for k in scores_df.columns},
        **edge_matrix
    }
    print(f"edge_matrix: {edge_matrix}")
    
    return edge_matrix
    
    return src_agg






# nltk.download('punkt')
# def calc_bleu(pred, ref):
#     pred_tokens = nltk.word_tokenize(pred)
#     ref_tokens = [nltk.word_tokenize(ref)]  # BLEU expects a list of reference sequences
#     return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1)

def calc_rouge_l(scorer, pred, ref):
    score = scorer.score(ref, pred)
    return score['rougeL'].fmeasure

def evaluate_edge_text(df):
    # df['BLEU'] = df.apply(lambda row: calc_bleu(str(row['edge_text']), str(row['gt_text'])), axis=1)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    df['ROUGE_L'] = df.apply(lambda row: calc_rouge_l(scorer, str(row['edge_text']), str(row['gt_text'])), axis=1)
    preds = df['edge_text'].astype(str).tolist()
    refs = df['gt_text'].astype(str).tolist()
    P, R, F1 = bert_score(preds, refs, lang='en', rescale_with_baseline=True)
    df['BERTScore_F1'] = F1.tolist()
    return df   



    
def extract_score_v3(llm_output: str):
        # 修正正则表达式：允许方括号完全缺失
        pattern = r"(GF|CF|PD|DA|IQ|CR):\s*\[?\s*(\d+)\s*\]?.*?"
        
        matches = re.findall(pattern, llm_output, re.IGNORECASE)
        
        # 初始化默认值（全部设为1）
        scores = {'GF': 0, 'CF': 0, 'PD': 0, 'DA': 0, 'IQ': 0, 'CR': 0}
        
        # 更新匹配到的键值
        for key, value in matches:
            scores[key.upper()] = int(value)  # 转换为大写并存储为整数
        
        # 计算平均分
        total = sum(scores.values())
        average = total / (5*len(scores)) # 0-1
        scores.update({"average": average})
        return scores



def evaluate_edges(
                gen_graph_df:pd.DataFrame=None,
                gen_eval_result_df:pd.DataFrame=None): # save text_eval_prompt.df
    edge_matrix = {}
    
    if gen_eval_result_df is not None:
        # assert gen_graph_df.shape[0] == gen_eval_result_df.shape[0], "gen_graph_df and gen_eval_result_df must have the same number of rows"
        # 对每一行提取score字典，并将每个key作为单独的列
        scores = []
        for idx, row in gen_eval_result_df.iterrows():
            score_dict = extract_score_v3(row["predict"])
            scores.append(score_dict)
        scores_df = pd.DataFrame(scores)
        edge_matrix.update({
            **{k: np.mean(scores_df[k]) for k in scores_df.columns}})
        
    else:
        scores_df = pd.DataFrame()
   
    
    # 计算label_acc
    def calc_label_acc(row):
        try:
            gt_label = eval(row["gt_label"]) if isinstance(row["gt_label"], str) else row["gt_label"]
        except:
            gt_label = row["gt_label"]
        if not isinstance(gt_label, (list, tuple)):
            gt_label = [gt_label]
        return int(row["edge_label"] in gt_label)
    
    if gen_graph_df is not None:
        if "edge_label" in gen_graph_df.columns and "gt_label" in gen_graph_df.columns:
            gen_graph_df = evaluate_edge_text(gen_graph_df)
            gen_graph_df["label_acc"] = gen_graph_df.apply(calc_label_acc, axis=1)
            # debug
            print(f"标签准确率(label_acc): {gen_graph_df['label_acc'].mean():.4f}")
            edge_matrix_struct = {}
            for col in ["label_acc", "ROUGE_L", "BERTScore_F1"]:
                if col in gen_graph_df.columns:
                    edge_matrix_struct[col] = np.mean(gen_graph_df[col])
            
        edge_matrix.update(edge_matrix_struct)
    print(f"edge_matrix: {edge_matrix}")

    return edge_matrix

def get_ctdg_edges(data:TemporalData, # 输入的边文件
                    max_node_number
                    ):
    """
    从边文件构建时序图快照并计算度分布
    :param edge_file: 边文件路径
    :param max_node_number: 最大节点数
    :param time_window: 时间窗口大小
    :param undirected: 是否构建无向图
    :return: 出度列表和唯一出度列表
    """
    
    edge_matrix = np.zeros((max_node_number + 1, max_node_number + 1)) # 行是src，列是candidate_dst

    # Convert src and dst to integer type to use as indices
    src = data.src.long() if hasattr(data.src, 'long') else data.src.astype(int)
    dst = data.dst.long() if hasattr(data.dst, 'long') else data.dst.astype(int)
    
    src_dst_pairs = np.stack([src, dst], axis=1)
    unique_pairs, counts = np.unique(src_dst_pairs, axis=0, return_counts=True)
    edge_matrix[unique_pairs[:, 0], unique_pairs[:, 1]] = counts
    
    return edge_matrix



def temporal_data_to_nx_graph(temporal_data: TemporalData, 
                             directed: bool = True) -> nx.Graph:
    """
    Convert TemporalData to NetworkX graph.
    
    Args:
        temporal_data: TemporalData object with src, dst, t, and optionally msg attributes
        directed: Whether to create a directed graph (default: True)
        
    Returns:
        NetworkX graph (DiGraph if directed=True, Graph if directed=False)
    """
    # Create appropriate graph type
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    
    # Get the data arrays
    src_nodes = temporal_data.src.detach().cpu().numpy()
    dst_nodes = temporal_data.dst.detach().cpu().numpy()
    timestamps = temporal_data.t.detach().cpu().numpy()
    
    # Check if message features exist
    has_msg = hasattr(temporal_data, 'msg') and temporal_data.msg is not None
    if has_msg:
        messages = temporal_data.msg.detach().cpu().numpy()
    
    # Add edges to the graph
    for i in range(len(src_nodes)):
        src = int(src_nodes[i])
        dst = int(dst_nodes[i])
        timestamp = int(timestamps[i])
        
        # Add edge with timestamp attribute
        edge_attrs = {'timestamp': timestamp}
        
        # Add message attributes if they exist
        if has_msg:
            edge_attrs['msg'] = messages[i]
            
        graph.add_edge(src, dst, **edge_attrs)
    
    return graph


import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def detect_echo_chambers_simple(
    graph: nx.Graph,
    node_features: np.ndarray,
    similarity_threshold: float = 0.5,
    min_chamber_size: int = 3
) -> Dict:
    """
    最简化的回音室检测方法（修复索引问题版，基于最大连通分量）
    
    参数:
    graph: NetworkX图对象
    node_features: [N, D] array, 每个节点的特征向量（如BERT embedding）
    similarity_threshold: 特征相似度阈值
    min_chamber_size: 回音室的最小节点数
    
    返回:
    dict: 包含回音室检测结果的字典

    https://www.pnas.org/doi/10.1073/pnas.1517441113
    """
    
    # 获取最大连通分量
    if graph.number_of_nodes() == 0:
        return {
            'echo_chamber_ratio': 0.0,
            'num_chambers': 0,
            'avg_chamber_size': 0.0
        }
    
    # 如果是无向图，找到最大的连通组件；如果是有向图，找到最大的弱连通组件
    if isinstance(graph, nx.DiGraph):
        connected_components = nx.weakly_connected_components(graph)
    else:
        connected_components = nx.connected_components(graph)
    
    # 找到最大的连通组件
    largest_component = max(connected_components, key=len)
    subgraph = graph.subgraph(largest_component)
    
    # 获取子图中的节点列表
    subgraph_nodes = list(subgraph.nodes())
    n_subgraph_nodes = len(subgraph_nodes)
    
    if n_subgraph_nodes == 0:
        return {
            'echo_chamber_ratio': 0.0,
            'num_chambers': 0,
            'avg_chamber_size': 0.0
        }
    
    # 限制节点数量以提高计算效率
    max_nodes = min(1000, n_subgraph_nodes)
    if n_subgraph_nodes > max_nodes:
        # 从最大连通分量中随机选择节点
        selected_node_indices = np.random.choice(n_subgraph_nodes, max_nodes, replace=False)
        selected_nodes = [subgraph_nodes[i] for i in selected_node_indices]
    else:
        selected_nodes = subgraph_nodes
    
    # 获取选中节点的特征
    # 需要将节点ID映射到node_features的索引
    selected_features_indices = np.array(selected_nodes)  # 假设节点ID直接对应特征索引
    selected_features = node_features[selected_features_indices]
    
    # 计算节点间的特征相似度
    feature_similarities = cosine_similarity(selected_features)
    
    # 创建节点索引映射
    selected_node_to_idx = {node: idx for idx, node in enumerate(selected_nodes)}
    
    # 创建邻接矩阵（只针对选中的节点）
    adj_matrix = np.zeros((len(selected_nodes), len(selected_nodes)), dtype=bool)
    for edge in subgraph.edges():
        node1, node2 = edge[0], edge[1]
        if node1 in selected_node_to_idx and node2 in selected_node_to_idx:
            idx1, idx2 = selected_node_to_idx[node1], selected_node_to_idx[node2]
            adj_matrix[idx1, idx2] = True
            if not isinstance(subgraph, nx.DiGraph):  # 无向图需要设置对称位置
                adj_matrix[idx2, idx1] = True
    
    # 余下代码保持不变...
    # 4. 找到相似节点
    similarity_mask = feature_similarities > similarity_threshold
    # 减去对角线（自己与自己的相似性）
    np.fill_diagonal(similarity_mask, False)
    
    # 统计每个节点的相似节点数
    similar_node_counts = np.sum(similarity_mask, axis=1)
    
    # 5. 找到候选回音室
    candidate_mask = similar_node_counts >= min_chamber_size
    candidate_indices = np.where(candidate_mask)[0]
    
    if len(candidate_indices) == 0:
        return {
            'echo_chamber_ratio': 0.0,
            'num_chambers': 0,
            'avg_chamber_size': 0.0
        }
    
    # 6. 计算所有候选回音室的内部连接密度
    echo_chamber_nodes = 0
    total_chambers = 0
    echo_chamber_all = []
    
    # 根据公式计算每个节点的回音室指标
    echo_chamber_indicators = np.zeros(len(selected_nodes))
    
    for i in range(len(selected_nodes)):
        # 获取与节点i相似的节点（基于特征相似度）
        similar_nodes = np.where(feature_similarities[i] > similarity_threshold)[0]
        
        if len(similar_nodes) > 0:
            # 计算在邻接矩阵中实际连接的比例
            actual_connections = np.sum(adj_matrix[i, similar_nodes])
            connection_ratio = actual_connections / len(similar_nodes)
            
            # 如果连接比例超过阈值δ，则该节点属于回音室
            if connection_ratio > 0.01:  # 这里的0.1对应公式中的δ
                echo_chamber_indicators[i] = 1
    
    # 计算回音室比例
    echo_chamber_ratio = np.mean(echo_chamber_indicators)
    
    return {
        'echo_chamber_ratio': echo_chamber_ratio,
        'num_chambers': int(np.sum(echo_chamber_indicators)),
        'avg_chamber_size': np.sum(echo_chamber_indicators) / len(selected_nodes) if len(selected_nodes) > 0 else 0
    }

def evaluate_echo_chamber_effect(
    gt_graph: nx.Graph,
    gen_graph: nx.Graph,
    node_features: np.ndarray
) -> Dict:
    """
    评估真实图与生成图中的回音室效应差异
    
    参数:
    gt_graph: 真实图 (NetworkX格式)
    gen_graph: 生成图 (NetworkX格式)
    node_features: 节点特征（BERT embedding）
    node_texts: 节点文本
    
    返回:
    dict: 回音室效应评估结果
    """
    
    # 检测真实图中的回音室
    gt_echo_chambers = detect_echo_chambers_simple(gt_graph, node_features)
    
    # 检测生成图中的回音室
    gen_echo_chambers = detect_echo_chambers_simple(gen_graph, node_features)
    
    # 比较回音室效应
    echo_chamber_metrics = {
        'echo_chamber_ratio_diff': abs(gt_echo_chambers['echo_chamber_ratio'] - gen_echo_chambers['echo_chamber_ratio']),
        'num_chambers_diff': abs(gt_echo_chambers['num_chambers'] - gen_echo_chambers['num_chambers']),
        'avg_chamber_size_diff': abs(gt_echo_chambers['avg_chamber_size'] - gen_echo_chambers['avg_chamber_size'])
    }
    
    return echo_chamber_metrics


import powerlaw 
def calculate_power_law_fitness(graph: nx.Graph,
                                xmin = 2):
    degree_list = [graph.degree(n) for n in graph.nodes()]
    results = powerlaw.Fit(list(degree_list), 
                            discrete=True,
                                # fit_method="KS",
                                xmin=xmin
                                )
    try:
        alpha = results.power_law.alpha
        sigma = results.power_law.sigma
        D = results.power_law.D
        
    except:
        D = 1.0 # default to upper bound of 

    return D
def evaluate_graph_macro_phenomena(
    pred_data: TemporalData,
    gt_data: TemporalData,
    max_node_number: int,
    node_feature: np.ndarray,
):
    pred_edge_matrix = get_ctdg_edges(pred_data, max_node_number)
    gt_edge_matrix = get_ctdg_edges(gt_data, max_node_number)

    graph_metrics = {}
    gt_nx_graph = temporal_data_to_nx_graph(gt_data)

    pred_nx_graph = temporal_data_to_nx_graph(pred_data)
    echo_chambers = evaluate_echo_chamber_effect(gt_nx_graph,
                                 pred_nx_graph,
                                 node_feature)

    graph_metrics.update( echo_chambers)    
    pred_result = evaluate_hubs(gt_edge_matrix, pred_edge_matrix, 
                               k=int(0.2 * gt_edge_matrix.shape[0]))

    graph_metrics.update({f"{k}@20%N_hub": v for k, v in pred_result.items()})

    pred_result = evaluate_hubs(gt_edge_matrix, pred_edge_matrix, 
                               k=100)
    
    graph_metrics.update({f"{k}@100_hub": v for k, v in pred_result.items()})

   

    graph_metrics.update({
        "D": calculate_power_law_fitness(pred_nx_graph)
    })

    return graph_metrics

if __name__ == "__main__":  


    import torch
    from torch_geometric.data import TemporalData
    from eval_utils.get_baseline_graph import get_baseline_graphs
    from tqdm import tqdm
    # 计算tigger和dggen的损失
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default="./data", help='data root dir')
    parser.add_argument('--data_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--bwr', type=int, default=2048, help='BWR参数')
    parser.add_argument('--use_feature', type=str, default="no", help='是否使用特征')
    parser.add_argument('--time_window', type=int, default=24*60*60, help='时间窗口大小')
    parser.add_argument('--pred_ratio', type=float, default=0.15, help='预测比例')
    parser.add_argument('--split', type=str, default='test', help='数据集分割')
    parser.add_argument('--cm_order', type=bool, default=True, help='是否使用cm_order')
    parser.add_argument('--cut_off_baseline', type=str, default="edge", help='cut_off_baseline')
    parser.add_argument('--node_msg', action="store_true", help='是否使用节点消息')
    parser.add_argument('--edge_msg', action="store_true", help='是否使用边消息')
    parser.add_argument('--graph_report_path', type=str, default="", help='graph_matrix.csv')
    parser.add_argument('--graph_list_report_path', type=str, default="", help='graph_list_matrix.csv')
    parser.add_argument('--graph_macro_report_path', type=str, default="", help='graph_macro_matrix.csv')


    args = parser.parse_args()
    
    dfs_snapshot = []

    if args.graph_report_path != "":
        dfs = []
        # for cut_off_baseline in ["edge", "time"]:
        for cut_off_baseline in ["time","edge"]:
            results = []
            args.cut_off_baseline = cut_off_baseline
            test_data, baseline_data_map, max_node_number, node_text, node_feature, pred_len, unique_times \
                = get_baseline_graphs(args)
            test_data = test_data[0]
            pred_times = unique_times[-pred_len:]
            test_edge_matrix = get_ctdg_edges(test_data, max_node_number)

            
            for baseline_name, baseline_data in tqdm(baseline_data_map.items(),
                                                     "evaluating baselines"
            ):
                baseline_data = baseline_data[0]
                baseline_edge_matrix = get_ctdg_edges(baseline_data, max_node_number)
                graph_matrixs = evaluate_graphs(test_edge_matrix,
                                                baseline_edge_matrix,
                                                test_data, 
                                                baseline_data,
                                                node_feature=node_feature)
                graph_matrixs["model"] = f"{baseline_name}_{cut_off_baseline}"
                results.append(graph_matrixs)

            output_path = args.graph_report_path
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            df = pd.DataFrame(results)
            df.to_csv(output_path)
            dfs.append(df)

        # 保存所有结果
        if dfs:  # 如果有graph_report结果
            df_all = pd.concat(dfs, ignore_index=True)  # 添加ignore_index=True
            output_path = args.graph_report_path
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            df_all.to_csv(output_path, index=False)  # 添加index=False避免重复索引
            print(f"All results saved to {output_path}")

    if args.graph_list_report_path != "":
        
        args.cut_off_baseline = "edge"
        
        results = []
        results_snapshot = []
        test_data, baseline_data_map, max_node_number, node_text, node_feature, pred_len, unique_times \
            = get_baseline_graphs(args)
        test_data = test_data[0]
        test_nx_graph = temporal_data_to_nx_graph(test_data)
        for baseline_name, baseline_data in tqdm(baseline_data_map.items(),
        "evaluating baselines"
        ):
            baseline_data = baseline_data[0]
            baseline_nx_graph = temporal_data_to_nx_graph(baseline_data)
            graph_list_matrixs = evaluate_graph_snapshots(
                [test_nx_graph],
                [baseline_nx_graph]
            )
            
            graph_list_matrixs["model"] = f"{baseline_name}"
            results_snapshot.append(graph_list_matrixs)

        output_path = args.graph_list_report_path
        dir = os.path.dirname(output_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        df = pd.DataFrame(results_snapshot)
        df.to_csv(output_path)

    if args.graph_macro_report_path != "":
        
        args.cut_off_baseline = "edge"
        
        test_data, baseline_data_map, max_node_number, node_text, node_feature, pred_len, unique_times \
            = get_baseline_graphs(args)
        test_data = test_data[0]
        results = []
        for baseline_name, baseline_data in tqdm(baseline_data_map.items(),
        "evaluating baselines"
        ):
            baseline_data = baseline_data[0]
            graph_macro_matrixs = evaluate_graph_macro_phenomena(
                    test_data,
                    baseline_data,
                    max_node_number,
                    node_feature
                )
            graph_macro_matrixs["model"] = f"{baseline_name}"
            results.append(graph_macro_matrixs)

        output_path = args.graph_macro_report_path
        dir = os.path.dirname(output_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        df = pd.DataFrame(results)
        df.to_csv(output_path)