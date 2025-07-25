import torch
import os
import pandas as pd
from torch_geometric.data import TemporalData
import numpy as np
import re

from .utils.bwr_ctdg import (BWRCTDGALLDataset, 
                            BWRCTDGDataset, 
                            Dataset_Template)
from .eval_utils import get_gt_data
from .eval_utils.eval_src_edges import evaluate_all_sources, get_ctdg_edges


def eval_graph_structure(
                max_node_number,    
                gt_graph:TemporalData,
                gen_graph:TemporalData):
    gen_matrix = get_ctdg_edges(gen_graph, max_node_number)
    gt_matrix = get_ctdg_edges(gt_graph, max_node_number)
    eval_matrixs = evaluate_all_sources(gt_matrix,
                                        gen_matrix,
                                        gt_graph,
                                        gen_graph)
    return eval_matrixs
   
    

    
    
def get_gen_data(df: pd.DataFrame,
                data_ctdg: BWRCTDGDataset,
                node_msg: bool = False,
                edge_msg: bool = False,
                ) -> TemporalData:
    # 初始化边列表
    edges = []
    for _, row in df.iterrows():
        src_id = int(float(row["src_idx"]))
        dst_id = int(float(row["dst_idx"]))
        t = int(float(row["t"]))
        if not edge_msg and not node_msg:
            edge = {
            "src_idx": src_id,
            "dst_idx": int(dst_id),
            "t": int(t)
            }
            edges.append(edge)
            continue
        elif node_msg and not edge_msg:
            msg = torch.tensor(np.concatenate([data_ctdg.node_feature[src_id], 
                                                data_ctdg.node_feature[dst_id]]), dtype=torch.float32)
        elif edge_msg and not node_msg:
            msg = torch.tensor(row["edge_msg"], dtype=torch.float32)
        else:
            msg = torch.tensor(np.concatenate([data_ctdg.node_feature[src_id], 
                                                data_ctdg.node_feature[dst_id],
                                                row["edge_msg"]
                                                ]), dtype=torch.float32)
            
        edge = {
            "src_idx": src_id,
            "dst_idx": dst_id,
            "t": t,
            "msg": msg
        }
        edges.append(edge)
    
    src = torch.tensor([edge["src_idx"] for edge in edges], dtype=torch.int64)
    dst = torch.tensor([edge["dst_idx"] for edge in edges], dtype=torch.int64)
    t = torch.tensor([edge["t"] for edge in edges], dtype=torch.int64)
    if not edge_msg and not node_msg:
        return TemporalData(src, dst, t)
    else:
        return TemporalData(src=src, dst=dst, t=t, msg=msg)
    
    
def extract_score_v3(llm_output: str):
        # 修正正则表达式：允许方括号完全缺失
        pattern = r"(CF|PD|DA|IQ|CR):\s*\[?\s*(\d+)\s*\]?.*?"
        
        matches = re.findall(pattern, llm_output, re.IGNORECASE)
        
        # 初始化默认值（全部设为1）
        scores = {'CF': 0, 'PD': 0, 'DA': 0, 'IQ': 0, 'CR': 0}
        
        # 更新匹配到的键值
        for key, value in matches:
            scores[key.upper()] = int(value)  # 转换为大写并存储为整数
        
        # 计算平均分
        total = sum(scores.values())
        average = total / (5*len(scores)) # 0-1
        scores.update({"average": average})
        return scores



def eval_graph_text(
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
        gen_graph_df["label_acc"] = gen_graph_df.apply(calc_label_acc, axis=1)
        # debug
        print(f"标签准确率(label_acc): {gen_graph_df['label_acc'].mean():.4f}")
    
    edge_matrix = {
        **{k: np.mean(gen_eval_result_df[k]) for k in scores_df.columns},
        "label_acc": np.mean(gen_graph_df["label_acc"])
    }
    
    
    return edge_matrix

    
def main(args):
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
    
    df = pd.read_csv(args.graph_result_path)
    if args.edge_text_result_path is not None:
        edge_text_df = pd.read_csv(args.edge_text_result_path)
        edge_matrix = eval_graph_text(df, edge_text_df)
    else:
        edge_matrix = eval_graph_text(df, None)
        
    report_edge_df = pd.DataFrame([edge_matrix])
    report_edge_df.to_csv(args.edge_report_path)
    
    gt_graph = get_gt_data(data_ctdg, 
                           node_msg=args.node_msg,
                           edge_msg=args.edge_msg)
    gen_graph = get_gen_data(df,
                             data_ctdg,
                             node_msg=args.node_msg,
                             edge_msg=args.edge_msg)
    
    eval_matrixs = eval_graph_structure(data_ctdg.node_text.shape[0]-1,
                              gt_graph,
                              gen_graph)
    
    
    
    print(f"评估指标: {eval_matrixs}")
    eval_matrixs["experiment_name"] = args.graph_result_path.replace(".csv", "")
    report_df = pd.DataFrame([eval_matrixs])
    os.makedirs(os.path.dirname(args.graph_report_path), exist_ok=True)
    report_df.to_csv(args.graph_report_path)
    
    
    

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
    parser.add_argument('--split', type=str, default='test', help='数据集分割')
    parser.add_argument('--use_feature', type=str, default='bert', help='whether to use text embeddings as feature') # or Bert
    parser.add_argument('--cm_order', type=bool, default=True, help='是否使用cm_order')

    # evaluation args
    parser.add_argument('--node_msg', action="store_true", help='是否使用节点消息 in graph embedding metric')
    parser.add_argument('--edge_msg', action="store_true", help='是否使用边消息 in graph embedding metric')
    
    # gen graph args
    parser.add_argument('--graph_result_path', type=str, default=None, help='graph result path')
    parser.add_argument('--graph_report_path', type=str, default=None, help='graph result report path')
    
    # edge text eval args
    parser.add_argument('--edge_text_result_path', type=str, default=None, help='edge text eval result path')
    parser.add_argument('--edge_report_path', type=str, default=None, help='edge text eval result report path')
    
    args = parser.parse_args()
    
    main(args)

 


    