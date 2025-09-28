
from torch_geometric.data import TemporalData
import torch

import os
import pandas as pd
import numpy as np
import re
from . import get_gt_data

from datetime import datetime
def get_snapshot_graph(path, 
                       initial_time: int,
                       pred_len:int, 
                       cut_time: int = None,
                       cut_edge_number:int = None,
                       edge_msg: bool = False,
                       time_window:int = 86400):
                       
    if (cut_time is None and cut_edge_number is None) or (cut_time is not None and cut_edge_number is not None):
        raise ValueError("pred_len和cut_edge_number必须且只能设置其中一个")
   
     # 读取CSV文件
    df = pd.read_csv(path) 
    
    

    UPPER = pd.to_datetime("2030-01-01").value//10**9
    def is_integer_value(x):
        try:
            # upper bound, dataset timestamp is all before 2030
            bool_value = int(x) < UPPER
            return bool_value
        except (TypeError, ValueError):
            return False

    mask = df['t'].apply(is_integer_value)
    df = df[mask].copy()
    df['t'] = pd.to_numeric(df['t'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['t'])
    df = df.sort_values(by='t')

    if "idgg" not in path:
        df['t'] *= time_window
        df['t'] += initial_time

    if cut_edge_number:
        df_sub = df.iloc[:cut_edge_number]
    else:
        # 计算时间范围
        min_time = initial_time if df['t'].min() < initial_time else df['t'].min()
        max_time = cut_time if df['t'].max() > cut_time else df['t'].max()
        
        df_sub = df[(df['t'] >= min_time) & (df['t'] <= max_time)]
        
    data = TemporalData(src=torch.tensor(df_sub['src'].values, dtype=torch.long), 
                        dst=torch.tensor(df_sub['dst'].values, dtype=torch.long), 
                        t=torch.tensor(df_sub['t'].values))
    if edge_msg:
        msg_path = os.path.join(os.path.dirname(path),
                                 "edge_embeddings.npy")
        msg = np.load(msg_path)
        # Subset the message data to match the same rows as df_sub
        msg = msg[df_sub.index.values]
        data.msg = msg 

    # 对于DGGen，dymond，tigger，timestamp应该是day/year * timewindow
    
           
    return data

def get_baseline_graphs(args):
    from LLMGGen.utils.bwr_ctdg import BWRCTDGALLDataset, BWRCTDGDataset
    dataset = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=os.path.join(args.data_root, args.data_name),
        use_feature=args.use_feature,
        cm_order=args.cm_order,
    )
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = dataset.train_data
    elif args.split == 'val':
        data_ctdg = dataset.val_data
    elif args.split == 'test':
        data_ctdg = dataset.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    data_baseline_map = {
        "8days_dytag_large":[
            "/data/jiarui_ji/DGGen/results/synthetic_data/8days_dytag_large_uf_False_20250522_233700_b604.csv",
            "/data/jiarui_ji/tigger/models/8days_dytag_large/results/generated_edges.csv"
        ],
        "8days_dytag_small":[
            "/data/jiarui_ji/DGGen/results/synthetic_data/8days_dytag_small_uf_False_20250522_233641_186a.csv",
            "/data/jiarui_ji/tigger/models/8days_dytag_small/results/generated_edges.csv"
        ],
        "8days_dytag_small_text":[
            "/data/jiarui_ji/DGGen/results/synthetic_data/8days_dytag_small_textfno.csv",
            "/data/jiarui_ji/tigger/models/8days_dytag_small_text/results/generated_edges.csv"
        ],
         "8days_dytag_small_text_en":[
            "LLMGGen/baselines/DGGen/results/synthetic_data/8days_dytag_small_text_en.csv",
            "LLMGGen/baselines/tigger/models/8days_dytag_small_text_en/results/generated_edges.csv",
            "LLMGGen/baselines/DYMOND/8days_dytag_small_text_en/learned_parameters/generated_graph/results/generated_edges.csv"
        ],
        "weibo_daily":[
            "LLMGGen/baselines/idgg_csv_processed/llama3-8b/weibo_daily/edge_weibo_daily.csv",
            "LLMGGen/baselines/DGGen/results/synthetic_data/weibo_daily.csv",
            "LLMGGen/baselines/tigger/models/weibo_daily/results/generated_edges.csv",
            
        ],
        "imdb":[
            "LLMGGen/baselines/idgg_csv_processed/llama3-8b/imdb/edge_imdb.csv",
            "LLMGGen/baselines/DGGen/results/synthetic_data/imdb.csv",
            "LLMGGen/baselines/tigger/models/imdb/results/generated_edges.csv",
            
        ],
        "weibo_tech":[
            "LLMGGen/baselines/idgg_csv_processed/llama3-8b/weibo_tech/edge_weibo_tech.csv",
            "LLMGGen/baselines/DGGen/results/synthetic_data/weibo_tech.csv",
            "LLMGGen/baselines/tigger/models/weibo_tech/results/generated_edges.csv",
            
        ],
        "propagate_large_cn":[
            "LLMGGen/baselines/DGGen/results/synthetic_data/propagate_large_cn.csv",
            "LLMGGen/baselines/tigger/scripts/models/propagate_large_cn/results/generated_edges.csv",
            
        ]
    }
    input_edge_ids = torch.concat(list(torch.tensor(v) for v in data_ctdg.input_edges_dict.values()))
    indices = input_edge_ids.int()
    t = data_ctdg.ctdg.t[indices]
    initial_time = t[0].item()
    cut_time = t[-1].item()
    pred_len = data_ctdg.pred_len
    max_node_number = data_ctdg.node_text.shape[0]-1

    
    test_data = get_gt_data(data_ctdg, node_msg=args.node_msg, edge_msg=args.edge_msg)
    baseline_graphs = {}

    if args.cut_off_baseline == "edge":
        for baseline_path in data_baseline_map[args.data_name]:
            match = re.search(r'LLMGGen/baselines/([^/]+)/', baseline_path)
            baseline_name = match.group(1)  # 返回第一个捕获组（即斜杠之间的内
            try:
                baseline_data = get_snapshot_graph(baseline_path, 
                                                    initial_time = initial_time,
                                                    cut_time = None,
                                                pred_len=pred_len, 
                                                cut_edge_number=test_data.src.shape[0],
                                                edge_msg = args.edge_msg,
                                                time_window=data_ctdg.time_window)      
                if not args.edge_msg and not args.node_msg:
                    baseline_graphs[baseline_name] = [baseline_data]
                else:
                    # update msg
                    if not args.edge_msg and args.node_msg:
                        msg = torch.tensor(np.concatenate([data_ctdg.node_feature[baseline_data.src], 
                                                        data_ctdg.node_feature[baseline_data.dst]], axis=1), dtype=torch.float32)
                    elif args.edge_msg and not args.node_msg:
                        msg = baseline_data.msg
                    else:
                        msg = torch.tensor(np.concatenate([data_ctdg.node_feature[baseline_data.src], 
                                                        data_ctdg.node_feature[baseline_data.dst], 
                                                        baseline_data.msg], axis=1), dtype=torch.float32)
                    baseline_data.msg = msg
                    baseline_graphs[baseline_name] = [baseline_data]     
            except:
                pass

    elif args.cut_off_baseline == "time":
        for baseline_path in data_baseline_map[args.data_name]:
            match = re.search(r'LLMGGen/baselines/([^/]+)/', baseline_path)
            baseline_name = match.group(1)  # 返回第一个捕获组（即斜杠之间的内
            
            try:
                baseline_data = get_snapshot_graph(baseline_path, 
                                                    initial_time =  initial_time,
                                                    cut_time = cut_time,
                                                pred_len=pred_len, 
                                                cut_edge_number=None,
                                                edge_msg = args.edge_msg,
                                                time_window=data_ctdg.time_window) 
                if not args.edge_msg and not args.node_msg:
                    baseline_graphs[baseline_name] = [baseline_data]
                else:
                    if not args.edge_msg and args.node_msg:
                        msg = torch.tensor(np.concatenate([data_ctdg.node_feature[baseline_data.src], 
                                                        data_ctdg.node_feature[baseline_data.dst]], axis=1), dtype=torch.float32)
                    elif args.edge_msg and not args.node_msg:
                        msg = baseline_data.msg
                    else:
                        msg = torch.tensor(np.concatenate([data_ctdg.node_feature[baseline_data.src], 
                                                        data_ctdg.node_feature[baseline_data.dst], 
                                                        baseline_data.msg], axis=1), dtype=torch.float32)
                    baseline_data.msg = msg
                    baseline_graphs[baseline_name] = [baseline_data]
            except Exception as e:
                pass
    else:
        raise ValueError(f"Invalid cut_off_baseline: {args.cut_off_baseline}")
    

    
    
    
    
    return [test_data], baseline_graphs, max_node_number, data_ctdg.node_text, data_ctdg.node_feature, data_ctdg.pred_len, data_ctdg.unique_times