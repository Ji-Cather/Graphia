
from torch_geometric.data import TemporalData
import torch

import os
import pandas as pd
import numpy as np

from . import get_gt_data


def get_snapshot_graph(path, time_window:int = None, cut_edge_number:int = None):
    if (time_window is None and cut_edge_number is None) or (time_window is not None and cut_edge_number is not None):
        raise ValueError("time_window和cut_edge_number必须且只能设置其中一个")
    
    # 读取CSV文件
    df = pd.read_csv(path)
    
    # 确保时间戳按升序排序
    df = df.sort_values(by='t')
    
    if cut_edge_number:
        df_sub = df.iloc[:cut_edge_number]
    else:
        # 计算时间范围
        min_time = df['t'].min()
        max_time = df['t'].max()+ 1 if (df['t'].max() - df['t'].min())< time_window else df['t'].min() + time_window
        
        df_sub = df[(df['t'] >= min_time) & (df['t'] < max_time)]
    
    data = TemporalData(src=torch.tensor(df_sub['src'].values), 
                        dst=torch.tensor(df_sub['dst'].values), 
                        t=torch.tensor(df_sub['t'].values))
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
            "LLMGGen/baselines/tigger/models/8days_dytag_small_text_en/results/generated_edges.csv"
        ]
    }
    
    test_data = get_gt_data(data_ctdg, node_msg=args.node_msg, edge_msg=args.edge_msg)
    baseline_graphs = {}
    if args.cut_off_baseline == "edge":
        for baseline_name, baseline_path in zip([ "dggen","tigger"], data_baseline_map[args.data_name]):
            baseline_data = get_snapshot_graph(baseline_path, 
                                               time_window=None, 
                                               cut_edge_number=test_data.src.shape[0])      
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
            
    elif args.cut_off_baseline == "time_window":
        for baseline_name, baseline_path in zip([ "dggen","tigger"], data_baseline_map[args.data_name]):
            if baseline_name in ["dggen"]:
                time_window = data_ctdg.pred_len*data_ctdg.time_window
            elif baseline_name in ["tigger"]:
                time_window = data_ctdg.pred_len
                
            baseline_data = get_snapshot_graph(baseline_path, 
                                               time_window=time_window, 
                                               cut_edge_number=None)
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
    else:
        raise ValueError(f"Invalid cut_off_baseline: {args.cut_off_baseline}")
    

    
    max_node_number = data_ctdg.node_text.shape[0]-1
    
    
    return [test_data], baseline_graphs["tigger"], baseline_graphs["dggen"], max_node_number, data_ctdg.node_text, data_ctdg.node_feature