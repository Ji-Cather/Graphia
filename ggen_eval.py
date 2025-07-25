import torch
import os
import pandas as pd
from torch_geometric.data import TemporalData



from .utils.bwr_ctdg import (BWRCTDGALLDataset, 
                            BWRCTDGDataset, 
                            Dataset_Template)
from .eval_utils import get_gt_data
from .eval_utils.eval_src_edges import evaluate_all_sources, get_ctdg_edges


def eval_graph(
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
        src_id = int(float(row["src"]))
        dst_id = int(float(row["dst"]))
        t = int(float(row["t"]))
        if not edge_msg and not node_msg:
            edge = {
            "src": src_id,
            "dst": int(dst_id),
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
            "src": src_id,
            "dst": dst_id,
            "t": t,
            "msg": msg
        }
        edges.append(edge)
    
    src = torch.tensor([edge["src"] for edge in edges], dtype=torch.int64)
    dst = torch.tensor([edge["dst"] for edge in edges], dtype=torch.int64)
    t = torch.tensor([edge["t"] for edge in edges], dtype=torch.int64)
    if not edge_msg and not node_msg:
        return TemporalData(src, dst, t)
    else:
        return TemporalData(src=src, dst=dst, t=t, msg=msg)
    

    
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
    gt_graph = get_gt_data(data_ctdg, 
                           node_msg=args.node_msg,
                           edge_msg=args.edge_msg)
    gen_graph = get_gen_data(df,
                             data_ctdg,
                             node_msg=args.node_msg,
                             edge_msg=args.edge_msg)
    
    eval_matrixs = eval_graph(data_ctdg.node_text.shape[0]-1,
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
    args = parser.parse_args()
    
    main(args)

 


    