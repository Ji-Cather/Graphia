import torch
import os
import pandas as pd
from torch_geometric.data import TemporalData
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .utils.bwr_ctdg import (BWRCTDGALLDataset, 
                            BWRCTDGDataset, 
                            Dataset_Template)
from .eval_utils import get_gt_data
from .eval_utils.eval_src_edges import (get_ctdg_edges,
                                        split_temporal_graph_to_digraph_list,
                                        temporal_data_to_nx_graph,
                                        evaluate_edges,
                                        evaluate_graphs,
                                        evaluate_nodes,
                                        evaluate_graph_snapshots,
                                        evaluate_graph_macro_phenomena
                                        )


def eval_graph_structure(
                max_node_number,    
                gt_graph:TemporalData,
                gen_graph:TemporalData,
                node_text:np.ndarray=None,
                node_feature:np.ndarray=None):
    gen_matrix = get_ctdg_edges(gen_graph, max_node_number)
    gt_matrix = get_ctdg_edges(gt_graph, max_node_number)
    
    # node_matrixs = evaluate_nodes(gt_matrix,
    #                                 gen_matrix,
    #                                 node_text = node_text)
    
    graph_matrixs = evaluate_graphs(gt_matrix,
                                    gen_matrix,
                                    gt_graph,
                                    gen_graph,
                                    node_feature = node_feature)
                                    

    return graph_matrixs


def eval_graph_snapshot_structure(
                pred_times:list[int],  
                gt_graph:TemporalData,
                gen_graph:TemporalData):
    split_gt_graphs = split_temporal_graph_to_digraph_list(gt_graph, pred_times)
    split_gen_graphs = split_temporal_graph_to_digraph_list(gen_graph, pred_times)
    eval_matrixs = evaluate_graph_snapshots(split_gt_graphs, split_gen_graphs)
    return eval_matrixs
    
def generate_bert_embeddings(texts, desc="Processing texts"):
    """
    Generate BERT embeddings for a list of texts using TinyBERT
    
    Args:
        texts (list): List of text strings to embed
        desc (str): Description for progress bar
    
    Returns:
        np.ndarray: Array of embeddings
    """
    # Load TinyBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # Generate embeddings
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            # Handle potential NaN or empty texts
            if pd.isna(text) or text == "":
                text = " "
            inputs = tokenizer(str(text), return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            # Use [CLS] token output as text representation
            embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(embedding)
    
    return torch.cat(embeddings, dim=0).cpu().numpy()
    
def get_gen_data(df: pd.DataFrame,
                data_ctdg: BWRCTDGDataset,
                data_name: str,
                df_path: str,
                node_msg: bool = False,
                edge_msg: bool = False,
                ) -> TemporalData:
    # 初始化边列表
    edges = []
    edge_msg_path = os.path.join(os.path.dirname(df_path), 
                                 "edge_embeddings.npy")
    if edge_msg and not os.path.exists(edge_msg_path):
        output_dir = os.path.dirname(edge_msg_path)
        os.makedirs(output_dir, exist_ok=True)
        template = Dataset_Template[data_name]
    
        edge_text_template = template["edge_text_template"]
        edge_text_cols = template["edge_text_cols"]
        df['text'] = df.apply(
        lambda row: edge_text_template.format(**{col: row[col] if col in row else '' for col in edge_text_cols}), 
        axis=1
    )
    
        edge_embeddings = generate_bert_embeddings(df['text'].tolist(), desc="Processing edge texts")
        # df['msg'] = edge_embeddings
        np.save(edge_msg_path, edge_embeddings)
    elif edge_msg:
        edge_embeddings = np.load(edge_msg_path)
    for idx, row in df.iterrows():
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
            msg = torch.tensor(edge_embeddings[idx], dtype=torch.float32)
        else:
            msg = torch.tensor(np.concatenate([data_ctdg.node_feature[src_id], 
                                                data_ctdg.node_feature[dst_id],
                                                edge_embeddings[idx]
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
        msgs = torch.tensor(np.array([edge["msg"] for edge in edges]))
        return TemporalData(src=src, dst=dst, t=t, msg=msgs)
    
    

 


    
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
    
    
    # 假设not teacher forcing，这边要加入degree predictor结果的load    
    if args.split == 'train':
        data_ctdg = bwr_ctdg.train_data
    elif args.split == 'val':
        data_ctdg = bwr_ctdg.val_data
    elif args.split == 'test':
        data_ctdg = bwr_ctdg.test_data
    else:
        raise ValueError(f"Invalid split: {args.split}")
    
    if args.edge_report_path is not None:
        
        if os.path.exists(args.edge_report_path) and os.path.exists(args.edge_text_result_path):
            report_edge_df = pd.read_csv(args.edge_report_path)
            edge_text_df = pd.read_csv(args.edge_text_result_path)
            edge_matrix = evaluate_edges(None, edge_text_df)
            edge_matrix = pd.DataFrame([edge_matrix])
            report_edge_df = pd.concat([report_edge_df, edge_matrix], axis=1)
        else:
            if os.path.exists(args.edge_result_path):
                edge_df = pd.read_csv(args.edge_result_path)
            else: edge_df = None
            if os.path.exists(args.edge_text_result_path):
                edge_text_df = pd.read_csv(args.edge_text_result_path)
                edge_matrix = evaluate_edges(edge_df, edge_text_df)
            
            else:
                edge_matrix = evaluate_edges(edge_df, None)
            report_edge_df = pd.DataFrame([edge_matrix])
        os.makedirs(os.path.dirname(args.edge_report_path), exist_ok=True)
        report_edge_df.to_csv(args.edge_report_path)
    
    if args.graph_report_path is not None:
        df = pd.read_csv(args.graph_result_path)
        gt_graph = get_gt_data(data_ctdg, 
                            node_msg=args.node_msg,
                            edge_msg=args.edge_msg)
        gen_graph = get_gen_data(df,
                                data_ctdg,
                                args.data_name,
                                df_path=args.graph_result_path,
                                node_msg=args.node_msg,
                                edge_msg=args.edge_msg)
        
        eval_matrixs = eval_graph_structure(
                                data_ctdg.node_text.shape[0]-1,
                                gt_graph,
                                gen_graph,
                                node_text=data_ctdg.node_text,
                                node_feature=data_ctdg.node_feature)
        
        
        
        print(f"评估指标: {eval_matrixs}")
        eval_matrixs["experiment_name"] = args.graph_result_path.replace(".csv", "")
        report_df = pd.DataFrame([eval_matrixs])
        os.makedirs(os.path.dirname(args.graph_report_path), exist_ok=True)
        report_df.to_csv(args.graph_report_path)

    if args.graph_list_report_path is not None:

        df = pd.read_csv(args.graph_result_path)
        gt_graph = get_gt_data(data_ctdg, 
                            node_msg=args.node_msg,
                            edge_msg=args.edge_msg)
        gen_graph = get_gen_data(df,
                                data_ctdg,
                                args.data_name,
                                df_path=args.graph_result_path,
                                node_msg=args.node_msg,
                                edge_msg=args.edge_msg)
        gt_graph_nx = temporal_data_to_nx_graph(gt_graph)
        gen_graph_nx = temporal_data_to_nx_graph(gen_graph)
        eval_matrixs = evaluate_graph_snapshots([gt_graph_nx], 
                                                [gen_graph_nx])
        
        print(f"评估指标: {eval_matrixs}")
        eval_matrixs["experiment_name"] = args.graph_result_path.replace(".csv", "")
        report_df = pd.DataFrame([eval_matrixs])
       
        os.makedirs(os.path.dirname(args.graph_list_report_path), exist_ok=True)
        report_df.to_csv(args.graph_list_report_path)
   
    if args.graph_macro_report_path != "":
        df = pd.read_csv(args.graph_result_path)
        gt_graph = get_gt_data(data_ctdg, 
                            node_msg=args.node_msg,
                            edge_msg=args.edge_msg)
        gen_graph = get_gen_data(df,
                                data_ctdg,
                                args.data_name,
                                df_path=args.graph_result_path,
                                node_msg=args.node_msg,
                                edge_msg=args.edge_msg)
        
        eval_matrixs = evaluate_graph_macro_phenomena(
            pred_data=gen_graph,
            gt_data=gt_graph,
            max_node_number=data_ctdg.node_feature.shape[0],
            node_feature=data_ctdg.node_feature,
        )
        print(f"评估指标: {eval_matrixs}")
        eval_matrixs["experiment_name"] = args.graph_result_path.replace(".csv", "")
        report_df = pd.DataFrame([eval_matrixs])
        os.makedirs(os.path.dirname(args.graph_macro_report_path), exist_ok=True)
        report_df.to_csv(args.graph_macro_report_path)
        
        
    

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
    parser.add_argument('--graph_result_path', type=str, default=None, help='ggen result path(query ggen)')
    parser.add_argument('--graph_report_path', type=str, default=None, help='graph result report path')
    parser.add_argument('--graph_list_report_path', type=str, default=None, help='graph list_result report path')
    parser.add_argument('--graph_macro_report_path', type=str, default="", help='graph_macro_matrix.csv')
    
    # edge text eval args
    parser.add_argument('--edge_result_path', type=str, default=None, help='ggen result path(edge ggen)')
    parser.add_argument('--edge_text_result_path', type=str, default=None, help='edge text eval result path')
    parser.add_argument('--edge_report_path', type=str, default=None, help='edge text eval result report path')
    
   

    args = parser.parse_args()
    
    main(args)

 


    