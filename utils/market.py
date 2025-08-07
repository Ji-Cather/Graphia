

import os.path as osp
from typing import Callable, Optional

import torch
import numpy as np

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
        
from torch import Tensor
import json
        
import pickle
import pandas as pd
from tqdm import tqdm

        

# load seq graphs
class MarketSeqDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        time_window = 24*60*60
    ) -> None:
        self.name = name.lower()
        self.time_window = time_window
        super().__init__(root, 
                         transform, 
                         pre_transform,
                         force_reload=force_reload)
        from torch_geometric.io import fs
        out = fs.torch_load(self.processed_paths[1])
        if len(out) == 2:  # Backward compatibility.
            data, _ = out
        else:
            data, _, data_cls = out

        self.node_feature = np.load(self.processed_paths[2])
        
        if not isinstance(data, dict):  # Backward compatibility.
            self.ctdg = data
        else:
            self.ctdg = TemporalData.from_dict(data)
        
        
        # 加载pickle文件中的NetworkX图列表
        try:
            with open(self.processed_paths[0], 'rb') as f:
                self.data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.PickleError) as e:
            print(f"无法加载NetworkX图列表: {e}")
            self.data = []
            
        
        self.slices = np.arange(len(self.data))
        self.mapping = torch.load(self.raw_paths[3], weights_only=False)
        self.cols = json.load(open(self.raw_paths[1]))
        self.semantic_ids = torch.load(self.raw_paths[2])
        
        
        

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return ['propagate_edge_feature.csv',
                'feature.json',
                "semids_n_head.pt",
                'mapping.pt',
                "member_node_feature.csv"
                ]

    @property
    def processed_file_names(self) -> str:
        return ['data.pkl', # dtdg + msg[edge feature]
                "data.pt", # ctdg + msg[edge feature]
                "node_feature.npy" # node feature
                ]
        

    
    def parse_message_semantic_ids(self, edge_df) -> Tensor:
        # get msg from raw feature
        semantic_ids = torch.load(self.raw_paths[2])
        item_id_raw = edge_df["id"].values
        
        # 为edge_df添加edge_id列
        sem_ids = semantic_ids[item_id_raw,:]
        # shape: edge*(n_sem+1)
        return sem_ids

    def process(self) -> None:
        import pandas as pd
        
        edge_df = pd.read_csv(self.raw_paths[0])
        
        # 将ds列转换为时间戳并按时间戳排序
        edge_df["ts"] = pd.to_datetime(edge_df["ds"]).astype(np.int64) // 10**9

        edge_df = edge_df.sort_values("ts")
        
        msgs = self.parse_message_semantic_ids(edge_df)

        
        # 按照src先dst后的顺序重新映射节点id
        mapping = torch.load(self.raw_paths[3], weights_only=False)
        
        # 使用mapping重新映射源节点和目标节点ID
        edge_df["src_member_id_new"] = edge_df["src_member_id_new"].map(mapping)
        edge_df["dst_member_id_new"] = edge_df["dst_member_id_new"].map(mapping)

        
        ctdg = TemporalData(src=Tensor(np.array(edge_df["src_member_id_new"])).to(torch.long), 
                            dst=Tensor(np.array(edge_df["dst_member_id_new"])).to(torch.long), 
                            msg = msgs, 
                            id = Tensor(np.array(edge_df["id"])).to(torch.long), # item ids
                            t=Tensor(np.array(edge_df["ts"]))
                            )
        if self.pre_transform is not None:
            ctdg = self.pre_transform(ctdg)


        self.ctdg = ctdg
        self.mapping = mapping
        self.data, self.node_feature = self.transfer_to_seqgraphs()
        self.semantic_ids = torch.load(self.raw_paths[2])
        
        
        # 保存snapshot_graphs列表
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(self.data, f)
        np.save(self.processed_paths[2], self.node_feature)
        self.save([ctdg], self.processed_paths[1])
        
        
    def transfer_to_seqgraphs(self):
        # 获取ctdg的t列
        t = self.ctdg.t
        # 获取ctdg的src列
        src = self.ctdg.src
        # 获取ctdg的dst列
        dst = self.ctdg.dst
        msgs = self.ctdg.msg
        
        if "id" in self.ctdg.keys():
            ids = self.ctdg.id
        else:
            ids = None
        
        # 获取最小和最大时间戳
        min_t = t.min().item()
        max_t = t.max().item()
        
        # 计算时间窗口数量
        num_windows = int((max_t - min_t) / self.time_window) + 1
        
        # 初始化存储每个时间窗口的图
        import networkx as nx
        snapshot_graphs = []
        
        # 获取所有节点（使用mapping的values）
        all_nodes = list(self.mapping.values())

        
        node_features = pd.read_csv(self.raw_paths[-1])
        node_features["member_id_new"] = node_features["member_id_new"].map(self.mapping)
        node_features.dropna(inplace=True)
        # 将feature_list按:分割并转换为数值矩阵
        node_features = np.array([
            [float(x) for x in feat.strip(":").split(":")]
            for feat in node_features["feature_list"]
        ])
        assert node_features.shape[0] == max(self.ctdg.src.max(), self.ctdg.dst.max()) +1
        
        # 为每个时间窗口创建snapshot graph
        for i in tqdm(range(num_windows), desc="Processing time windows"):
            # 计算当前时间窗口的起始和结束时间
            start_time = min_t + i * self.time_window
            end_time = start_time + self.time_window
            
            # 创建一个MultiDiGraph
            G = nx.MultiDiGraph()
            
            # 添加所有节点
            G.add_nodes_from(all_nodes)
            
            # 找出在当前时间窗口内的边的索引
            mask = (t >= start_time) & (t < end_time)
            window_indices = mask.nonzero(as_tuple=True)[0]
            
            # 获取当前时间窗口内的源节点和目标节点
            window_src = src[window_indices].cpu().numpy()
            window_dst = dst[window_indices].cpu().numpy()
            window_msgs = msgs[window_indices].cpu().numpy()
            if ids is not None:
                window_ids = ids[window_indices].cpu().numpy()
                # 添加边到图中
                for s, d, msg, id in zip(window_src, window_dst, window_msgs,window_ids):
                    G.add_edge(int(s), int(d), msg = msg, id = id, ts = t[window_indices[-1]])
            else:
                # 添加边到图中
                for s, d, msg in zip(window_src, window_dst, window_msgs):
                    G.add_edge(int(s), int(d), msg = msg, ts = t[window_indices[-1]])
            
            
            # 将图添加到列表中
            snapshot_graphs.append(G)
        
        
        return snapshot_graphs, node_features

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'


if __name__ == "__main__":
    
    dataset = MarketSeqDataset(root="/data/jiarui_ji/DGGen/data", 
                            name="8days")
    print(dataset.node_feature.shape)
