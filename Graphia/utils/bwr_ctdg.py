
import numpy as np
from torch_geometric.data import InMemoryDataset
import torch
import os
import math
from torch_geometric.data import TemporalData
from typing import Optional, Callable
import pandas as pd

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from typing import Union

## Note: description should begin with "This network..."
Dataset_Template = {
    "weibo_tech": {
        "description": "This network represent social network among Weibo users focused on technology and digital products.",
        "edge_text_template": """
<ts_str>{ts_str}</ts_str>
<label>{label}</label>
<src_text>{src_text}</src_text>
<dst_text>{dst_text}</dst_text>""",
        "broadcast_message":"""
### Global Broadcast Message:
Dear Weibo Users:

Breaking News!
To celebrate the New Year, Weibo is launching an unprecedented super benefit campaign!

[Repost] this Weibo and @3 friends for a chance to win an iPhone 15 Pro Max!

Comment on this Weibo with your New Year's wish, and we'll randomly select 100 lucky users to help make your wish come true!
""",
        "goal":"Post as source user to elicit a detailed text response from the destination user; label interaction type (comment or repost).",
        "edge_text_cols": ["ts_str", "label", "src_text", "dst_text"],
        "edge_text_hint": {
            "label": "Main category of the product",
            "ts_str": "Interaction time",
            "src_text": "Src user text",
            "dst_text": "Dst user text"
        },
        "node_text_template": """
<user_name>{user_name}</user_name>
<user_source>{user_source}</user_source>
<user_gender>{user_gender}</user_gender>
<user_location>{user_location}</user_location>
<user_followers>{user_followers}</user_followers>
<user_friends>{user_friends}</user_friends>
<user_description>{user_description}</user_description>""",
        "node_text_hint": {
            "user_name": "User name",
            "user_source": "User source",
            "user_gender": "User gender",
            "user_location": "User location",
            "user_followers": "User followers",
            "user_friends": "User followees",
            "user_description": "User description",
        },
        "node_text_cols": ["user_name", "user_source", "user_gender", "user_location", "user_followers", "user_friends", "user_description"],
    },
    "8days_dytag_small_text_en": {
        "description": "This network represent market network of taoke members; nodes are members, edges are item propagations.",
        "goal":"Maximize the likelihood of item propagation from source member to destination member; label the main category of the item.",
        "edge_text_template": """
<ts_str>{ts_str}</ts_str>
<label>{label}</label>
<cate_level2_name>{cate_level2_name}</cate_level2_name>
<cate_level3_name>{cate_level3_name}</cate_level3_name>
<price_range>{price_range}</price_range>""",
        "edge_text_cols": ["label", "cate_level2_name", "cate_level3_name", "price_range", "ts_str"],
        "edge_text_hint": {
            "label": "Main category of the item",
            "ts_str": "Propagation time",
            "cate_level2_name": "Second-category of the item",
            "cate_level3_name": "Third-category of the item",
            "price_range": "Price range of the item",
        },
        "broadcast_message":"""
### Global Broadcast Message:

Pucker up for pure joy! Our irresistibly creamy, [Pure milk.] is now on a MASSIVE discount!
[Pure milk.] is on 10\% off!
But hurry! This zesty deal is melting away faster than our yogurt on your tongue.

""",
        "node_text_template": "<portrait_name>{portrait_name}</portrait_name>",
        "node_text_cols": ["portrait_name"],
        "node_text_hint": {
            "portrait_name": "Portrait name of the market member",
        },
    },
    
    "weibo_daily": {
        "description": "This network represent social interaction graph among Weibo users focused on technology and digital products.",
        "goal": "Post as source user to elicit a detailed text response from the destination user; label interaction type (comment or repost), and assign a timestamp no earlier than the node's prior collaboration year.",
        "edge_text_template": """
<ts_str>{ts_str}</ts_str>
<label>{label}</label>
<src_text>{src_text}</src_text>
<dst_text>{dst_text}</dst_text>""",
        "edge_text_cols": ["ts_str", "label", "src_text", "dst_text"],
        "edge_text_hint": {
            "label": "Main category of the product",
            "ts_str": "Interaction time",
            "src_text": "Src user text",
            "dst_text": "Dst user text"
        },
        "broadcast_message":"""
### Global Broadcast Message:
Breaking Update: Comment Challenge!
Attention Weibo users! 🚨

We've just unlocked a special comment-to-win event! For the next 24 hours, every comment you make could be your ticket to instant prizes! 🌟
Here's how it works:

[Comment] on any post using #CommentChallenge
Each comment gives you an entry
Witty, creative, or engaging comments get bonus entries!""",
        "node_text_template": """
<user_name>{user_name}</user_name>
<user_source>{user_source}</user_source>
<user_gender>{user_gender}</user_gender>
<user_location>{user_location}</user_location>
<user_followers>{user_followers}</user_followers>
<user_friends>{user_friends}</user_friends>
<user_description>{user_description}</user_description>""",
        "node_text_hint": {
            "user_name": "User name",
            "user_source": "User source",
            "user_gender": "User gender", 
            "user_location": "User location",
            "user_followers": "User followers",
            "user_friends": "User followees",
            "user_description": "User description",
        },
        "node_text_cols": ["user_name", "user_source", "user_gender", "user_location", "user_followers", "user_friends", "user_description"],
    },
   "weibo_daily_long": {
        "description": "This network represent social interaction graph among Weibo users focused on technology and digital products.",
        "goal": "Post as source user to elicit a detailed text response from the destination user; label interaction type (comment or repost), and assign a timestamp no earlier than the node's prior collaboration year.",
        "edge_text_template": """
<ts_str>{ts_str}</ts_str>
<label>{label}</label>
<src_text>{src_text}</src_text>
<dst_text>{dst_text}</dst_text>""",
        "edge_text_cols": ["ts_str", "label", "src_text", "dst_text"],
        "edge_text_hint": {
            "label": "Main category of the product",
            "ts_str": "Interaction time",
            "src_text": "Src user text",
            "dst_text": "Dst user text"
        },
        "broadcast_message":"""
### Global Broadcast Message:
Breaking Update: Comment Challenge!
Attention Weibo users! 🚨

We've just unlocked a special comment-to-win event! For the next 24 hours, every comment you make could be your ticket to instant prizes! 🌟
Here's how it works:

[Comment] on any post using #CommentChallenge
Each comment gives you an entry
Witty, creative, or engaging comments get bonus entries!""",
        "node_text_template": """
<user_name>{user_name}</user_name>
<user_source>{user_source}</user_source>
<user_gender>{user_gender}</user_gender>
<user_location>{user_location}</user_location>
<user_followers>{user_followers}</user_followers>
<user_friends>{user_friends}</user_friends>
<user_description>{user_description}</user_description>""",
        "node_text_hint": {
            "user_name": "User name",
            "user_source": "User source",
            "user_gender": "User gender", 
            "user_location": "User location",
            "user_followers": "User followers",
            "user_friends": "User followees",
            "user_description": "User description",
        },
        "node_text_cols": ["user_name", "user_source", "user_gender", "user_location", "user_followers", "user_friends", "user_description"],
    },
}



def custom_collate(batch):
    # batch 是一个列表，每个元素是单个样本的字段字典
    # 提取所有字段并堆叠为批次
    keys = batch[0].keys()  # 获取所有字段的键
    # 按key提取并堆叠
    batch_data = {}
    for key in keys:
        if key in ['input_edge_ids', 'output_edge_ids']:
            # 对于边的索引，使用列表的形式
            batch_data[key] = [item[key] for item in batch]
        else:
            batch_data[key] = np.stack([item[key] for item in batch])
   
    return batch_data





# 分别save train_data_ctdg, val_data_ctdg, test_data_ctdg
class BWRCTDGALLDataset(InMemoryDataset):
    
    def __init__(self, 
                pred_ratio:float = 0.1,
                bwr: int = 0, # if 0, calculate bwr
                time_window: int = 24*60*60, # 24 hours
                use_feature:str = 'no',
                cm_order:bool = False,# use cm_order to order nodes
                root: Optional[str] = None, # save root for node seq data
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                log: bool = True,
                force_reload: bool = False,
                 ):
        """
        初始化DTCG数据结构，从快照图中生成邻接表
        :param snapshots: list[NetworkX图对象]
        :param max_src_node_id: int, 最大源节点ID
        :param seq_len: int, 序列长度
        :param k_quantiles: int, quantile的数量
        """
        self.pred_ratio = pred_ratio
        self.bwr = bwr
        self.time_window = time_window
        self.use_feature = use_feature 
        self.cm_order = cm_order
        self.data_name = os.path.basename(root)
        
        super().__init__(root, 
                         transform, 
                         pre_transform,
                         force_reload=force_reload,
                         )
        # unstable; to be modified
        import sys
        setattr(sys.modules["__main__"], "BWRCTDGDataset", BWRCTDGDataset)
        self.data = torch.load(os.path.join(self.processed_dir, 
                                                  "bwr_ctdg_{mode}.pt".format(mode="all")),
                                                map_location="cpu")
        self.train_data = torch.load(os.path.join(self.processed_dir, 
                                                  "bwr_ctdg_{mode}.pt".format(mode="train")),
                                                map_location="cpu")
        self.val_data = torch.load(os.path.join(self.processed_dir, 
                                                "bwr_ctdg_{mode}.pt".format(mode="val")),
                                                map_location="cpu")
        self.test_data = torch.load(os.path.join(self.processed_dir, 
                                                 "bwr_ctdg_{mode}.pt".format(mode="test")),
                                                map_location="cpu")
        
        self.src_min = self.data.ctdg.src.min()
        self.src_max = self.data.ctdg.src.max()
        self.dst_min = self.data.ctdg.dst.min()
        self.dst_max = self.data.ctdg.dst.max()
        
            
    @property
    def raw_file_names(self) -> str:
        return [
            'node_feature.csv',
            'edge_feature.csv',
        ]

    @property
    def processed_file_names(self) -> str:
        return [
                'bwr_ctdg_all.pt',
                "bwr_ctdg_train.pt",
                'bwr_ctdg_val.pt',
                'bwr_ctdg_test.pt',
                ]
    
    
        
    def cm_order_nodes(self, 
                       edge_feature_df: pd.DataFrame,
                       node_feature_df: pd.DataFrame,
                       test_edge_df_input: pd.DataFrame,
                       max_node_number: int,
                       plot_path: Optional[str] = None,
                       calculate_bwr: bool = True):
        """
        使用Reverse Cuthill-McKee算法对CTDG中的节点进行重排序
        Args:
           edge_feature_df: 原始edge feature
           node_feature_df: 原始node feature
        Returns:
            edge_feature_df: 重排序后的edge feature
            node_feature_df: 重排序后的node feature
        """
        # 构建邻接矩阵
        adj_matrix = torch.zeros((max_node_number + 1, max_node_number + 1), dtype=torch.float)
        
        # 填充邻接矩阵
        for i in range(len(test_edge_df_input)):
            src, dst = test_edge_df_input['src'].iloc[i], test_edge_df_input['dst'].iloc[i]
            adj_matrix[src, dst] += 1 
            adj_matrix[dst, src] += 1
            
        # 使用scipy的reverse_cuthill_mckee算法
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        from scipy.sparse import csr_matrix
        
        # 转换为scipy稀疏矩阵
        adj_sparse = csr_matrix(adj_matrix.numpy())
        
        # 应用RCM算法获取节点顺序
        node_order = reverse_cuthill_mckee(adj_sparse)
        # 删除node_order中的0
        node_order = node_order[node_order != 0]
        # 创建节点id到新顺序的映射
        # 加1是因为原始节点id从1开始
        node_order_mapping = {old: new + 1 for new, old in enumerate(node_order)} 
        
        # 对node_feature_df的id进行mapping
        node_feature_df['node_id'] = node_feature_df['node_id'].map(node_order_mapping)
        assert node_feature_df['node_id'].max() == node_feature_df.shape[0], f"invalid node_id"
        
        # 对edge_feature_df的src和dst进行mapping
        edge_feature_df['src'] = edge_feature_df['src'].map(node_order_mapping).astype(int)
        edge_feature_df['dst'] = edge_feature_df['dst'].map(node_order_mapping).astype(int)
        
        # 对node_feature_df按照id进行排序
        node_feature_df = node_feature_df.sort_values('node_id',axis=0)
        node_feature_df["node_id"] = node_feature_df["node_id"].astype(int)
        # 计算BWR
        if calculate_bwr:
            # 获取所有边的坐标
            edge_coords = edge_feature_df[['src', 'dst']].values
            
            # 计算每个边到对角线的距离
            distances = []
            for src, dst in edge_coords:
                # 计算到对角线的x和y距离
                x_dist = abs(src - dst)
                y_dist = abs(src - dst)
                # 取最大值作为该边的距离
                max_dist = max(x_dist, y_dist)
                if max_dist > 0:
                    distances.append((max_dist, src, dst))
            
            # 按距离排序并获取topk
            distances.sort(reverse=True)
            # max
            max_bwr = int(np.max([distance[0] for distance in distances]))
            # mean
            mean_bwr = int(np.mean([distance[0] for distance in distances]))
            # 90%
            p90_bwr = int(np.percentile([distance[0] for distance in distances], 90))
            
            
            print(f"Calculated BWR: {max_bwr}, {mean_bwr}, {p90_bwr}")
            self.bwr = p90_bwr
            
            
        
        if plot_path is not None:
            import matplotlib.pyplot as plt
            
            # 绘制原始邻接矩阵
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.spy(adj_matrix)
            plt.title('Original Adjacency Matrix')
            
            # 绘制重排序后的邻接矩阵
            plt.subplot(122)
            reordered_adj = torch.zeros_like(adj_matrix)
            for i in range(len(edge_feature_df)):
                src, dst = edge_feature_df['src'].iloc[i], edge_feature_df['dst'].iloc[i]
                reordered_adj[src -1, dst -1] += 1
                reordered_adj[dst -1, src -1] += 1
            plt.spy(reordered_adj)
            
            # 添加BWR区域的阴影带
            # 在邻接矩阵上标记BWR区域
            bwr_mask = torch.zeros_like(adj_matrix)
            # 直接在对角线附近标记BWR区域
            for i in range(self.bwr):
                # 获取对角线偏移后的坐标
                diag_coords = torch.arange(max_node_number - i)
                src_coords = diag_coords
                dst_coords = diag_coords + i
                # 标记上三角区域，确保索引不越界
                valid_indices = (dst_coords < max_node_number)
                bwr_mask[src_coords[valid_indices], dst_coords[valid_indices]] = 1
                # 标记下三角区域
                bwr_mask[dst_coords[valid_indices], src_coords[valid_indices]] = 1
            # 绘制BWR区域
            plt.imshow(bwr_mask, cmap='Reds', alpha=0.3)
            
            plt.title(f'Reordered Adjacency Matrix with BWR—{self.bwr}')
            
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        return node_order_mapping, node_feature_df, edge_feature_df
        
            
    def aggregate_df_text(self, df: pd.DataFrame, text_template: str, text_cols: list):
        df['text'] = df.apply(lambda row: text_template.format(**row[text_cols].to_dict()), axis=1)
        return df
        
    def process(self):
        # load node_feature, edge_feature
        node_feature_df = pd.read_csv(os.path.join(self.root,'raw', self.raw_file_names[0]))
        edge_feature_df = pd.read_csv(os.path.join(self.root,'raw', self.raw_file_names[1]))
        edge_feature_df.sort_values('ts', inplace=True)
       
        unique_times = np.unique(edge_feature_df['ts'].values//self.time_window) * self.time_window
        sum_len = len(unique_times)
        pred_len = int(sum_len * self.pred_ratio)
        input_len = sum_len - 3*pred_len
        assert input_len > 0, f"invalid pred ratio {self.pred_ratio}, must be smaller than {1/3}"
        assert input_len + 3*pred_len == sum_len, f"invalid pred ratio {self.pred_ratio}"
        train_cut_id =  input_len + pred_len
        val_start_id = pred_len
        val_cut_id = 2*pred_len +input_len
        test_start_id = 2*pred_len
        
        test_edge_df_input = edge_feature_df[edge_feature_df["ts"] < unique_times[-pred_len]]
        
        if self.cm_order:
            os.makedirs(os.path.join(self.root,'plot'), exist_ok=True)
            node_map, node_feature_df, edge_feature_df = self.cm_order_nodes(edge_feature_df, 
                                                                   node_feature_df, 
                                                                   test_edge_df_input,
                                                                   node_feature_df.shape[0],
                                    # plot_path=os.path.join(self.root,'plot', 'cm_order_nodes.png'),
                                    calculate_bwr = self.bwr==0)
       
        
        edge_feature_df['edge_id'] = np.arange(edge_feature_df.shape[0])
        ## market network
        node_feature_df = self.aggregate_df_text(node_feature_df, 
                                                 text_template=Dataset_Template[self.data_name]["node_text_template"], 
                                                 text_cols=Dataset_Template[self.data_name]["node_text_cols"])
        # 在node_feature_df的开头添加一行
        new_row = pd.DataFrame({'node_id': [0], 'text': [""]})
        node_feature_df = pd.concat([new_row, node_feature_df], ignore_index=True)
        node_feature_df.sort_values('node_id', inplace=True)
        assert node_feature_df['node_id'].max() == node_feature_df.shape[0] - 1, f"invalid node_id"
        edge_feature_df["ts_str"] = edge_feature_df["ts"].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        edge_feature_df = self.aggregate_df_text(edge_feature_df, 
                                                 text_template=Dataset_Template[self.data_name]["edge_text_template"], 
                                                 text_cols=Dataset_Template[self.data_name]["edge_text_cols"])
        # Map unique values in 'labels' to int
        label2int = {label: idx for idx, label in enumerate(edge_feature_df['label'].unique())}
        edge_feature_df['label_int'] = edge_feature_df['label'].map(label2int)
        label_texts = list(label2int.keys())
        ## convert text to bert embedding
        embed_feature_dim = 128
        if self.use_feature == 'no':
            edge_feature = np.zeros((edge_feature_df.shape[0], embed_feature_dim))
            node_feature = np.zeros((node_feature_df.shape[0], embed_feature_dim))
            label_feature = np.zeros((len(label_texts),embed_feature_dim))
        elif self.use_feature == 'bert':
            ## convert text to bert embedding
            from transformers import AutoTokenizer, AutoModel
            import torch.nn.functional as F

            # 加载TinyBERT模型和分词器
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            model.eval()

            # 处理节点文本特征
            node_texts = node_feature_df['text'].tolist()
            node_embeddings = []
            
            with torch.no_grad():
                for text in tqdm(node_texts, desc="Processing node texts"):
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    outputs = model(**inputs)
                    # 使用[CLS]标记的输出作为文本表示
                    embedding = outputs.last_hidden_state[:, 0, :]
                    node_embeddings.append(embedding)
            
            node_feature = torch.cat(node_embeddings, dim=0).cpu().numpy()

            # 处理边文本特征
            edge_texts = edge_feature_df['text'].tolist()
            edge_embeddings = []
            
            with torch.no_grad():
                for text in tqdm(edge_texts, desc="Processing edge texts"):
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :]
                    edge_embeddings.append(embedding)
            
            edge_feature = torch.cat(edge_embeddings, dim=0).cpu().numpy()


            
            label_embeddings = []
            with torch.no_grad():
                for text in tqdm(label_texts, desc="Processing label texts"):
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :]
                    label_embeddings.append(embedding)
            
            label_feature = torch.cat(label_embeddings, dim=0).cpu().numpy()

        else:
            raise ValueError(f"Invalid use_feature: {self.use_feature}")
        
        
        
        train_edge_id = edge_feature_df[edge_feature_df["ts"]<=unique_times[train_cut_id]]['edge_id']
        val_edge_id = edge_feature_df[(edge_feature_df["ts"] >= unique_times[val_start_id]) & 
                              (edge_feature_df["ts"] <= unique_times[val_cut_id])]['edge_id']
        test_edge_id = edge_feature_df[(edge_feature_df["ts"] >= unique_times[test_start_id]) ]['edge_id']
        
        edge_feature_df = edge_feature_df.set_index('edge_id')
        
        for edge_id, mode, unique_times_split in zip([train_edge_id, val_edge_id, test_edge_id, edge_feature_df.index], \
            ["train", "val", "test", "all"], [unique_times[:train_cut_id], 
                                              unique_times[val_start_id:val_cut_id], 
                                              unique_times[test_start_id:], unique_times]):
            # Sort edge_id to ensure temporal order
            edge_id = edge_id.sort_values()
            ctdg = TemporalData(
                src=torch.tensor(edge_feature_df['src'][edge_id].values, dtype=torch.long),
                dst=torch.tensor(edge_feature_df['dst'][edge_id].values, dtype=torch.long),
                t=torch.tensor(edge_feature_df['ts'][edge_id].values, dtype=torch.long),
                label=torch.tensor(edge_feature_df['label_int'][edge_id].values, dtype=torch.long),
                edge_id_all=torch.tensor(edge_id.values, dtype=torch.long),
                edge_id=torch.arange(len(edge_id)) # 相对的位置
            )
            assert label_texts[ctdg.label[0]] in edge_feature_df.iloc[edge_id]['text'].values[0],\
            "incorrect edge order!!"
            if mode == "train":
                exist_node = np.unique(np.concatenate([ctdg.src, ctdg.dst, np.array([0])]))
            
            edge_has_new = ~(np.isin(ctdg.src, exist_node) & np.isin(ctdg.dst, exist_node))
            node_has_new = ~ np.isin(np.arange(node_feature_df.shape[0]), exist_node)
            
            ctdg.new_node = torch.tensor(edge_has_new, dtype=torch.bool)
            
            bwr_ctdg = BWRCTDGDataset(
                bwr=self.bwr,
                time_window=self.time_window,
                unique_times=unique_times_split,
                ctdg=ctdg,
                new_node_mask=node_has_new,
                node_feature=node_feature,
                edge_feature=edge_feature[edge_id.values],
                label_feature=label_feature,
                node_text=node_feature_df['text'].values,
                edge_text=edge_feature_df.iloc[edge_id]['text'].values,
                label_text = label_texts,
                input_len = input_len,
                pred_len = pred_len
            )
            
            # # test
            # from torch.utils.data import DataLoader
            # data_loader = DataLoader(bwr_ctdg, batch_size=5, collate_fn=custom_collate)
            # for batch in data_loader:
            #     print(batch)
            
            torch.save(bwr_ctdg, os.path.join(self.processed_dir, 
                                              "bwr_ctdg_{mode}.pt".format(mode=mode)))
            
            
from collections import defaultdict


class BWRCTDGDataset:
    # dataset len： N//bwr 
    # each sample: (bwr, outdegree)
    def __init__(self, 
                 time_window: int, # pred window
                 unique_times: np.ndarray, # [T] the time of each window
                 bwr: int,
                 ctdg: TemporalData, # M, 
                 input_len: int,
                 pred_len: int,
                 new_node_mask: np.ndarray = None, # [N+1]
                 node_feature: np.ndarray = None, # [N+1, 0 for null,node_feature_dim]
                 edge_feature: np.ndarray = None, # [M, edge_feature_dim]
                 label_feature: np.ndarray = None, # [cat_num, label_feature_dim]
                 node_text: np.ndarray = None, # [N+1, 0 for null, "str"]
                 edge_text: np.ndarray = None, # [M, "str"]
                 label_text: np.ndarray = None, # [cat_num,]
                 ):
        self.time_window = time_window
        self.bwr = bwr
        self.len = math.ceil(int(ctdg.src.max()) / self.bwr)
        self.new_node_mask = new_node_mask
        self.unique_times = unique_times
        self.ctdg = ctdg
        self.input_len = input_len
        self.pred_len = pred_len
        
        graph_times = []
       
        ctdg_src_node_list = []
        # 1742601600
        ctdg_src_unique_dst_list = []
        
                
        # 按cut off time分类input和output edges
        input_edges_dict = defaultdict(list)
        output_edges_dict = defaultdict(list)
        
        # 对每个时间窗口进行统计
        for t in range(len(unique_times)):
            # 获取当前时间窗口的边索引
            if t == len(unique_times) - 1:
                mask = (self.ctdg.t >= unique_times[t]) 
            elif t == 0:
                mask = (self.ctdg.t < unique_times[t+1])
            else:
                mask = (self.ctdg.t >= unique_times[t]) & (self.ctdg.t < unique_times[t+1])
                
            if t >= self.input_len:
                for src, edge_id in zip(self.ctdg.src[mask], self.ctdg.edge_id[mask]):
                    output_edges_dict[src.item()].append(edge_id.item())
            else:
                for src, edge_id in zip(self.ctdg.src[mask], self.ctdg.edge_id[mask]):
                    input_edges_dict[src.item()].append(edge_id.item())
            
            graph_times.append(self.ctdg.t[mask][-1])
            # 统计当前时间窗口内每个源节点的度
            unique_src, src_counts = np.unique(self.ctdg.src[mask], return_counts=True)
            src_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
            # 将结果添加到列表中
            ctdg_src_node_list.append(src_counts)
            
            edge_list = list(zip(self.ctdg.src[mask], self.ctdg.dst[mask]))
            unique_edge = list(set(edge_list))
            unique_edge_src = [edge[0] for edge in unique_edge]
            unique_src, src_counts = np.unique(unique_edge_src, return_counts=True)
            src_unique_counts = dict(zip(unique_src.tolist(), src_counts.tolist()))
            ctdg_src_unique_dst_list.append(src_unique_counts)
        
        self.input_edges_dict = input_edges_dict
        self.output_edges_dict = output_edges_dict
        
        self.ctdg_src_node_degree = np.array([[src_degree_list.get(int(src_id), 0) 
                                               for src_degree_list in ctdg_src_node_list] 
                           for src_id in np.arange(ctdg.src.max()+1)])
        self.ctdg_src_unique_degree= np.array([[src_unique_degree_list.get(int(src_id), 0) 
                                               for src_unique_degree_list in ctdg_src_unique_dst_list] 
                           for src_id in np.arange(ctdg.src.max()+1)]) # [N, N]
        
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.label_feature = label_feature
        self.node_text = node_text
        self.edge_text = edge_text
        self.label_text = label_text
        interaction_cache = defaultdict(self.create_node_cache)
        self.interaction_cache = self.compute_node_structure_metrics(interaction_cache)
    

        
    def create_node_cache(self):
        return {
            'frequency': 0,
            'count': defaultdict(int),
            'interactions': [],
            'neighbors': set(),
            'times_as_source': 0,
            'times_as_destination': 0,
            'historical_interaction': [],
            'common_neighbors': [],
            'neighbor_average_frequency': [],
        }
        
    def __getitem__(self, idx):
        # src_infos[T, bwr], src_edge_infos [T, bwr, bwr]
        if idx == self.len-1:
            bwr_src_ids = np.arange(self.ctdg_src_node_degree.shape[0]-self.bwr, self.ctdg_src_node_degree.shape[0])
        else:
            bwr_src_ids = np.arange(idx * self.bwr, (idx+1) * self.bwr) # bwr
            
        
        src_node_feature, src_interact_times, src_pred_time, src_node_degree, src_node_unique_degree = self.src_infos(bwr_src_ids)
        input_edge_ids, output_edge_ids = self.src_edge_infos(bwr_src_ids)
        
        if hasattr(self, 'pred_src_dx'):
            src_model_pred_degree = self.pred_src_dx[bwr_src_ids]
            return {
                'src_node_feature': src_node_feature,
                'src_interact_times': src_interact_times,
                'src_pred_time': src_pred_time,
                'src_node_degree': src_node_degree,
                'src_node_unique_degree': src_node_unique_degree,
                'input_edge_ids': input_edge_ids,
                'output_edge_ids': output_edge_ids,
                'src_node_ids': bwr_src_ids,
                'src_model_pred_degree': src_model_pred_degree,
            }
        else:
            return {
                'src_node_feature': src_node_feature,
                'src_interact_times': src_interact_times,
                'src_pred_time': src_pred_time,
                'src_node_degree': src_node_degree,
                'src_node_unique_degree': src_node_unique_degree,
                'input_edge_ids': input_edge_ids,
                'output_edge_ids': output_edge_ids,
                'src_node_ids': bwr_src_ids,
            }
        
    def load_degree_predictor_results(self,
                                      degree_result_path:str):
        degree_result = torch.load(degree_result_path)
        self.pred_src_dx = degree_result['degree'].round().int().numpy()
        # self.pred_src_unique_dx = degree_result['unique_degree'].round().int().numpy()
    
    
    
    def src_infos(self, src_node_ids): # src_node_ids: [bwr]
        #   src_node_feature, src_node_interact_times, src_node_pred_feature, src_node_pred_interact_times, src_node_pred_degree 

        src_node_ids_flat = src_node_ids
        src_node_interact_times = np.repeat(self.unique_times.reshape(1, -1), 
                                            len(src_node_ids_flat), axis=0)
        
        src_node_degree = self.ctdg_src_node_degree[src_node_ids_flat]
        src_node_unique_degree = self.ctdg_src_unique_degree[src_node_ids_flat]
        src_node_feature = self.node_feature[src_node_ids_flat]
        src_node_degree_log = np.log1p(src_node_degree + 1e-10)
        src_node_unique_degree_log = np.log1p(src_node_unique_degree + 1e-10)
        src_node_agg_feature = np.concatenate([src_node_degree_log[:,:self.input_len], 
                                               src_node_unique_degree_log[:,:self.input_len], 
                                               src_node_feature], axis=1)
        
        return src_node_agg_feature, src_node_interact_times[:,:self.input_len], src_node_interact_times[:,self.input_len:], \
            src_node_degree[:,self.input_len:], src_node_unique_degree[:,self.input_len:]
    
    def src_edge_infos(self, src_node_ids):
        # bwr, edges  [edge_number == degree(src_node_id)]
        src_input_edges = []
        src_output_edges = []
        for src_id in src_node_ids:
            # 获取input edges
            if src_id in self.input_edges_dict:
                src_input_edges.append(torch.tensor(self.input_edges_dict[src_id]))
            else:
                src_input_edges.append(torch.tensor([]))
                
            # 获取output edges
            if src_id in self.output_edges_dict:
                src_output_edges.append(torch.tensor(self.output_edges_dict[src_id]))
            else:
                src_output_edges.append(torch.tensor([]))
                
        
        return src_input_edges, src_output_edges
    
    
    def compute_node_structure_metrics(self,
                                       interaction_cache:dict = {}):
        
        for src_id, edge_ids in tqdm(self.input_edges_dict.items(), 
                                     desc="Computing node structure metrics"):
            for edge_id in edge_ids:
                dst_id = self.ctdg.dst[edge_id].item()
                timestamp = self.ctdg.t[edge_id].item()
                interaction_cache[src_id]['frequency'] += 1
                interaction_cache[src_id]['times_as_source'] += 1
                interaction_cache[src_id]['count'][dst_id] += 1
                interaction_cache[src_id]['interactions'].append({
                    'timestamp': timestamp,
                    'interacting_node_id': dst_id,
                })
                interaction_cache[src_id]['neighbors'].add(dst_id)

                interaction_cache[dst_id]['frequency'] += 1
                interaction_cache[dst_id]['times_as_destination'] += 1
                interaction_cache[dst_id]['count'][src_id] += 1
                interaction_cache[dst_id]['interactions'].append({
                    'timestamp': timestamp,
                    'interacting_node_id': src_id,
                })
                interaction_cache[dst_id]['neighbors'].add(src_id)
                
                interaction_count = self.calculate_interaction_count(src_id, dst_id, interaction_cache)
                common_neighbor_count = self.calculate_common_neighbor_count(src_id, dst_id, interaction_cache)
                neighbor_avg_freq_src = self.get_neighbor_average_frequency(src_id, interaction_cache)
                neighbor_avg_freq_dst = self.get_neighbor_average_frequency(dst_id, interaction_cache)
                
                interaction_cache[src_id]['historical_interaction'].append(interaction_count)
                interaction_cache[dst_id]['historical_interaction'].append(interaction_count)
                interaction_cache[src_id]['common_neighbors'].append(common_neighbor_count)
                interaction_cache[dst_id]['common_neighbors'].append(common_neighbor_count)
                interaction_cache[src_id]['neighbor_average_frequency'].append(neighbor_avg_freq_src)
                interaction_cache[dst_id]['neighbor_average_frequency'].append(neighbor_avg_freq_dst)
        return interaction_cache
        
    
    def __len__(self):
        return self.len
    
    
    def calculate_common_neighbor_count(self, node_id, interacting_node_id, interaction_cache):
        """
        Calculate the number of common neighbors between node_id and interacting_node_id.
        """
        node_neighbors = interaction_cache[node_id]['neighbors']
        interacting_node_neighbors = interaction_cache[interacting_node_id]['neighbors']
        common_neighbors = node_neighbors & interacting_node_neighbors
        return len(common_neighbors)


    def calculate_interaction_count(self, node_id, other_node_id, interaction_cache):
        """
        Calculate the number of interactions between node_id and other_node_id.
        """
        return interaction_cache[node_id]['count'].get(other_node_id, 0)


    def get_neighbor_average_frequency(self, node_id, interaction_cache):
        """
        Calculate the average interaction frequency of a node's neighbors.
        """
        neighbors = interaction_cache[node_id]['neighbors']
        if not neighbors:
            return 0.0
        total_freq = sum(interaction_cache[neighbor]['frequency'] for neighbor in neighbors)
        return total_freq / len(neighbors)
    
    
    def get_dst_nodes_texts(self, 
                       src_id, 
                       dst_ids, 
                       interaction_cache,
                       good_metric:list = ['text',
                                           'HI',
                                           'CN',
                                           'NSM']) -> list:
        agent_text_parts = []
        
        for dst_id in dst_ids:
            interaction_count = self.calculate_interaction_count(src_id, dst_id, interaction_cache)
            common_neighbor_count = self.calculate_common_neighbor_count(src_id, dst_id, interaction_cache)
             # Destination Node Metrics
            dest_metrics = {
                'SF': interaction_cache[dst_id]['frequency'],
                'AFN': (
                    int(np.mean(list(interaction_cache[dst_id]['count'].values())))
                    if interaction_cache[dst_id]['count'] else 0
                )
            }
            agent_text_parts.append(f"[For dst-node ({dst_id}):]")
            if 'text' in good_metric:
                agent_text_parts.append(f"- Text: {self.node_text[dst_id]}")

            if 'HI' in good_metric:
                agent_text_parts.append(f"- HI: {interaction_count}")

            if 'CN' in good_metric:
                agent_text_parts.append(f"- CN count between {src_id} and {dst_id}: {common_neighbor_count}")

            if 'NSM' in good_metric:
                agent_text_parts.append(f"- SF: {dest_metrics['SF']}")
                agent_text_parts.append(f"- AFN: {dest_metrics['AFN']}")
            agent_text_parts.append("")
        
        return "\n".join(agent_text_parts)
    
    def get_src_node_texts(self, 
                           src_id, 
                           interaction_cache:dict = {},
                           good_metric:list = ['text',
                                           'NSM']):
        agent_text_parts = []
        if len(good_metric) > 1:
            source_metrics = {
                            'SF': interaction_cache[src_id]['frequency'],
                            'AFN': (
                                int(np.mean(list(interaction_cache[src_id]['count'].values())))
                                if interaction_cache[src_id]['count'] else 0
                            )
                        }
        if 'text' in good_metric:
            agent_text_parts.append(f"- Text: {self.node_text[src_id]}")


        if 'NSM' in good_metric:
            agent_text_parts.append(f"- SF: {source_metrics['SF']}")
            agent_text_parts.append(f"- AFN: {source_metrics['AFN']}")
            
        agent_text_parts.append("")

        agent_text = "\n".join(agent_text_parts)
        return agent_text

    def get_dst_ids(self, input_edge_ids):
        history_dst_ids = []
        if input_edge_ids.shape[0] > 0:
            history_dst_ids = self.ctdg.dst[input_edge_ids]
        else:
            history_dst_ids = []
        return history_dst_ids

    def get_history_dst_edge_texts(self, 
                                    input_edge_ids,
                                    dst_id: None):
        edge_dst_ids = self.get_dst_ids(input_edge_ids)
        # Rank input_edge_ids: put edges where edge_dst_ids == dst_id at the front
        if input_edge_ids.shape[0] > 0:
            edge_dst_ids = self.ctdg.dst[input_edge_ids]
            # Find indices where dst matches
            match_mask = (edge_dst_ids == dst_id)
            # Indices for matching and non-matching
            match_indices = np.where(match_mask)[0]
            non_match_indices = np.where(~match_mask)[0]
            # Reorder input_edge_ids so matches come first
            ranked_indices = np.concatenate([match_indices, non_match_indices])
            input_edge_ids = input_edge_ids[ranked_indices]
        
        history_dst_edge_texts = []
        if input_edge_ids.shape[0] > 0:
            history_dst_edge_texts = self.edge_text[input_edge_ids]
            if isinstance(history_dst_edge_texts, str):
                history_dst_edge_texts = [history_dst_edge_texts]
            history_dst_edge_texts = [f"<dst_node>{dst_node_id}</dst_node>" + edge_text for dst_node_id, edge_text
                                                                                                in 
                                                                                                zip(edge_dst_ids,
                                                                                                history_dst_edge_texts
                                                                                                )]
            
        else:
            history_dst_edge_texts = []
        

        history_dst_edge_texts = [f"<edge>{edge_text}</edge>" for edge_text in history_dst_edge_texts]
        return "\n\n".join(history_dst_edge_texts)
    
    

    def get_memory_dst(self, 
                       src_id, 
                       input_edge_ids, 
                       interaction_cache):
        
        history_dst_node_ids = np.unique(self.get_dst_ids(input_edge_ids))
        history_dst_node_texts = self.get_dst_nodes_texts(src_id,
                                                          history_dst_node_ids,
                                                          interaction_cache=interaction_cache)
        
        return history_dst_node_texts
    

    def get_memory_seq_dst(self, src_id, dst_id):
        """
        获取源节点在截止时间之前与特定目标节点的交互历史，用于序列化推荐任务
        只考虑cut off time之前src和指定dst的交互edge id
        
        Args:
            src_id: 源节点ID
            dst_id: 目标节点ID
        
        Returns:
            str: 格式化的交互历史文本
        """
        # 获取截止时间
        cut_off_time = self.unique_times[self.input_len]
        
        # 创建掩码：在截止时间之前且目标节点匹配
        mask = (self.ctdg.t < cut_off_time) & (self.ctdg.dst == dst_id) & (self.ctdg.src == src_id)
        
        # 获取匹配的边的索引
        matched_edge_ids = self.ctdg.edge_id[mask]
        
        # 如果没有匹配的边，返回空字符串
        if len(matched_edge_ids) == 0:
            return ""
        
        # 获取这些边的文本
        history_edge_texts = self.edge_text[matched_edge_ids]
        
        # 如果只有一个文本，转换为列表
        if isinstance(history_edge_texts, str):
            history_edge_texts = [history_edge_texts]
        
        # 格式化文本
        formatted_texts = [f"<edge>{text}</edge>" for text in history_edge_texts]
        
        return "\n\n".join(formatted_texts)
    
    def get_neighbor_dst(self, 
                         src_id, 
                         interaction_cache:dict = {}):
        neighbor_ids = list(interaction_cache[src_id]['neighbors'])
        neighbor_dst_ids = []
        for neighbor_id in neighbor_ids:
            neighor_history_dst_ids = self.get_dst_ids(np.array(self.input_edges_dict[neighbor_id]))
            neighbor_dst_ids.extend(neighor_history_dst_ids)
        neighbor_dst_ids = np.unique(np.array(neighbor_dst_ids))
        neighbor_dst_texts = self.get_dst_nodes_texts(src_id,
                                                      neighbor_dst_ids,
                                                      interaction_cache=interaction_cache)
        return neighbor_dst_texts

    def cal_sim(self):
        if hasattr(self, 'sim_mean') and hasattr(self, 'sim_std'):
            return self.sim_mean, self.sim_std

        if self.node_feature.size == 0:
            raise ValueError("Node feature matrix is empty, cannot compute similarity.")

        sim = self.node_feature @ self.node_feature.T
        self.sim_mean = sim.mean()
        self.sim_std = sim.std()

        return self.sim_mean, self.sim_std
    
if __name__ == "__main__":

    # def generate_temporal_graphs():
    #     """生成时序图数据并转换为CTDG格式"""
    #     # 设置参数
    #     num_days = 8
    #     points_per_day = 10
        
    #     # 初始化存储结构
    #     temporal_graphs = []
    #     graph_times = []
        
    #     # 生成每个时间点的图
    #     for day in range(num_days):
    #         current_time = day * 24 
    #         # 生成当前时间点的图
    #         # 这里使用随机图作为示例，实际应用中需要替换为真实的图生成逻辑
    #         num_nodes = 100  # 示例节点数
    #         edge_prob = 0.1  # 示例边概率
            
    #         # 生成邻接矩阵
    #         adj_matrix = np.random.binomial(points_per_day, edge_prob, (num_nodes, num_nodes))
    #         # 确保无自环
    #         np.fill_diagonal(adj_matrix, 0)
            
    #         # 转换为边列表格式
    #         src_nodes, dst_nodes = np.where(adj_matrix > 0)
    #         edge_times = np.full_like(src_nodes, current_time)
    #         edge_ids = np.arange(len(src_nodes))
            
    #         # 创建CTDG窗口
    #         ctdg_window = TemporalData(
    #             src=src_nodes,
    #             dst=dst_nodes,
    #             t=edge_times,
    #             edge_id=edge_ids
    #         )
    #         temporal_graphs.append(ctdg_window)
        
    #     # 使用PyTorch张量直接聚合所有时序图
    #     # 将所有图的边合并到一个大图中
    #     all_src_nodes = torch.cat([torch.tensor(graph.src) for graph in temporal_graphs])
    #     all_dst_nodes = torch.cat([torch.tensor(graph.dst) for graph in temporal_graphs])
    #     all_edge_times = torch.cat([torch.tensor(graph.t) for graph in temporal_graphs])
    #     all_edge_ids = torch.cat([torch.tensor(graph.edge_id) for graph in temporal_graphs])
        
        
    #     # 创建聚合后的单个图
    #     aggregated_graph = TemporalData(
    #         src=all_src_nodes.numpy(),
    #         dst=all_dst_nodes.numpy(), 
    #         t=all_edge_times.numpy(),
    #         edge_id=all_edge_ids.numpy()
    #     )
    #     return aggregated_graph
        
    # # 创建CTDG对象
    # ctdg = generate_temporal_graphs()
    
    # # 创建BWRCTDGDataset
    # bwr_size = 5 # 
    # input_window_len = 7  # 示例输入窗口长度
    # node_feature = np.zeros(((max(ctdg.src.max(), ctdg.dst.max()))+1, 10))
    # edge_feature = np.zeros((len(ctdg.src), 10))
    
    # bwr_dataset = BWRCTDGDataset(
    #     ctdg=ctdg,
    #     bwr=bwr_size,
    #     time_window=24,
    #     node_feature=node_feature,
    #     edge_feature=edge_feature
    # )
    
    
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(bwr_dataset, batch_size=5)
    # for batch in data_loader:
    #     print(batch)


    # bwr_ctdg = BWRCTDGALLDataset(
    #     pred_ratio=0.15,
    #     bwr=2048,
    #     time_window=24*60*60,
    #     # root='data/weibo_tech',
    #     root = 'data/8days_dytag_small_text',
    #     # use_feature='bert',
    #     use_feature='no',
    #     # cm_order=True,
    #     cm_order= False,
    #     force_reload=True
    # ) # weibo 每天是无重边的
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_ratio', type=float, default=0.15,
                        help='预测比例')
    parser.add_argument('--root', type=str, default='data/weibo_tech',
                        help='数据根目录')
    parser.add_argument('--time_window', type=int, default=24*60*60,
                        help='时间窗口大小(秒)')
    parser.add_argument('--bwr', type=int, default=2048,
                        help='BWR大小')
    parser.add_argument('--use_feature', type=str, default='bert',
                        choices=['bert', 'no'],
                        help='是否使用特征')
    parser.add_argument('--cm_order', type=bool, default=True, help='是否使用cm_order')
    parser.add_argument('--force_reload', action='store_true',
                        help='是否强制重新加载数据')
    
    args = parser.parse_args()
    
    bwr_ctdg = BWRCTDGALLDataset(
        pred_ratio=args.pred_ratio,
        bwr=args.bwr,
        time_window=args.time_window,
        root=args.root,
        use_feature=args.use_feature,
        cm_order=args.cm_order,
        # force_reload=args.force_reloadœ
    )
    



    print(f"Root: {bwr_ctdg.root}")
    print(f"BWR: {bwr_ctdg.bwr}")
    print(f"Node feature: {bwr_ctdg.data.node_feature.shape}")
    print(f"Edge feature: {bwr_ctdg.data.edge_feature.shape}")
   
    test_input_time = bwr_ctdg.test_data.unique_times[bwr_ctdg.test_data.input_len]
    mask_pred = bwr_ctdg.data.ctdg.t < test_input_time
    output_edge_ids = torch.concat(list(torch.tensor(v) for v in bwr_ctdg.test_data.output_edges_dict.values()))
    input_edge_ids = torch.concat(list(torch.tensor(v) for v in bwr_ctdg.test_data.input_edges_dict.values()))
    print(f"input_length_day: {bwr_ctdg.train_data.input_len}")
    print(f"pred_length_day: {bwr_ctdg.train_data.pred_len}")
    print(f"graph_length_day: {bwr_ctdg.train_data.input_len+ 3*bwr_ctdg.train_data.pred_len}")
    print(f"avg_edge_length_day: {np.mean(bwr_ctdg.data.ctdg.src.shape[0]/(bwr_ctdg.train_data.input_len+ 3*bwr_ctdg.train_data.pred_len))}")
    print(f"cat_num: {bwr_ctdg.train_data.label_feature.shape[0]}")

    print(f"input_edge_length_test: {input_edge_ids.shape[0]}")
    print(f"prediction_edge_length_test: {output_edge_ids.shape[0]}")

    # # 测试数据集的 src_node_degree 按照 N*T 对第0维求sum，求mean，max，min
    # test_src_node_degree = bwr_ctdg.data.ctdg_src_node_degree.sum(axis=0)
    # # 按照 N*T，第0维求sum
    # degree_sum = test_src_node_degree.sum()
    # degree_mean = test_src_node_degree.mean()
    # degree_max = test_src_node_degree.max()
    # degree_min = test_src_node_degree.min()
    # print("Test dataset src_node_degree stats per time window:")
    # print("Sum:", degree_sum)
    # print("Mean:", degree_mean)
    # print("Max:", degree_max)
    # print("Min:", degree_min)