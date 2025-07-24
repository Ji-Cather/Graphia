
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import networkx as nx

from torch_geometric.data import InMemoryDataset
import torch
import os

from .bwr_ctdg import BWRCTDGDataset, BWRCTDGALLDataset
from .timefeatures import time_features
class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False,
                             num_workers=2)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        
class NodeSeqALLData(InMemoryDataset):
    
    def __init__(self, 
                data_name, # ctdg&dtcg all dataset name
                val_ratio:float = 0.1,
                test_ratio:float = 0.1,
                pred_len:int = 1, # pred len
                input_len:int = 5,
                seq_len:int = 6,
                plot:bool = False,
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
        self.data_name = data_name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seq_len = seq_len
        self.input_len = input_len
        self.plot = plot
        self.pred_len = pred_len
        assert self.input_len + self.pred_len == self.seq_len, "input_len + pred_len must be equal to seq_len"
        super().__init__(root, 
                         transform, 
                         pre_transform,
                         force_reload=force_reload,
                         )
        
            
    @property
    def raw_file_names(self) -> str:
        return []

    @property
    def processed_file_names(self) -> str:
        return ['src_node_ids_{mode}.pt',
                'src_node_degrees_{mode}.pt',
                'src_node_interact_times_{mode}.pt',
                'src_node_time_features_{mode}.pt',
                'snapshot_edges_{mode}.npy'
                ]
        
    def process_node_seq_data(self, snapshots, max_src_node_id, resample, shuffle, mode):
        try:
            src_node_ids = torch.load(self.processed_paths[0].format(mode=mode),
                                                map_location="cpu")
            src_node_degrees = torch.load(self.processed_paths[1].format(mode=mode),
                                                map_location="cpu")
            src_node_interact_times = torch.load(self.processed_paths[2].format(mode=mode),
                                                map_location="cpu")
            src_node_time_features = torch.load(self.processed_paths[3].format(mode=mode),
                                                map_location="cpu")
            snapshot_edges = np.load(self.processed_paths[4].format(mode=mode), allow_pickle=True)
            return src_node_ids, src_node_degrees, src_node_interact_times, src_node_time_features, snapshot_edges
        except:
            all_seq_len = len(snapshots)
            src_node_degrees = np.zeros((max_src_node_id + 1, all_seq_len))
            src_node_interact_times = np.zeros((max_src_node_id + 1, all_seq_len))
            snapshot_edges = []
            
            for i, snapshot in enumerate(snapshots):
                # 直接使用NetworkX的degree函数获取节点度数
                time = max([data[2]['ts'] for data in snapshot.edges(data=True)])
                edges = sorted(snapshot.edges(data=True), key=lambda x: x[0])
                edges = np.array([[data[2]['id'], data[2]['ts'], data[1]] for data in edges])
                
                adj_list = dict(snapshot.degree())
                # 计算每个节点的度数
                src_node_degrees[:, i] = np.array([adj_list.get(node, 0) for node in range(max_src_node_id + 1)])
                src_node_interact_times[:, i] = np.full((max_src_node_id + 1,), time)
                src_edges = {0: edges[: int(src_node_degrees[0,i])]}
                src_edges.update(
                    {node: edges[int(src_node_degrees[:node,i].sum()): int(src_node_degrees[:node+1,i].sum())] 
                                    for node in range(1, max_src_node_id + 1)}
                                    )
                snapshot_edges.append(src_edges)
                
            src_node_ids = np.arange(max_src_node_id + 1)
            src_node_time_features = []
            for i in range(all_seq_len):
                time = pd.to_datetime(src_node_interact_times[:, i], unit='s')
                src_node_time_features.append(time_features(time, timeenc=0, freq='d'))
            src_node_time_features = np.stack(src_node_time_features, axis=1)
                
            if mode == "train" and resample:
            # if False: # tbm
                
                # 计算每个节点序列最后一个时间步的degree
                last_degrees = src_node_degrees[:, -1]
                
                # 计算degree的量级(取log10)
                degree_magnitudes = np.floor(np.log10(last_degrees + 1))
                
                # 统计每个量级的样本数量
                unique_magnitudes, magnitude_counts = np.unique(degree_magnitudes, return_counts=True)
                
                # 计算每个量级的采样权重(倒数)
                magnitude_weights = 1.0 / magnitude_counts
                magnitude_weights = magnitude_weights / np.sum(magnitude_weights)
                
                # 为每个样本分配权重
                sample_weights = np.zeros_like(last_degrees)
                for mag, weight in zip(unique_magnitudes, magnitude_weights):
                    sample_weights[degree_magnitudes == mag] = weight
                    
                # 按权重进行重采样
                num_samples = len(last_degrees)
                sampled_indices = np.random.choice(
                    np.arange(num_samples),
                    size=num_samples,
                    p=sample_weights/np.sum(sample_weights),
                    replace=True
                )
                
                # 根据采样的索引重新构建数据
                src_node_degrees = src_node_degrees[sampled_indices]
                src_node_interact_times = src_node_interact_times[sampled_indices]
                src_node_time_features = src_node_time_features[sampled_indices]
                src_node_ids = src_node_ids[sampled_indices]
                snapshot_edges = [snapshot_edges[i] for i in sampled_indices]
            else:
                src_node_ids = src_node_ids
                src_node_degrees = src_node_degrees
                src_node_interact_times = np.array(src_node_interact_times)
                src_node_time_features = src_node_time_features
                snapshot_edges = snapshot_edges
                
            # 随机打乱数据并重新组织
            if mode == "train" and shuffle:
                # 创建索引数组
                indices = np.arange(len(src_node_ids))
                # 随机打乱索引
                np.random.shuffle(indices)
                # 将打乱后的数据赋值回原变量
                src_node_ids = src_node_ids[indices]
                src_node_degrees = src_node_degrees[indices]
                src_node_interact_times = src_node_interact_times[indices]
                src_node_time_features = src_node_time_features[indices]
                snapshot_edges = [snapshot_edges[i] for i in indices]

            snapshot_edges = np.array(snapshot_edges, dtype=object)
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(src_node_ids, self.processed_paths[0].format(mode=mode))
            torch.save(src_node_degrees, self.processed_paths[1].format(mode=mode))
            torch.save(src_node_interact_times, self.processed_paths[2].format(mode=mode))
            torch.save(src_node_time_features, self.processed_paths[3].format(mode=mode))
            np.save(self.processed_paths[4].format(mode=mode), snapshot_edges)

            return src_node_ids, src_node_degrees, src_node_interact_times, src_node_time_features, snapshot_edges

        
        
    def process_data(self):
        self.read_data()
        
        
    def read_data(self):
        from .utils.market import  MarketSeqDataset
        from torch_geometric.data import TemporalData
        # 将时间戳转换为datetime格式,方便后续分析
        from .utils.item_data import Market
        import torch  
        dataset = MarketSeqDataset(root="/data/jiarui_ji/DGGen/data", 
                            name = self.data_name,
                            # force_reload=True,
                            time_window = 24*60*60)
        item_data = Market(root="/data/jiarui_ji/RQ-VAE-Recommender/dataset/market",
                            split="n_head")
        
        dtcg = dataset.data
        max_src_node_id = int(dataset.ctdg.src.max())
        assert isinstance(dataset.ctdg, TemporalData)
        
        all_seq_len = len(dtcg)
        seq_len = self.seq_len
        train_seq = dtcg[:int((1-self.val_ratio-self.test_ratio)*all_seq_len)]
        val_seq = dtcg[int((1-self.test_ratio)*all_seq_len) - seq_len:int((1-self.test_ratio)*all_seq_len)]
        test_seq = dtcg[-seq_len:]
        train_edge_len = sum([len(list(snapshot.edges(data=True))) for snapshot in train_seq])
        edge_idxs = torch.arange(dataset.ctdg.src.shape[0])
        
        train_edge_map = edge_idxs[:train_edge_len]
        full_data_edge = Data(dataset.ctdg.src.cpu().numpy(), 
                        dataset.ctdg.dst.cpu().numpy(), 
                        dataset.ctdg.t.cpu().numpy(), 
                        edge_idxs.cpu().numpy(), 
                        dataset.ctdg.msg.cpu().numpy()
                        )
        train_data_edge = Data(dataset.ctdg.src[train_edge_map].cpu().numpy(),
                            dataset.ctdg.dst[train_edge_map].cpu().numpy(),
                            dataset.ctdg.t[train_edge_map].cpu().numpy(),
                            edge_idxs[train_edge_map].cpu().numpy(),
                            dataset.ctdg.msg[train_edge_map].cpu().numpy()
                            )
        # 定义特征维度常量
        item_features = item_data.data["item"].x
        # 将PyTorch张量转换为NumPy数组，以便与后续代码兼容
        edge_raw_features = item_features[dataset.ctdg.id.cpu().numpy()].cpu().numpy()
        node_raw_features = dataset.node_feature
        
        max_degree = dataset.ctdg.src.max()
        
        self.node_seq_data = {}
        
        for mode_seq, mode, mode_args in zip([train_seq, val_seq, test_seq], 
                                             ["train", "val", "test"],
                                             [{"resample": False, "shuffle": False}, 
                                              {"resample": False, "shuffle": False}, 
                                              {"resample": False, "shuffle": False}]):
            src_node_ids, src_node_degrees, src_node_interact_times, src_node_time_features, snapshot_edges = \
                self.process_node_seq_data(mode_seq, max_src_node_id, mode = mode,**mode_args)
            src_node_degrees = np.log1p(src_node_degrees + 1e-10).reshape(src_node_degrees.shape[0], -1, 1)
            
            self.node_seq_data[mode] = NodeSeqData(src_node_ids, 
                                                   src_node_degrees, 
                                                   src_node_interact_times, 
                                                   src_node_time_features, 
                                                   node_raw_features, 
                                                   max_src_node_id,
                                                   snapshot_edges,
                                                   self.input_len,
                                                   self.pred_len,
                                                   return_edges=True)
            
            if self.plot:
                import matplotlib.pyplot as plt
                from scipy.stats import boxcox, genextreme, norm
                # 计算所有节点度数的分布
                all_degrees = src_node_degrees[:,-1,:].flatten()
                plt.figure(figsize=(10,6))
                plt.hist(all_degrees, bins=50, density=True)
                plt.xlabel('Node Degree (log)')
                plt.ylabel('Frequency')
                plt.title('Node Degree Distribution')
                plt.grid(True)
                plt.savefig(f'log_node_degree_distribution_{mode}.png')
                plt.clf()
        

        
        return node_raw_features, edge_raw_features, full_data_edge, train_data_edge, \
        self.node_seq_data["train"], self.node_seq_data["val"], self.node_seq_data["test"], max_degree, \
        train_seq, val_seq, test_seq
    
    
    
   
   
    
class NodeSeqData:
    def __init__(self, 
                 src_node_ids, 
                 src_node_degrees, 
                 src_node_interact_times, 
                 src_node_time_features, 
                 node_raw_features,
                 max_src_node_id,
                 snapshot_edges,
                 input_len,
                 pred_len,
                 return_edges=False
                 ):
        # edge_id, td, dst_node_id
        self.src_node_ids = src_node_ids
        self.src_node_degrees = src_node_degrees
        self.src_node_interact_times = src_node_interact_times
        self.src_node_time_features = src_node_time_features
        self.node_raw_features = node_raw_features
        self.snapshot_edges = snapshot_edges
        self.input_len = input_len
        self.pred_len = pred_len
        assert self.input_len >0 and self.pred_len >0, "input_len and pred_len must be greater than 0"
        self.max_src_node_id = max_src_node_id
        self.src_node_feature_dim = src_node_degrees.shape[-1] + node_raw_features.shape[1]
        self.dst_node_feature_dim = node_raw_features.shape[1]
        
        self.src_node_y_dim = 1 # to be modified, may be dau + time
        self.return_edges = return_edges
    
    def __getitem__(self, idx):
        """
        获取指定索引的节点序列数据
        :param idx: int, 索引
        :return: tuple, 
        """
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        # return x_feature,x_timestamp,  y_feature, y, y_timestamp, node_idx
        x_feature, x_timestamp, y_feature, y, y_timestamp, node_idx = \
            np.concatenate([self.src_node_degrees[idx][:, :self.input_len], 
                          self.node_raw_features[idx].reshape(idx.shape[0],1,-1).repeat(self.input_len, 1)], axis=2), \
                self.src_node_time_features[idx][:, :self.input_len], \
                np.concatenate([self.src_node_degrees[idx][:, -self.input_len-self.pred_len:-self.pred_len], 
                self.node_raw_features[idx].reshape(idx.shape[0],1,-1).repeat(self.input_len, 1)], axis=2), \
                self.src_node_degrees[idx][:, -self.pred_len:], \
                self.src_node_time_features[idx][:, -self.input_len-self.pred_len:-self.pred_len], \
                self.src_node_ids[idx]
        
        if self.return_edges:
            # input edges, pred edges [input_len, pred_len]
            input_edges = [[self.snapshot_edges[j][int(_)] for j in range(self.input_len)] for _ in idx]
            pred_edges = [[self.snapshot_edges[j][int(_)] for j in range(self.input_len, 
                                                                         self.input_len+self.pred_len)] for _ in idx]
        
            return x_feature, x_timestamp, y_feature, y, y_timestamp, node_idx,input_edges, pred_edges
        else:
            return x_feature, x_timestamp, y_feature, y, y_timestamp, node_idx
        
        
    def __len__(self):
        return len(self.src_node_degrees)
    
    def plot_batch_degree_distribution(self, mode, num_batches=8):
        """
        随机抽样8个batch并绘制其节点度数分布直方图
        :param num_batches: int, 要抽样的batch数量
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 获取数据集的总长度
        total_length = len(self.src_node_degrees)
        
        # 确保要抽样的batch数量不超过总数据量
        num_batches = min(num_batches, total_length)
        
        # 随机选择batch的索引
        batch_indices = np.random.choice(total_length, num_batches, replace=False)
        
        # 创建子图网格
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 为每个选中的batch绘制直方图
        for i, idx in enumerate(batch_indices):
            # 获取当前batch的节点度数
            batch_degrees = self.src_node_degrees[idx].flatten()
            
            # 在对应的子图上绘制直方图
            axes[i].hist(batch_degrees, bins=30, density=True, alpha=0.7)
            axes[i].set_title(f'Batch {idx} 节点度数分布')
            axes[i].set_xlabel('节点度数 (log)')
            axes[i].set_ylabel('频率')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'batch_node_degree_distribution_{mode}.png')
        plt.clf()
        plt.close()
        
def get_node_classification_data(data_name: str, val_ratio: float, test_ratio: float, args):
    """
    chop the dataset into snapshots
    """

    dataset_all = NodeSeqALLData(data_name,
                                 val_ratio = val_ratio,
                                 test_ratio = test_ratio,
                                 root = f"data/{data_name}",
                                 seq_len = 6,
                                 input_len = 5,
                                 pred_len = 1)    
    return dataset_all.read_data()


def get_ctdg_generation_data(args):
    """
    get the data for ctdg generation task
    
    train_data_ctdg:
    past_len, pred_len, time_window, bwr
    ctdg
    
    """
    bwr_ctdg = BWRCTDGALLDataset(
            pred_ratio= args.pred_ratio,
            bwr=args.bwr,
            use_feature=args.use_feature,
            time_window=args.time_window,
            root= os.path.join(args.data_root, args.data_name),
            cm_order = args.cm_order
        )
    return bwr_ctdg.train_data, bwr_ctdg.val_data, bwr_ctdg.test_data
        
def get_link_prediction_data(args):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param data_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    if args.data_name in ["8days_dytag_small_text_en"]:
        # train for dytag dataset
        from .utils.bwr_ctdg import BWRCTDGDataset, BWRCTDGALLDataset
        import torch  
        bwr_ctdg = BWRCTDGALLDataset(
            pred_ratio= args.pred_ratio,
            bwr=args.bwr,
            use_feature=args.use_feature,
            time_window=args.time_window,
            root=os.path.join(args.data_root, args.data_name),
            cm_order = args.cm_order
        )
        node_raw_features = bwr_ctdg.train_data.node_feature
        
        edge_raw_features = bwr_ctdg.data.edge_feature
        
        assert edge_raw_features.shape[0] == bwr_ctdg.data.ctdg.edge_id_all.shape[0]


        full_data_temp = bwr_ctdg.data.ctdg
        train_data_temp = bwr_ctdg.train_data.ctdg
        val_data_temp = bwr_ctdg.val_data.ctdg
        test_data_temp = bwr_ctdg.test_data.ctdg

        
        full_data = Data(full_data_temp.src.cpu().numpy(), 
                        full_data_temp.dst.cpu().numpy(), 
                        full_data_temp.t.cpu().numpy(), 
                        full_data_temp.edge_id_all.cpu().numpy(), 
                        full_data_temp.label.cpu().numpy(), 
                        )
        edge_ids_seen = set(train_data_temp.edge_id_all.cpu().numpy().tolist())
        train_data = Data(
            train_data_temp.src.cpu().numpy(),
            train_data_temp.dst.cpu().numpy(),
            train_data_temp.t.cpu().numpy(),
            train_data_temp.edge_id_all.cpu().numpy(),
            train_data_temp.label.cpu().numpy(), 
        )
        val_edge_ids = val_data_temp.edge_id_all.cpu().numpy()
        edge_unseen_mask = torch.tensor(np.array([eid not in edge_ids_seen for eid in val_edge_ids]))

        val_data = Data(
            val_data_temp.src.cpu().numpy()[edge_unseen_mask],
            val_data_temp.dst.cpu().numpy()[edge_unseen_mask],
            val_data_temp.t.cpu().numpy()[edge_unseen_mask],
            val_data_temp.edge_id_all.cpu().numpy()[edge_unseen_mask],
            val_data_temp.label.cpu().numpy()[edge_unseen_mask],
        )

        new_node_val_data = Data(
            val_data_temp.src[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.dst[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.t[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.edge_id_all[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.label[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
        )
        edge_ids_seen = set(np.concatenate([train_data_temp.edge_id_all.cpu().numpy(),
                                            val_data_temp.edge_id_all.cpu().numpy()
                                            ]).tolist())
        test_edge_ids = test_data_temp.edge_id_all.cpu().numpy()
        edge_unseen_mask = torch.tensor(np.array([eid not in edge_ids_seen for eid in test_edge_ids]))

        test_data = Data(
            test_data_temp.src.cpu().numpy()[edge_unseen_mask],
            test_data_temp.dst.cpu().numpy()[edge_unseen_mask],
            test_data_temp.t.cpu().numpy()[edge_unseen_mask],
            test_data_temp.edge_id_all.cpu().numpy()[edge_unseen_mask], 
            test_data_temp.label.cpu().numpy()[edge_unseen_mask],
        )
        
        new_node_test_data = Data(test_data_temp.src[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),  
                        test_data_temp.dst[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.t[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.edge_id_all[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(), 
                        test_data_temp.label[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        )
        
        label_num = np.unique(full_data.labels).shape[0]
        return node_raw_features, edge_raw_features, full_data, train_data, \
        val_data, test_data, new_node_val_data, new_node_test_data, label_num


        

    

def get_edge_classification_data(args):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param data_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
     # Load data and train val test split
    if args.data_name in ["8days_dytag_small_text_en"]:
        # train for dytag dataset
        from .utils.bwr_ctdg import BWRCTDGDataset, BWRCTDGALLDataset
        import torch  
        bwr_ctdg = BWRCTDGALLDataset(
            pred_ratio= args.pred_ratio,
            bwr=args.bwr,
            use_feature=args.use_feature,
            time_window=args.time_window,
            root=os.path.join(args.data_root, args.data_name),
            cm_order = args.cm_order
        )
        node_raw_features = bwr_ctdg.train_data.node_feature
        
        edge_raw_features = bwr_ctdg.data.edge_feature
        
        assert edge_raw_features.shape[0] == bwr_ctdg.data.ctdg.edge_id_all.shape[0]


        full_data_temp = bwr_ctdg.data.ctdg
        train_data_temp = bwr_ctdg.train_data.ctdg
        val_data_temp = bwr_ctdg.val_data.ctdg
        test_data_temp = bwr_ctdg.test_data.ctdg

        
        full_data = Data(full_data_temp.src.cpu().numpy(), 
                        full_data_temp.dst.cpu().numpy(), 
                        full_data_temp.t.cpu().numpy(), 
                        full_data_temp.edge_id_all.cpu().numpy(), 
                        full_data_temp.label.cpu().numpy(), 
                        )
        edge_ids_seen = set(train_data_temp.edge_id_all.cpu().numpy().tolist())
        train_data = Data(
            train_data_temp.src.cpu().numpy(),
            train_data_temp.dst.cpu().numpy(),
            train_data_temp.t.cpu().numpy(),
            train_data_temp.edge_id_all.cpu().numpy(),
            train_data_temp.label.cpu().numpy(), 
        )
        val_edge_ids = val_data_temp.edge_id_all.cpu().numpy()
        edge_unseen_mask = torch.tensor(np.array([eid not in edge_ids_seen for eid in val_edge_ids]))

        val_data = Data(
            val_data_temp.src.cpu().numpy()[edge_unseen_mask],
            val_data_temp.dst.cpu().numpy()[edge_unseen_mask],
            val_data_temp.t.cpu().numpy()[edge_unseen_mask],
            val_data_temp.edge_id_all.cpu().numpy()[edge_unseen_mask],
            val_data_temp.label.cpu().numpy()[edge_unseen_mask],
        )

        new_node_val_data = Data(
            val_data_temp.src[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.dst[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.t[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.edge_id_all[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.label[edge_unseen_mask & val_data_temp.new_node].cpu().numpy(),
        )
        edge_ids_seen = set(np.concatenate([train_data_temp.edge_id_all.cpu().numpy(),
                                            val_data_temp.edge_id_all.cpu().numpy()
                                            ]).tolist())
        test_edge_ids = test_data_temp.edge_id_all.cpu().numpy()
        edge_unseen_mask = torch.tensor(np.array([eid not in edge_ids_seen for eid in test_edge_ids]))

        test_data = Data(
            test_data_temp.src.cpu().numpy()[edge_unseen_mask],
            test_data_temp.dst.cpu().numpy()[edge_unseen_mask],
            test_data_temp.t.cpu().numpy()[edge_unseen_mask],
            test_data_temp.edge_id_all.cpu().numpy()[edge_unseen_mask], 
            test_data_temp.label.cpu().numpy()[edge_unseen_mask],
        )
        
        new_node_test_data = Data(test_data_temp.src[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),  
                        test_data_temp.dst[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.t[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.edge_id_all[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(), 
                        test_data_temp.label[edge_unseen_mask & test_data_temp.new_node].cpu().numpy(),
                        )
        
        label_num = np.unique(full_data.labels).shape[0]
        return node_raw_features, edge_raw_features, full_data, train_data, \
        val_data, test_data, new_node_val_data, new_node_test_data, label_num
        

class DegreeQuantileConverter:
    def __init__(self, k: int = 10):
        """
        将连续的degree值转换为k个quantile的表示
        :param k: int, quantile的数量
        """
        self.k = k
        self.quantiles = None
        self.quantile_values = None
        
    def fit(self, degrees: np.ndarray):
        """
        根据输入的degrees计算quantile值
        :param degrees: np.ndarray, shape (N,)
        """
        self.quantile_values = np.quantile(degrees, np.linspace(0, 1, self.k))
        
    def transform(self, degrees: np.ndarray) -> np.ndarray:
        """
        将degrees转换为quantile表示
        :param degrees: np.ndarray, shape (batch_size, seq_len, 1)
        :return: np.ndarray, shape (batch_size, seq_len, k)
        """
        if self.quantile_values is None:
            raise ValueError("请先调用fit方法计算quantile值")
            
        batch_size, seq_len, _ = degrees.shape
        result = np.zeros((batch_size, seq_len, self.k))
        
        # 对每个degree值计算其对应的quantile分布
        for i in range(batch_size):
            for j in range(seq_len):
                degree = degrees[i, j, 0]
                # 计算degree在每个quantile区间的权重
                weights = np.zeros(self.k)
                for k in range(self.k-1):
                    if degree <= self.quantile_values[k+1]:
                        # 计算在当前区间的相对位置
                        pos = (degree - self.quantile_values[k]) / (self.quantile_values[k+1] - self.quantile_values[k])
                        weights[k] = 1 - pos
                        weights[k+1] = pos
                        break
                else:
                    # 如果degree大于所有quantile值，将权重全部分配给最后一个quantile
                    weights[-1] = 1
                result[i, j] = weights
                
        return result