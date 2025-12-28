
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

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray,
                 edge_ids_split: np.ndarray = None,):
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
        self.edge_ids_split = edge_ids_split
        




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
    if args.data_name in ["8days_dytag_small_text_en",
                        "weibo_daily",
                        "weibo_tech",
                        "weibo_daily_long"]:
        # train for dytag dataset

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
    if args.data_name in ["8days_dytag_small_text_en",
                        "weibo_daily",
                        "weibo_tech"]:
        # train for dytag dataset
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
                        full_data_temp.edge_id.cpu().numpy(),
                        )
        train_data = Data(
            train_data_temp.src.cpu().numpy(),
            train_data_temp.dst.cpu().numpy(),
            train_data_temp.t.cpu().numpy(),
            train_data_temp.edge_id_all.cpu().numpy(),
            train_data_temp.label.cpu().numpy(), 
            train_data_temp.edge_id.cpu().numpy(),
        )
        val_edge_ids = val_data_temp.edge_id_all.cpu().numpy()

        val_edge_output_mask = (val_data_temp.t >= bwr_ctdg.val_data.unique_times[bwr_ctdg.val_data.input_len])


        val_data = Data(
            val_data_temp.src.cpu().numpy()[val_edge_output_mask],
            val_data_temp.dst.cpu().numpy()[val_edge_output_mask],
            val_data_temp.t.cpu().numpy()[val_edge_output_mask],
            val_data_temp.edge_id_all.cpu().numpy()[val_edge_output_mask],
            val_data_temp.label.cpu().numpy()[val_edge_output_mask],
            val_data_temp.edge_id.cpu().numpy()[val_edge_output_mask],
        )

        new_node_val_data = Data(
            val_data_temp.src[val_edge_output_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.dst[val_edge_output_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.t[val_edge_output_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.edge_id_all[val_edge_output_mask & val_data_temp.new_node].cpu().numpy(),
            val_data_temp.label[val_edge_output_mask & val_data_temp.new_node].cpu().numpy(),
        )
        


        test_edge_output_mask = (test_data_temp.t >= bwr_ctdg.test_data.unique_times[bwr_ctdg.test_data.input_len])

        test_data = Data(
            test_data_temp.src.cpu().numpy()[test_edge_output_mask],
            test_data_temp.dst.cpu().numpy()[test_edge_output_mask],
            test_data_temp.t.cpu().numpy()[test_edge_output_mask],
            test_data_temp.edge_id_all.cpu().numpy()[test_edge_output_mask], 
            test_data_temp.label.cpu().numpy()[test_edge_output_mask],
            test_data_temp.edge_id.cpu().numpy()[test_edge_output_mask],
        )
        
        new_node_test_data = Data(
                        test_data_temp.src[test_edge_output_mask & test_data_temp.new_node].cpu().numpy(),  
                        test_data_temp.dst[test_edge_output_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.t[test_edge_output_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.edge_id_all[test_edge_output_mask & test_data_temp.new_node].cpu().numpy(), 
                        test_data_temp.label[test_edge_output_mask & test_data_temp.new_node].cpu().numpy(),
                        test_data_temp.edge_id.cpu().numpy()[test_edge_output_mask & test_data_temp.new_node],
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