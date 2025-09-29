import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from .models.TGAT import TGAT
from .models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from .models.CAWN import CAWN
from .models.TCL import TCL
from .models.GraphMixer import GraphMixer
from .models.DyGFormer import DyGFormer
from .models.modules import MergeLayer
from .models.modules import MLPClassifier, MLPClassifier_edge
from .utils.DataLoader import get_idx_data_loader, get_link_prediction_data, get_edge_classification_data
from .utils.utils import get_neighbor_sampler

def load_checkpoint(model: nn.Module, 
                    model_name: str,
                    save_model_path: str,
                    map_location: str = None):
        """
        load model at save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        model.load_state_dict(torch.load(save_model_path, map_location=map_location))
        if model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")
        if model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.node_raw_messages = torch.load(save_model_nonparametric_data_path, map_location=map_location)


def create_link_prediction_model(model_name,
                                 save_model_path: str,
                                 node_raw_features,
                                 edge_raw_features,
                                 data,
                                 neighbor_sampler,
                                 device:str = "cpu"): # create model
    # Save model config as JSON
    config_path = os.path.join(os.path.dirname(save_model_path), f"{os.path.splitext(os.path.basename(save_model_path))[0]}_config.json")
    # Load args_dict from JSON config if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            args_dict = json.load(f)
    else:
        args_dict = {}  # or set to default values as needed

    args_dict["device"] = device
    required_keys_map = {
        "GraphMixer":[
            "time_feat_dim", "num_neighbors", "num_layers", "dropout", "device"
        ]
    }
    
    args_dict = {k: args_dict[k] for k in required_keys_map[model_name]}

    if model_name == 'TGAT':
        dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler, ** args_dict)
    elif model_name in ['JODIE', 'DyRep', 'TGN']:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(data.src_node_ids, data.dst_node_ids, data.node_interact_times)
        dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'CAWN':
        dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'TCL':
        dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    else:
        raise ValueError(f"Wrong value for model_name {model_name}!")
    link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                hidden_dim=node_raw_features.shape[1], output_dim=1)
    model = nn.Sequential(dynamic_backbone, link_predictor)

    load_checkpoint(model,
                    model_name,
                    save_model_path,
                    map_location=args_dict["device"])  # Load the model state dict from the saved path
    return model
         
def create_edge_classification_model(model_name,
                                 save_model_path: str,
                                 node_raw_features,
                                 edge_raw_features,
                                 data,
                                 neighbor_sampler,
                                 cat_num:int,
                                 device:str = "cpu"): # create model
    # Save model config as JSON
    config_path = os.path.join(os.path.dirname(save_model_path), f"{os.path.splitext(os.path.basename(save_model_path))[0]}_config.json")
    # Load args_dict from JSON config if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            args_dict = json.load(f)
    else:
        args_dict = {}  # or set to default values as needed

    args_dict["device"] = device
    required_keys_map = {
        "GraphMixer":[
            "time_feat_dim", "num_neighbors", "num_layers", "dropout", "device"
        ]
    }
    
    args_dict = {k: args_dict[k] for k in required_keys_map[model_name]}

    if model_name == 'TGAT':
        dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler, ** args_dict)
    elif model_name in ['JODIE', 'DyRep', 'TGN']:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(data.src_node_ids, data.dst_node_ids, data.node_interact_times)
        dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'CAWN':
        dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'TCL':
        dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    elif model_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,** args_dict)
    else:
        raise ValueError(f"Wrong value for model_name {model_name}!")
    
    edge_classifier = MLPClassifier_edge(
    input_dim=node_raw_features.shape[1], 
    dropout=args_dict.get("dropout",0.1), 
    cat_num=cat_num)
    model = nn.Sequential(dynamic_backbone, edge_classifier)

    load_checkpoint(model,
                    model_name,
                    save_model_path,
                    map_location=args_dict["device"])  # Load the model state dict from the saved path
    return model

def compute_src_dsts_score(
                        src_ids:np.ndarray,
                        dst_ids:np.ndarray,
                        interact_times:np.ndarray,
                        model_name:str,
                        model:nn.Module,
                        device:str = "cpu",
                        batch_size :int = 32,
                        num_neighbors: int = 20, 
                        time_gap: int = 2000,
                        model_type:str = "lp"):
  
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(dst_ids.shape[0])), batch_size=batch_size, shuffle=False)


    model.eval()
    edge_prediction_scores = []
    for batch_idx, evaluate_data_indices in enumerate(test_idx_data_loader):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                src_ids[evaluate_data_indices],  dst_ids[evaluate_data_indices], \
                interact_times[evaluate_data_indices]

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
             
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            if model_type == "lp":
                probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            elif model_type == "ec":
                probabilities = model[1](x_1=batch_src_node_embeddings, x_2 = batch_dst_node_embeddings, rel_embs = model[0].edge_raw_features)
                # probabilities = torch.max(probabilities, dim=1)[1]

            edge_prediction_scores.append(probabilities)

    edge_prediction_scores = torch.cat(edge_prediction_scores, dim=0) # for ec model, return edge labels 
    return edge_prediction_scores

if __name__ == "__main__":
   
    data_name = "8days_dytag_small_text_en"

    src_id = np.array([0])
    dst_id = np.array([5])
    interact_times = np.array([1])

    # for debug
    from .utils.load_configs import get_link_prediction_args, get_edge_classification_args
    # # get data for training, validation and testing
    
   
    # get arguments
    # args = get_link_prediction_args(is_evaluation=False)
    #args.device = 'cpu'
    # node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num = \
    #     get_link_prediction_data(data_name=args.data_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, args = args)

     
    # full_neighbor_sampler = get_neighbor_sampler(
    #     data=full_data, 
    #     sample_neighbor_strategy=args.sample_neighbor_strategy,
    #     time_scaling_factor=args.time_scaling_factor, seed=0)

    # model = create_link_prediction_model(
    #     model_name="GraphMixer",
    #     save_model_path=os.path.join(save_model_folder, f"{save_model_name}.pkl"),
    #     node_raw_features=node_raw_features,  # Example node features
    #     edge_raw_features=edge_raw_features,  # Example edge features
    #     data=full_data,  # Replace with actual data object if needed
    #     neighbor_sampler=full_neighbor_sampler  # Replace with actual neighbor sampler if needed
    # )

    
    # scores = compute_src_dsts_score(
    #     src_ids=src_id,
    #     dst_ids=dst_id,
    #     interact_times=interact_times,
    #     model_name="GraphMixer",
    #     model=model,
    #     model_type = "lp"
    # )
    # scores
    save_model_folder = "Graphia/ec_models/saved_models/GraphMixer/8days_dytag_small_text_en/edge_classification_GraphMixer_seed0bert/"
    save_model_name = "edge_classification_GraphMixer_seed0bert"

    args = get_edge_classification_args(is_evaluation=False)
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num = \
        get_link_prediction_data(args = args)
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data, 
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor, seed=0)
   
    model = create_edge_classification_model(
        model_name="GraphMixer",
        save_model_path=os.path.join(save_model_folder, f"{save_model_name}.pkl"),
        node_raw_features=node_raw_features,  # Example node features
        edge_raw_features=edge_raw_features,  # Example edge features
        data=full_data,  # Replace with actual data object if needed
        neighbor_sampler=full_neighbor_sampler,  # Replace with actual neighbor sampler if needed
        cat_num = cat_num,
    )
    scores = compute_src_dsts_score(
        src_ids=src_id,
        dst_ids=dst_id,
        interact_times=interact_times,
        model_name="GraphMixer",
        model=model,
        model_type = "ec"
    )

    scores