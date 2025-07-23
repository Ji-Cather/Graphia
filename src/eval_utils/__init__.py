
import numpy as np
import torch
from torch_geometric.data import TemporalData

from ..utils.bwr_ctdg import BWRCTDGALLDataset, BWRCTDGDataset


def get_gt_data(data:BWRCTDGDataset,
                node_msg: bool = False,
                edge_msg: bool = False,
                output_edge_ids: np.array = None,
                ):
    if output_edge_ids == None:
        output_edge_ids = torch.concat(list(torch.tensor(v) for v in data.output_edges_dict.values()))
        
    indices = output_edge_ids.int()
    src = data.ctdg.src[indices]
    dst = data.ctdg.dst[indices]
    t = data.ctdg.t[indices]
    if not edge_msg and not node_msg:
        return TemporalData(src=src, dst=dst, t=t)
    elif edge_msg and not node_msg:
        msg = torch.tensor(data.edge_feature[indices], dtype=torch.float32)
    elif not edge_msg and node_msg:
        msg = torch.tensor(np.concatenate([data.node_feature[src], 
                                           data.node_feature[dst]], axis=1), dtype=torch.float32)
    else:
        msg = torch.tensor(np.concatenate([data.node_feature[src], 
                                           data.node_feature[dst], 
                                           data.edge_feature[indices]], axis=1), dtype=torch.float32)
        
    return TemporalData(src=src, dst=dst, t=t, msg=msg)