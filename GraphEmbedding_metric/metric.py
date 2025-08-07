import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import TemporalData
from typing import Dict


class GraphEmbeddingEvaluator:
    """
    Evaluator for Continuous-Time Dynamic Graphs (CTDGs) using the GraphEmbedding-Metric.

    The GraphEmbedding-Metric compares generated CTDGs against reference graphs by projecting
    node and graph representations into lower-dimensional spaces and measuring similarity.

    Args:
        node_dim (int, optional): Dimension for node embeddings. Default: 100
        graph_dim (int, optional): Dimension for graph embeddings. Default: 100
        seed (int, optional): Random seed for reproducibility. Default: 42
    """

    def __init__(self, 
                 node_dim: int = 100, 
                 graph_dim: int = 100, 
                 seed: int = 42,
                 max_events: int = 1e6):
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.seed = seed
        self.MAX_EVENTS = int(max_events) #Maximum number of events to consider

    def eval(self, input_dict: Dict[str, TemporalData]) -> Dict[str, float]:
        """
        Evaluate the similarity between a generated CTDG and a reference CTDG.

        Args:
            input_dict (Dict[str, TemporalData]): Dictionary containing:
                - 'reference': Reference CTDG as PyG TemporalData
                - 'generated': Generated CTDG as PyG TemporalData

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'GraphEmbedding-Metric': Similarity score (higher is better)
        """
        reference_graph = input_dict["reference"]
        generated_graph = input_dict["generated"]
        reference_node_features = input_dict['reference_node']
        generated_node_features = input_dict['generated_node']

        # Extract graph representations
        reference_embedding = self._compute_graph_embedding(reference_graph, reference_node_features)
        generated_embedding = self._compute_graph_embedding(generated_graph, generated_node_features)

        # Compute cosine similarity between graph embeddings
        similarity = self._compute_similarity(reference_embedding, generated_embedding)

        return {"GraphEmbedding-Metric": similarity}

    def _compute_graph_embedding(self, 
                                 events: TemporalData, node_features) -> Tensor:
        """
        Compute the graph-level embedding for a CTDG using GraphEmbedding projection.

        Args:
            events (TemporalData): PyG TemporalData object containing the CTDG

        Returns:
            Tensor: Graph-level embedding
        """
        return GraphEmbedding_metric(events, 
                                     node_features, 
                                     self.node_dim, 
                                     self.graph_dim, 
                                     self.seed,
                                     self.MAX_EVENTS)

    def _compute_similarity(self, embedding1: Tensor, embedding2: Tensor) -> float:
        """
        Compute the similarity between two graph embeddings.

        Args:
            embedding1 (Tensor): First graph embedding
            embedding2 (Tensor): Second graph embedding

        Returns:
            float: Similarity score (higher is better)
        """
        # Normalize embeddings
        norm1 = torch.norm(embedding1, dim=0, keepdim=True)
        norm2 = torch.norm(embedding2, dim=0, keepdim=True)

        embedding1_normalized = embedding1 / (norm1 + 1e-8)
        embedding2_normalized = embedding2 / (norm2 + 1e-8)

        # Compute cosine similarity
        similarity = torch.mean(
            torch.sum(embedding1_normalized * embedding2_normalized, dim=0)
        )

        return similarity.item()


def create_node_representations(
    events: TemporalData, node_features, embd: Tensor
) -> Dict[int, Tensor]:
    """
    Create node representations from temporal events.

    Args:
        events (TemporalData): PyG TemporalData object containing the CTDG
        embd (Tensor): Random embedding matrix

    Returns:
        Dict[int, Tensor]: Dictionary mapping node IDs to their representations
    """
    node_reps = {}

    # Min-max normalization for messages
    msg_min = events.msg.min(dim=0)[0]
    msg_max = events.msg.max(dim=0)[0]
    normalized_msgs = (events.msg - msg_min) / (
        msg_max - msg_min + 1e-6
    )  # Avoid division by zero

    # Min-max normalization for time
    time_min = events.t.min()
    time_max = events.t.max()
    normalized_times = (events.t - time_min) / (time_max - time_min + 1e-6)

    def normalize_features(features_dict):
        normalized_features = {}
        for node, features in features_dict.items():
            msg_min = features.min(dim=0, keepdim=True)[0]
            msg_max = features.max(dim=0, keepdim=True)[0]
            normalized_features[node] = (features - msg_min) / (msg_max - msg_min + 1e-6)
        return normalized_features

    node_features = normalize_features(node_features)



    for i in range(events.src.size(0)):
        src, dst = int(events.src[i].item()), int(events.dst[i].item())
        msg = normalized_msgs[i]
        time_enc = normalized_times[i].reshape(1)

        try:
            src_node_msg = node_features[src]
        except Exception as e:
            print(e)
            print("random src node msg")
            src_node_msg = torch.randn(768)
        
        try:
            dst_node_msg = node_features[dst]
        except Exception as e:
            print(e)
            print("random dst node msg")
            dst_node_msg = torch.randn(768)

        combined_src = torch.cat([dst_node_msg, msg, time_enc, torch.tensor([0])], dim=0)
        combined_dst = torch.cat([src_node_msg, msg, time_enc, torch.tensor([1])], dim=0)

        if src not in node_reps:
            node_reps[src] = []
        if dst not in node_reps:
            node_reps[dst] = []

        node_reps[src].append(combined_src)
        node_reps[dst].append(combined_dst)
    

    

    for node in node_reps.keys():
        event_reps = torch.cat(node_reps[node]) 
        try:
            feat_rep = node_features[node]
        except Exception as e:
            print(e)
            print("random node msg")
            feat_rep = torch.randn(768)
        combined = torch.cat([event_reps, feat_rep]) 
        

        total_dim = len(combined)
        output_dim = embd.shape[1] 

        if total_dim > embd.shape[0]:
            # print('total_dim > embd.shape[0]',total_dim,embd.shape[0])
            combined = combined[:embd.shape[0]]
            proj = combined @ embd[:, :output_dim]
        else:
            proj = combined @ embd[:total_dim, :output_dim]
        
        node_reps[node] = proj
        
    return node_reps


def GraphEmbedding_project_graph_level(node_reps: Dict[int, Tensor], proj_dim: int) -> Tensor:
    """
    Project node representations to graph-level representation.

    Args:
        node_reps (Dict[int, Tensor]): Dictionary mapping node IDs to their representations
        proj_dim (int): Projection dimension

    Returns:
        Tensor: Graph-level representation
    """
    node_dim = list(node_reps.values())[0].size(0)
    num_nodes = len(node_reps.keys())

    # Reindex nodes
    node_reps = {index + 1: feature for index, feature in enumerate(node_reps.values())}

    # Unstructured ROM-based method
    orthonormal_matrix = torch.randn((num_nodes, proj_dim)) * torch.sqrt(
        torch.tensor(1.0 / num_nodes)
    )

    # Create node matrix
    node_matrix = torch.zeros((num_nodes, node_dim))
    for node_id, rep in node_reps.items():
        node_matrix[int(node_id) - 1] = rep

    # Project to graph level
    graph_rep = node_matrix.T @ orthonormal_matrix

    return graph_rep


def GraphEmbedding_metric(
    events: TemporalData, 
    node_features, 
    node_proj_dim: int, 
    graph_proj_dim: int, 
    seed: int,
    max_events: int = 100000
) -> Tensor:
    """
    Compute GraphEmbedding-Metric representation for a CTDG.

    Args:
        events (TemporalData): PyG TemporalData object containing the CTDG
        node_proj_dim (int): Dimension for node projections
        graph_proj_dim (int): Dimension for graph projections
        seed (int): Random seed for reproducibility

    Returns:
        Tensor: Graph representation
    """
    torch_geometric.seed_everything(seed)
    MAX_EVENTS = int(max_events)  # hardcoded maximum

    # Create random projection matrix
    embd = torch.randn((MAX_EVENTS, node_proj_dim)) * torch.sqrt(
        torch.tensor(1.0 / MAX_EVENTS)
    )

    # Create node representations
    node_reps = create_node_representations(events, node_features, embd)

    # Project to graph level
    graph_rep = GraphEmbedding_project_graph_level(node_reps, graph_proj_dim)

    return graph_rep