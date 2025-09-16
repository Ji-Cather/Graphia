import os
import sys
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
import sklearn
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components
import powerlaw
import time
import argparse


directory = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(directory)
sys.path.append(parent)
from networkx.algorithms.centrality import closeness_centrality,betweenness_centrality
from torch_geometric.data import TemporalData

def calculate_mmd(x1, x2, beta):
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()

    #print("MMD means", x1x1.mean(),x1x2.mean(),x2x2.mean())

    return diff

def gaussian_kernel(x1, x2, beta = 1.0):
    L=pairwise_distances(x1,x2).reshape(-1)
    return np.exp(-beta*np.square(L))

def average_metric(method_metric, repeated, header, i):
    for metric in header:
        method_metric[i][metric] = method_metric[i][metric] / repeated

def sum_metric(aaa, method_metric, i):
    header = aaa.keys()
    if len(method_metric) <= i:
        method_metric.append(aaa)
    else:
        for metric in header:
            method_metric[i][metric] = method_metric[i][metric] + aaa[metric]


def mean_median(org_graph, generated_graph, f, name):
    org_graph = np.array(org_graph)
    generated_graph = np.array(generated_graph)
    metric = np.divide(np.abs(org_graph - generated_graph), np.abs(org_graph))
    mean = np.mean(metric)
    median = np.median(metric)
    f.write('{}:\n'.format(name))
    f.write('Mean = {}\n'.format(mean))
    f.write('Median = {}\n'.format(median))
    return mean, median


def sampling(network, temporal_graph, n, p=0.5):
    for i in range(n):
        for j in range(n):
            if network[i, j] == 1 and np.random.uniform(low=0.0, high=1) <= p:
                temporal_graph[i, j] = 1

def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha

def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_array(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """
    import igraph
    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)

def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)

def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er

def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:, None].dot(counts[None, :])
        if normalize:
            blocks_outer = np.multiply(block, 1 / blocks_outer)
        return blocks_outer

    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1 - np.eye(in_blocks.shape[0])).mean()
    return diag_mean, offdiag_mean

def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()

def compute_graph_statistics(A_in):
    A = A_in.copy()
    A_graph = nx.from_numpy_array(A).to_undirected()
    statistics = {}
    # start_time = time.time()
    d_max, d_min, d_mean = statistics_degrees(A)
    statistics['d_mean'] = d_mean
    # LCC = statistics_LCC(A)
    # statistics['LCC'] = LCC.shape[0]
    statistics['wedge_count'] = statistics_wedge_count(A)
    claw_count = statistics_claw_count(A)
    # statistics['power_law_exp'] = statistics_power_law_alpha(A)
    statistics['triangle_count'] = statistics_triangle_count(A)
    # statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)
    # statistics['n_components'] = connected_components(A, directed=False)[0]
    # cc = closeness_centrality(A_graph)
    # statistics['closeness_centrality_mean'] = np.mean(list(cc.values()))
    # statistics['closeness_centrality_median'] = np.median(list(cc.values()))
    # cc = betweenness_centrality(A_graph)
    # statistics['betweenness_centrality_mean'] = np.mean(list(cc.values()))
    # statistics['betweenness_centrality_median'] = np.median(list(cc.values()))
    if claw_count != 0:
        statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / claw_count
    else:
        statistics['clustering_coefficient'] = 0

    return statistics


def calculate_between_centrality(A):
    A_graph = nx.from_numpy_array(A).to_undirected()
    return np.array(list(betweenness_centrality(A_graph).values()))

def calculate_closeness_centrality(A):
    A_graph = nx.from_numpy_array(A).to_undirected()
    return np.array(list(closeness_centrality(A_graph).values()))




def create_timestamp_edges(graphs,timestamps):   ### Club the graph snapshot with its timestamp
    edge_lists = []
    node_set = set()
    for graph, time in zip(graphs,timestamps):
        edge_list = []
        for start,adjl in graph.items():
            for end,ct in adjl.items():
                edge_list.append((start,end,time))
                node_set.add(end)
            node_set.add(start)
        edge_lists.append(edge_list)
    return edge_lists,node_set



def update_dict(dict_,key,val):
    if key not in dict_:
        dict_[key] = [val]
    else:
        dict_[key].append(val)
    return dict_






def evaluate_graph_metric(gt_adj: np.ndarray, pred_adj: np.ndarray):
    gt_graph_statistics = compute_graph_statistics(gt_adj)
    pred_graph_statistics = compute_graph_statistics(pred_adj)
    # mmd_results = {}
    mmd_beta_map = {
        "d_mean": 1,
        "LCC": 1,
        "wedge_count": 1,
        "power_law_exp": 1,
        "triangle_count": 1,
        "rel_edge_distr_entropy": 1,
        "n_components": 1,
        "closeness_centrality_mean": 1,
        "closeness_centrality_median": 1,
        "betweenness_centrality_mean": 1,
        "betweenness_centrality_median": 1,
        "clustering_coefficient": 1,
        "degree": 1,
    }
    # for metric in gt_graph_statistics.keys():
    #     mmd_results[metric] = calculate_mmd(gt_graph_statistics[metric], pred_graph_statistics[metric], mmd_beta_map[metric])
    


    abs_results = {}
    for metric in gt_graph_statistics.keys():
        abs_results[metric] = np.abs(gt_graph_statistics[metric] - pred_graph_statistics[metric])
    
    return abs_results


if __name__ == "__main__":
    gt_adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    pred_adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mmd_results = eval_graph_metric(gt_adj, pred_adj)
    print(mmd_results)