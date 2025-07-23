zero_shot: filter_rule+neighbor_sampling

评估指标: {'precision_node': 0.04300387314722485, 'f1_node': 0.04511454836910088, 'ndcg@10_node': 0.34468309061576147, 'auc_node': 0.5262047061426172, 'JL-Metric': 0.28011393547058105, 'LCC': 141, 'wedge_count': 74184.0, 'power_law_exp': 1.5410788099820394, 'triangle_count': 1388, 'rel_edge_distr_entropy': 0.4742211792824489, 'n_components': 158, 'closeness_centrality_mean': 0.0085695885995067, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.02582583933493052}

zero_shot: select dst id+neighbor_sampling
评估指标: {'precision_node': 0.0441957986248837, 'f1_node': 0.046289095384147105, 'ndcg@10_node': 0.35189685815986044, 'auc_node': 0.5267805453022577, 'JL-Metric': 0.2706511318683624, 'LCC': 135, 'wedge_count': 71747.0, 'power_law_exp': 1.5266698273752732, 'triangle_count': 1351, 'rel_edge_distr_entropy': 0.46890976616740143, 'n_components': 164, 'closeness_centrality_mean': 0.008595979446473733, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.020591745094200337}

given ground truth node text: select dst id+neighbor_sampling
评估指标: {'precision_node': 0.17779830497637839, 'f1_node': 0.18849723668951435, 'ndcg@10_node': 0.6336234912321297, 'auc_node': 0.6102870017433619, 'JL-Metric': 0.24322006106376648, 'LCC': 425, 'wedge_count': 76698.0, 'power_law_exp': 1.0083418530230723, 'triangle_count': 406, 'rel_edge_distr_entropy': 0.46304421721456646, 'n_components': 157, 'closeness_centrality_mean': 0.0174605136666228, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.014560578289652203}


given ground truth node text: select dst id+filter_rule+neighbor_sampling
评估指标: {'precision_node': 0.18354692588741287, 'f1_node': 0.1945019598767179, 'ndcg@10_node': 0.5714600535486238, 'auc_node': 0.6136068660453315, 'JL-Metric': 0.2364063709974289, 'LCC': 324, 'wedge_count': 76630.0, 'power_law_exp': 1.131498466706792, 'triangle_count': 656, 'rel_edge_distr_entropy': 0.46163382219157356, 'n_components': 68, 'closeness_centrality_mean': 0.015110777717049686, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.017801738427896018}
