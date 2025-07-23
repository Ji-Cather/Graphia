8days_dytag_small_text_en

zero_shot: filter_rule+neighbor_sampling
评估指标: {'precision_node': 0.1123346492984177, 'f1_node': 0.11813722738443418, 'ndcg@10_node': 0.5149928109833547, 'auc_node': 0.5708388332859601, 'JL-Metric': 0.1410193294286728, 'LCC': 177, 'wedge_count': 76711.0, 'power_law_exp': 1.3922319969900396, 'triangle_count': 1430, 'rel_edge_distr_entropy': 0.4673169401361872, 'n_components': 84, 'closeness_centrality_mean': 0.009797224935405479, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.03191299530688782}

query_with_gt: given ground truth node text: select dst id+neighbor_sampling
评估指标: {'precision_node': 0.17779830497637839, 'f1_node': 0.18849723668951435, 'ndcg@10_node': 0.6336234912321297, 'auc_node': 0.6102870017433619, 'JL-Metric': 0.1559937596321106, 'LCC': 425, 'wedge_count': 76698.0, 'power_law_exp': 1.0083418530230723, 'triangle_count': 406, 'rel_edge_distr_entropy': 0.46304421721456646, 'n_components': 157, 'closeness_centrality_mean': 0.0174605136666228, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.014560578289652203}


query_with_gt + filter : given ground truth node text: select dst id+filter_rule+neighbor_sampling
评估指标: {'precision_node': 0.18354692588741287, 'f1_node': 0.1945019598767179, 'ndcg@10_node': 0.5714600535486238, 'auc_node': 0.6136068660453315, 'JL-Metric': 0.19143907725811005, 'LCC': 324, 'wedge_count': 76630.0, 'power_law_exp': 1.131498466706792, 'triangle_count': 656, 'rel_edge_distr_entropy': 0.46163382219157356, 'n_components': 68, 'closeness_centrality_mean': 0.015110777717049686, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.017801738427896018}

query using gt query text
评估指标: {'precision_node': 0.2435038286835921, 'f1_node': 0.2553927184416545, 'ndcg@10_node': 0.6378091628564152, 'auc_node': 0.6447698623483806, 'JL-Metric': 0.15369100868701935, 'LCC': 382, 'wedge_count': 72467.0, 'power_law_exp': 0.9939210519948691, 'triangle_count': 601, 'rel_edge_distr_entropy': 0.44025235329213097, 'n_components': 108, 'closeness_centrality_mean': 0.016828908886951186, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.011764491010872296}

query_with_gt, reward_with_llm: given ground truth node text: select dst id+neighbor_sampling+reward
评估指标: {'precision_node': 0.18037284096202014, 'f1_node': 0.18644624174885133, 'ndcg@10_node': 0.5113899716840956, 'auc_node': 0.6030030486546014, 'JL-Metric': 0.17944739758968353, 'LCC': 444, 'wedge_count': 72990.0, 'power_law_exp': 1.224437300348356, 'triangle_count': 111, 'rel_edge_distr_entropy': 0.5300189629709556, 'n_components': 113, 'closeness_centrality_mean': 0.022941696331536355, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.00729809818995838}

query_with_gt, reward_with_overlap_len: given ground truth node text: select dst id+neighbor_sampling+reward(overlap len)
评估指标: {'precision_node': 0.1848212237098984, 'f1_node': 0.19125745010104403, 'ndcg@10_node': 0.4840262431513587, 'auc_node': 0.6058425585151536, 'JL-Metric': 0.1900186985731125, 'LCC': 454, 'wedge_count': 73502.0, 'power_law_exp': 1.1423594374944486, 'triangle_count': 111, 'rel_edge_distr_entropy': 0.530741467724436, 'n_components': 136, 'closeness_centrality_mean': 0.023064171021764692, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.007451019634353112}


7B
sft 7B + filter rule 
评估指标: {'precision_node': 0.09273624510678219, 'f1_node': 0.10070602553295731, 'auc_node': 0.563077980482845, 'ndcg@10_node': 0.40304795895960216, 'precision_hub': 1.0, 'f1_hub': 1.0, 'auc_hub': 1.0, 'ndcg@1126_hub': 1.0, 'edge_overlap': 0.15536301165222588, 'JL-Metric': 0.17288485169410706, 'LCC': 295, 'wedge_count': 63363.0, 'power_law_exp': 1.2269283379050804, 'triangle_count': 637, 'rel_edge_distr_entropy': 0.5053507273507643, 'n_components': 2, 'closeness_centrality_mean': 0.018783956078000725, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.004445542633618384}


这里的问题在于，对于结果是aggregation，要是要和rl的进行比较需要用drop（keep first）
5query aggregation result+ sft 7B + filter rule 
评估指标: {'precision_node': 0.15815830563175431, 'f1_node': 0.16675401987847285, 'auc_node': 0.5990648865895501, 'ndcg@10_node': 0.5381055667515886, 'precision_hub': 1.0, 'f1_hub': 1.0, 'auc_hub': 1.0, 'ndcg@1126_hub': 1.0, 'edge_overlap': 0.24260531819539888, 'JL-Metric': 0.18404048681259155, 'LCC': 321, 'wedge_count': 68075.0, 'power_law_exp': 0.9699550744871073, 'triangle_count': 379, 'rel_edge_distr_entropy': 0.4107910625488568, 'n_components': 81, 'closeness_centrality_mean': 0.015241677391269626, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.006890107149005624}

5query argmax(result)
评估指标: {'precision_node': 0.11810911420379523, 'f1_node': 0.12668364279550948, 'auc_node': 0.5776972306761018, 'ndcg@10_node': 0.44408820726345527, 'precision_hub': 1.0, 'f1_hub': 1.0, 'auc_hub': 1.0, 'ndcg@1126_hub': 1.0, 'edge_overlap': 0.16880788766059157, 'JL-Metric': 0.16707243025302887, 'LCC': 342, 'wedge_count': 72131.0, 'power_law_exp': 1.202383088886675, 'triangle_count': 579, 'rel_edge_distr_entropy': 0.5099941929492614, 'n_components': 52, 'closeness_centrality_mean': 0.01971434289451882, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.010394417051251789}

1query 7B + filter rule 
评估指标: {'precision_node': 0.04449119729079635, 'f1_node': 0.04667585361183021, 'auc_node': 0.5274257575711542, 'ndcg@10_node': 0.42682648030463766, 'precision_hub': 1.0, 'f1_hub': 1.0, 'auc_hub': 1.0, 'ndcg@1126_hub': 1.0, 'edge_overlap': 0.0917239318792949, 'JL-Metric': 0.18638597428798676, 'LCC': 136, 'wedge_count': 65887.0, 'power_law_exp': 1.3799048846283997, 'triangle_count': 1276, 'rel_edge_distr_entropy': 0.441836070298139, 'n_components': 107, 'closeness_centrality_mean': 0.007931885655331993, 'closeness_centrality_median': 0.0, 'clustering_coefficient': 0.009270779353442955}