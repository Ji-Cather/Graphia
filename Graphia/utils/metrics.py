import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
# from .utils.graph_structure_evaluation import Degree

def get_rank(target_score, candidate_score):
    tmp_list = target_score - candidate_score
    rank = len(tmp_list[tmp_list < 0]) + 1
    return rank


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}

# def mmd_degree_metrics(predicts: torch.Tensor, labels: torch.Tensor):
#     # predicts,shape [num_graph,num_nodes]
#     descriptor = Degree(is_parallel = False)
#     metric = descriptor.evaluate_from_degree_array(predicts.int().cpu().detach().numpy(), 
#                                                    labels.int().cpu().detach().numpy())
#     return metric


def get_retrival_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """
    get metrics for the link prediction task
    :param pos_scores: Tensor, shape (num_samples, )
    :param neg_scores: Tensor, shape (neg_size, num_samples)
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    try:
        pos_scores = pos_scores.cpu().detach().numpy()
    except:
        pass
    try:
        neg_scores = np.array([sub_score.cpu().numpy() for sub_score in neg_scores]).T # num_samples * neg_size
    except:
        neg_scores = np.array([sub_score for sub_score in neg_scores]).T # num_samples * neg_size

    H1, H3, H10 = [], [], []
    for i in range(len(pos_scores)):
        rank = get_rank(pos_scores[i], neg_scores[i])
        if rank <= 1:
            H1.append(1)
        else:
            H1.append(0)
        
        if rank <= 3:
            H3.append(1)
        else:
            H3.append(0)

        if rank <= 10:
            H10.append(1)
        else:
            H10.append(0)

    return {'H1': np.mean(H1), 'H3': np.mean(H3), 'H10': np.mean(H10)}


# def get_retrival_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
#     """
#     get metrics for the link prediction task
#     :param pos_scores: Tensor, shape (num_samples, )
#     :param neg_scores: Tensor, shape (neg_size, num_samples)
#     :return:
#         dictionary of metrics {'metric_name_1': metric_1, ...}
#     """
#     try:
#         pos_scores = pos_scores.cpu().detach().numpy()
#     except:
#         pass
#     try:
#         neg_scores = np.array([sub_score.cpu().numpy() for sub_score in neg_scores]).T # num_samples * neg_size
#     except:
#         neg_scores = np.array([sub_score for sub_score in neg_scores]).T # num_samples * neg_size

#     H10, H50, H100 = [], [], []
#     for i in range(len(pos_scores)):
#         rank = get_rank(pos_scores[i], neg_scores[i])
#         if rank <= 10:
#             H10.append(1)
#         else:
#             H10.append(0)
        
#         if rank <= 3:
#             H50.append(1)
#         else:
#             H50.append(0)

#         if rank <= 10:
#             H100.append(1)
#         else:
#             H100.append(0)

#     return {'H10': np.mean(H10), 'H50': np.mean(H50), 'H100': np.mean(H100)}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the edge classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    acc = accuracy_score(labels, predicts)

    P_macro = precision_score(labels, predicts, average="macro")
    R_macro = recall_score(labels, predicts, average="macro")
    F_macro = f1_score(labels, predicts, average="macro")

    P_micro = precision_score(labels, predicts, average="micro")
    R_micro = recall_score(labels, predicts, average="micro")
    F_micro = f1_score(labels, predicts, average="micro")

    P_weight = precision_score(labels, predicts, average="weighted")
    R_weight = recall_score(labels, predicts, average="weighted")
    F_weight = f1_score(labels, predicts, average="weighted")

    return {
        'acc': acc,
        'p_macro': P_macro, 'R_macro': R_macro, 'F_macro': F_macro, 'p_micro': P_micro, 'R_micro': R_micro, 'F_micro': F_micro, 'p_weight': P_weight, 'R_weight': R_weight, 'F_weight': F_weight}


def calc_label_acc_precision_recall_f1_batch(predicts, labels):
    """
    批量计算函数，按照逐样本处理逻辑
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    
    sample_metrics = []
    
    for pred, gt in zip(predicts, labels):
        # 确保标签是列表格式
        if not isinstance(gt, (list, tuple)):
            gt = [gt]
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        
        # 计算accuracy
        acc = int(any(p in gt for p in pred)) if len(pred) > 0 else 0
        
        # 计算precision
        if len(pred) == 0:
            precision = 0.0
        else:
            correct_predictions = len(set(pred) & set(gt))
            precision = correct_predictions / len(pred)
        
        # 计算recall
        if len(gt) == 0:
            recall = 0.0
        else:
            correct_predictions = len(set(pred) & set(gt))
            recall = correct_predictions / len(gt)
        
        # 计算f1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        sample_metrics.append((acc, precision, recall, f1))
    
    # 提取所有样本的指标
    accs, precisions, recalls, f1s = zip(*sample_metrics)
    
    # 计算整体统计
    acc = np.mean(accs)
    P_macro = np.mean(precisions)
    R_macro = np.mean(recalls)
    F_macro = np.mean(f1s)
    
    # Micro平均计算
    total_correct = sum(len(set(pred if isinstance(pred, (list, tuple)) else [pred]) & 
                           set(gt if isinstance(gt, (list, tuple)) else [gt])) 
                       for pred, gt in zip(predicts, labels))
    total_pred = sum(len(pred if isinstance(pred, (list, tuple)) else [pred]) for pred in predicts)
    total_gt = sum(len(gt if isinstance(gt, (list, tuple)) else [gt]) for gt in labels)
    
    P_micro = total_correct / total_pred if total_pred > 0 else 0.0
    R_micro = total_correct / total_gt if total_gt > 0 else 0.0
    F_micro = 2 * (P_micro * R_micro) / (P_micro + R_micro) if (P_micro + R_micro) > 0 else 0.0
    
    # Weighted平均（这里与macro相同，因为是逐样本计算）
    P_weight = P_macro
    R_weight = R_macro
    F_weight = F_macro
    
    return {
        'acc': acc,
        'P_macro': P_macro, 'R_macro': R_macro, 'F_macro': F_macro,
        'P_micro': P_micro, 'R_micro': R_micro, 'F_micro': F_micro,
        'P_weight': P_weight, 'R_weight': R_weight, 'F_weight': F_weight
    }