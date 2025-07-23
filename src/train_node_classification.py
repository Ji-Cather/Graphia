# 从 node embedding中提取出 degree, query semantic id

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
import matplotlib.pyplot as plt
from scipy import stats

from .models.TGAT import TGAT
from .models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from .models.CAWN import CAWN
from .models.TCL import TCL
from .models.GraphMixer import GraphMixer
from .models.DyGFormer import DyGFormer
# from .models.degree import DegreePredictor
from .models.degree_bwr import HistogramMatchingLoss, CombinedDegreeLoss, DegreePredictor
from .models.modules import MergeLayer, MLPClassifier
from .utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from .utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from .utils.metrics import get_link_prediction_metrics
from .utils.DataLoader import get_idx_data_loader, get_node_classification_data
from .utils.EarlyStopping import EarlyStopping
from .utils.load_configs import get_node_classification_args
from .evaluate_models_utils import evaluate_model_link_prediction, evaluate_model_node_classification


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args(is_evaluation=False)
    #args.device = 'cpu'

    node_raw_features, edge_raw_features, full_data, train_data, \
    train_node_seq_data, val_node_seq_data, test_node_seq_data, max_degree = \
        get_node_classification_data(data_name=args.data_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, args = args)
    
    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    
    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_node_seq_data))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_node_seq_data))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_node_seq_data))), batch_size=args.batch_size, shuffle=False)
    
    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []
    
    
    train_dgnn = args.train_dgnn

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}{args.use_feature}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs_nc/{args.model_name}/{args.data_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs_nc/{args.model_name}/{args.data_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        logger.info(f'node feature size {node_raw_features.shape}')
        logger.info(f'edge feature size {edge_raw_features.shape}')
        logger.info(f'node feature example {node_raw_features[1][:5]}')
        logger.info(f'edge feature example {edge_raw_features[1][:5]}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        degree_list = train_node_seq_data.src_node_degrees.flatten()
        num_quantiles = 20
        
        if args.train_dgnn:
            # tbd, 需要考虑node feature的维度 测试各个node seq之间的相关性
            degree_classifier = DegreePredictor(seq_len = train_node_seq_data.seq_len, 
                                            max_degree = max_degree, 
                                            condition_dim = node_raw_features.shape[1],
                                            degree_list = degree_list,
                                            num_quantiles = num_quantiles)
        else:
            if args.use_feature == 'no':
                degree_classifier = DegreePredictor(seq_len = train_node_seq_data.seq_len, 
                                            max_degree = max_degree, 
                                            # degree_list = degree_list,
                                            condition_dim = 0,
                                            log2_quantile = True,
                                            num_quantiles = num_quantiles)
            else:
                degree_classifier = DegreePredictor(seq_len = train_node_seq_data.seq_len, 
                                            max_degree = max_degree, 
                                            # degree_list = degree_list,
                                            condition_dim = node_raw_features.shape[1],
                                            log2_quantile = True,
                                            num_quantiles = num_quantiles)
        
        model = nn.Sequential(dynamic_backbone, degree_classifier)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models_nc/{args.model_name}/{args.data_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # 创建组合损失函数
        # combined_loss = CombinedDegreeLoss(degree_weight=0.5, quantile_weight=0.3, histogram_weight=0.2, use_wasserstein=True)
        combined_loss = CombinedDegreeLoss(degree_weight=0.8, quantile_weight=0.2, histogram_weight=0.0)
                
        src_node_ids = np.arange(train_node_seq_data.src_node_degrees.shape[0])

        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, batch_train_src_node_ids in enumerate(train_idx_data_loader_tqdm):
                batch_train_src_node_ids = batch_train_src_node_ids.numpy()
                batch_src_node_degrees_train, batch_src_node_interact_times_train = train_node_seq_data[batch_train_src_node_ids]
                
                # not used 
                # batch_train_src_mask = batch_src_node_degrees_train[:,-1] != 0
                # batch_train_src_node_ids = batch_train_src_node_ids[batch_train_src_mask][:,0]
                
                # 使用因果推断方式训练
                batch_loss = 0
                seq_len = batch_src_node_degrees_train.shape[1]
                
                for t in range(1, seq_len):
                    # 使用前t个时间步预测第t+1个时间步
                    input_seq = torch.tensor(batch_src_node_degrees_train[:, :t].reshape(-1,t,1), device=args.device, dtype=torch.float32)
                    target_degree = torch.tensor(batch_src_node_degrees_train[:, t].reshape(-1,1,1), device=args.device, dtype=torch.float32)
                    input_times = batch_src_node_interact_times_train[:, :t]
                    
                    if train_dgnn:
                        # 计算节点嵌入
                        # tbd, 应该关注的是degree的相关性，而不是edge的相关性 应该类似degree corr
                        batch_src_node_embeddings_single = model[0].compute_src_node_temporal_embeddings(batch_train_src_node_ids, input_times[:,-1])
                        pass
                    else:
                        # 计算原始degree的quantile表示
                        if args.use_feature == 'no':
                            node_feature = None
                        else:
                            node_feature = torch.tensor(node_raw_features[batch_train_src_node_ids], device=args.device, 
                                                        dtype=torch.float32)
                        pred_degree, level_logits, rate = model[1](x_degree=input_seq, 
                                                                           x_condition=node_feature,
                                                                           return_quantile=True)
                    
                    # 计算原始degree的quantile表示
                    target_quantile = model[1].degree_converter.transform(target_degree) # N*seq_len*2
                    # 使用组合损失函数
                    loss, degree_loss, quantile_loss, histogram_loss, rate_loss = combined_loss(
                        target_degree, target_quantile, pred_degree, level_logits, rate
                    )
                    
                    batch_loss += loss
                
                # 平均每个时间步的损失
                if seq_len > 1:
                    loss = batch_loss/ (seq_len - 1)
                else:
                    loss = batch_loss
                
                train_losses.append(loss.item())
                train_metrics.append({
                    'degree_loss': degree_loss.item(),
                    'quantile_loss': quantile_loss.item(),
                    'histogram_loss': histogram_loss.item(),
                    'total_loss': loss.item()
                })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            if (epoch + 1) % args.test_interval_epochs == 0:
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after training so it can be used for new validation nodes
                    train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                val_losses, val_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_data=val_node_seq_data,
                                                                        loss_func=combined_loss,
                                                                        node_raw_features=node_raw_features,
                                                                        use_feature=args.use_feature,
                                                                        train_dgnn=train_dgnn)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name in val_metrics[0].keys():
                    logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
                
                # 打印损失计算统计
                loss_stats = combined_loss.get_loss_stats()
                logger.info("\n训练阶段损失计算统计:")
                logger.info(f"  - MSE损失: 计算次数={loss_stats['degree_loss_count']}, 总时间={loss_stats['degree_loss_time']:.4f}秒, 平均时间={loss_stats['avg_degree_loss_time']:.6f}秒")
                logger.info(f"  - Quantile损失: 计算次数={loss_stats['quantile_loss_count']}, 总时间={loss_stats['quantile_loss_time']:.4f}秒, 平均时间={loss_stats['avg_quantile_loss_time']:.6f}秒")
                logger.info(f"  - 直方图匹配损失: 计算次数={loss_stats['histogram_loss_count']}, 总时间={loss_stats['histogram_loss_time']:.4f}秒, 平均时间={loss_stats['avg_histogram_loss_time']:.6f}秒")
                
                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=test_idx_data_loader,
                                                                             evaluate_data=test_node_seq_data,
                                                                             loss_func=combined_loss,
                                                                              node_raw_features=node_raw_features,
                                                                                use_feature=args.use_feature,
                                                                             train_dgnn=train_dgnn)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.data_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_data=val_node_seq_data,
                                                                        loss_func=combined_loss,
                                                                         node_raw_features=node_raw_features,
                                                                        use_feature=args.use_feature,
                                                                        train_dgnn=train_dgnn)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=test_idx_data_loader,
                                                                        evaluate_data=test_node_seq_data,
                                                                        loss_func=combined_loss,
                                                                        node_raw_features=node_raw_features,
                                                                        use_feature=args.use_feature,
                                                                        train_dgnn=train_dgnn)


        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric


        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.data_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    
    sys.exit()
