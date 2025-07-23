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


from .models.informer.model import Informer, InformerTranverse
from .utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from .utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from .evaluate_models_utils import evaluate_model_graph_generation
from .utils.DataLoader import get_idx_data_loader, get_node_classification_data
from .utils.EarlyStopping import EarlyStopping
from .utils.load_configs import get_node_regression_args

from .models.graphgenerator import GraphGenerator

class ExponentialLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, y_pred, y_true):
        weights_threshold = 1.01
        error = torch.abs(y_true - y_pred)
        weights = torch.exp((y_true - 1e-10))  # 目标值越大，权重指数增长
        # weights = torch.where(weights < weights_threshold, 0.1, weights)
        loss = torch.mean(weights * error)
        return loss
  


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_regression_args(is_evaluation=False)
    #args.device = 'cpu'

    node_raw_features, edge_raw_features, full_data, train_data, \
    train_node_seq_data, val_node_seq_data, test_node_seq_data, max_degree,\
    train_snapshots, val_snapshots, test_snapshots = \
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

        
        
        
        args.quantile_mapping = True
        if args.model_name == 'GraphGenerator':
            graph_generator = GraphGenerator(bwr = args.batch_size,
                                        src_node_feature_dim = train_node_seq_data.src_node_feature_dim,
                                        dst_node_feature_dim = train_node_seq_data.dst_node_feature_dim,
                                        edge_feature_dim = edge_raw_features.shape[1],
                                        max_node_num = node_raw_features.shape[0],
                                        input_len = train_node_seq_data.input_len,
                                        src_node_out_feature_dim = train_node_seq_data.src_node_y_dim,
                                        device = args.device,
                                        node_raw_features = node_raw_features,
                                        edge_raw_features = edge_raw_features
                                        )
        else:
            raise ValueError(f'model name {args.model_name} not found')
    
        
        model = graph_generator
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
        criterion = nn.L1Loss()  # MAE
        # criterion = ExponentialLoss()

        for epoch in range(args.num_epochs):

            model.train()
            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            epoch_losses = []
            for batch_idx, node_seq_idx in enumerate(train_idx_data_loader_tqdm):
                batch_src_node_feature, batch_src_node_interact_times, \
                batch_src_node_pred_feature, batch_src_node_pred_degree, \
                batch_src_node_pred_interact_times, batch_src_node_ids,\
                input_edges, target_edges = train_node_seq_data[node_seq_idx]
                
                batch_src_node_feature = torch.tensor(batch_src_node_feature, dtype=torch.float32).to(args.device)
                batch_src_node_interact_times = torch.tensor(batch_src_node_interact_times, dtype=torch.float32).to(args.device)
                batch_src_node_pred_feature = torch.tensor(batch_src_node_pred_feature, dtype=torch.float32).to(args.device)
                batch_src_node_pred_interact_times = torch.tensor(batch_src_node_pred_interact_times, dtype=torch.float32).to(args.device)
                target_src_node_degrees = torch.tensor(batch_src_node_pred_degree, dtype=torch.float32).to(args.device)
            
                
                pred_loss = model( 
                                batch_src_node_feature, batch_src_node_interact_times,
                                batch_src_node_pred_feature, batch_src_node_pred_interact_times,
                                input_edges, 
                                target_src_node_degrees, 
                                target_edges)
                total_loss = pred_loss['src_loss'] + pred_loss['dst_loss']
                
                train_losses.append(total_loss.item())
                train_metrics.append({
                    'total_loss': total_loss.item()
                })

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {total_loss.item()}')

            logger.info(f'Epoch: {epoch + 1}, train loss: {np.mean(train_losses):.4f}')
            

            eval_losses, eval_metrics, eval_metrics_all = evaluate_model_graph_generation(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_data=val_node_seq_data,
                                                                        loss_func=criterion,
                                                                        mode = 'val',
                                                                        quantile_mapping=args.quantile_mapping)


            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(eval_losses):.4f}')
            for metric_name in eval_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([eval_metric[metric_name] for eval_metric in eval_metrics]):.4f}')
            for metric_name in eval_metrics_all.keys():
                logger.info(f'validate {metric_name}, {eval_metrics_all[metric_name]:.4f}')
            
           
            # select the best model based on all the validate metrics
            val_metric_indicator = []
            val_metric_indicator.append(
                ('average_metrics', np.mean(list(eval_metrics_all.values())), False)
            )
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics, test_metrics_all = evaluate_model_graph_generation(model_name=args.model_name,
                                                                          model=model,
                                                                          neighbor_sampler=full_neighbor_sampler,
                                                                          evaluate_idx_data_loader=test_idx_data_loader,
                                                                          evaluate_data=test_node_seq_data,
                                                                          mode = 'test',
                                                                          loss_func=criterion,
                                                                          quantile_mapping=args.quantile_mapping)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                for metric_name in test_metrics_all.keys():
                    logger.info(f'test {metric_name}, {test_metrics_all[metric_name]:.4f}')
        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.data_name}...')


        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric
            
        for metric_name in test_metrics_all.keys():
            logger.info(f'test {metric_name}, {test_metrics_all[metric_name]:.4f}')
            test_metric_dict[metric_name] = test_metrics_all[metric_name]

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
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
    logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
    logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    
    sys.exit()
