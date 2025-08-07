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
from torch.utils.data import Dataset, DataLoader

from .models.informer.model import Informer, InformerTranverse, InformerDecoder
from .utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from .utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from .evaluate_models_utils import evaluate_model_node_regression_v2
from .utils.DataLoader import get_ctdg_generation_data
from .utils.bwr_ctdg import BWRCTDGDataset, custom_collate
from .utils.EarlyStopping import EarlyStopping
from .utils.load_configs import get_node_regression_args

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
   
    train_data_ctdg, val_data_ctdg, test_data_ctdg = \
        get_ctdg_generation_data(args = args)
    
    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    
    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}_bwr{args.bwr}_qm{args.quantile_mapping}_uf{args.use_feature}_cm{args.cm_order}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs_deg/{args.save_model_name}/{args.data_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs_deg/{args.save_model_name}/{args.data_name}/{args.save_model_name}/{str(time.time())}.log")
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

        # degree_regressor = InformerTranverse(
        #                         bwr = train_data_ctdg.bwr,
        #                         node_feature_dim = train_data_ctdg.node_feature.shape[1]+ \
        #                         2*train_data_ctdg.input_len,
        #                         input_len = train_data_ctdg.input_len,
        #                         src_node_out_feature_dim = 2*train_data_ctdg.pred_len,
        #                         factor=4,
        #                         d_model=64,
        #                         n_heads=4,
        #                         e_layers=2,
        #                         d_layers=2,
        #                         d_ff=64*4,
        #                         dropout=0.05,
        #                         attn='prob',
        #                         freq='d',
        #                         output_attention=False
        #                         ).to(args.device)
        degree_regressor = InformerDecoder(
                            bwr = train_data_ctdg.bwr,
                            node_feature_dim = train_data_ctdg.node_feature.shape[1]+ \
                            2*train_data_ctdg.input_len,
                            input_len = train_data_ctdg.input_len,
                            pred_len= train_data_ctdg.pred_len,
                            src_node_out_feature_dim = 2,
                            factor=4,
                            d_model=64,
                            n_heads=4, 
                            e_layers=2,
                            d_layers=2,
                            d_ff=64*4,
                            dropout=0.05,
                            attn='prob',
                            output_attention=False,
                            T = 'm',
                            freq = 'd',
                            )
        
        
        
        
        model = degree_regressor
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models_deg/{args.save_model_name}/{args.data_name}"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        save_result_folder = f"./saved_results_deg/{args.save_model_name}/{args.data_name}"
        os.makedirs(save_result_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # 创建组合损失函数
        # combined_loss = CombinedDegreeLoss(degree_weight=0.5, quantile_weight=0.3, histogram_weight=0.2, use_wasserstein=True)
        criterion = nn.L1Loss()  # MAE
        # criterion = ExponentialLoss()
        train_data_ctdg_loader = DataLoader(train_data_ctdg, 
                                            batch_size=args.batch_size, 
                                            shuffle=False,
                                            collate_fn=custom_collate
                                            )
        pred_len = train_data_ctdg.pred_len
        
        for epoch in range(args.num_epochs):

            model.train()
            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_ctdg_data_loader_tqdm = tqdm(train_data_ctdg_loader, ncols=120)
            train_pred_degrees_all = []
            train_target_degrees_all = []
            train_pred_unique_degrees_all = []
            train_target_unique_degrees_all = []
            for batch_idx, batch_data in enumerate(train_ctdg_data_loader_tqdm):
                # p(degree(node), degree_unique(node)) | past degree(node), past degree_unique(node), past times, pred time)

                batch_src_node_feature = batch_data["src_node_feature"]
                batch_src_node_interact_times = batch_data["src_interact_times"]
                batch_src_node_pred_time = batch_data["src_pred_time"]
                batch_src_node_degree = batch_data["src_node_degree"]
                batch_src_node_unique_degree = batch_data["src_node_unique_degree"]
                # convert to tensors

                batch_src_node_feature = torch.tensor(batch_src_node_feature, dtype=torch.float32).to(args.device)
                batch_src_node_interact_times = torch.tensor(batch_src_node_interact_times, dtype=torch.float32).to(args.device)
                batch_src_node_pred_time = torch.tensor(batch_src_node_pred_time, dtype=torch.float32).to(args.device)
                batch_src_node_degree = torch.tensor(batch_src_node_degree, dtype=torch.float32).to(args.device)
                batch_src_node_unique_degree = torch.tensor(batch_src_node_unique_degree, dtype=torch.float32).to(args.device)
              
              
                pred_output = model.forward_encoder(batch_src_node_feature, 
                                                    batch_src_node_interact_times, 
                                                    batch_src_node_pred_time)
                
                
                loss_degree = criterion(pred_output[:,:,:pred_len], torch.log1p(batch_src_node_degree))
                loss_unique = criterion(pred_output[:,:,pred_len:], torch.log1p(batch_src_node_unique_degree))
                loss = loss_degree + loss_unique
                
                train_losses.append(loss.item())
                train_metrics.append({
                    'total_loss': loss.item()
                })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_ctdg_data_loader_tqdm.set_description(f'Epoch: {epoch + 1},  train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                train_pred_degrees_all.append(pred_output[:,:,:pred_len])
                train_target_degrees_all.append(torch.log1p(batch_src_node_degree))
                train_pred_unique_degrees_all.append(pred_output[:,:,pred_len:])
                train_target_unique_degrees_all.append(torch.log1p(batch_src_node_unique_degree))
            
            train_pred_degrees_all = torch.cat(train_pred_degrees_all, dim=0).flatten()
            train_target_degrees_all = torch.cat(train_target_degrees_all, dim=0).flatten()
            train_pred_unique_degrees_all = torch.cat(train_pred_unique_degrees_all, dim=0).flatten()
            train_target_unique_degrees_all = torch.cat(train_target_unique_degrees_all, dim=0).flatten()
            batch_loss_degree = criterion(train_pred_degrees_all, train_target_degrees_all)
            batch_loss_unique = criterion(train_pred_unique_degrees_all, train_target_unique_degrees_all)
            batch_loss = batch_loss_degree + batch_loss_unique
            logger.info(f'Epoch: {epoch + 1}, train loss: {batch_loss.item()}, degree loss: {batch_loss_degree.item()}, unique loss: {batch_loss_unique.item()}')
            

            eval_losses, eval_metrics, eval_metrics_all = evaluate_model_node_regression_v2(model_name=args.model_name,
                                                                        model=model,
                                                                        evaluate_data=val_data_ctdg,
                                                                        loss_func=criterion,
                                                                        mode = 'val',
                                                                        quantile_mapping=args.quantile_mapping,
                                                                        args=args)


            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses)}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics])}')
            logger.info(f'validate loss: {np.mean(eval_losses)}')
            for metric_name in eval_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([eval_metric[metric_name] for eval_metric in eval_metrics])}')
            for metric_name in eval_metrics_all.keys():
                logger.info(f'validate {metric_name}, {eval_metrics_all[metric_name]}')
            
           
            # select the best model based on all the validate metrics
            val_metric_indicator = []
            val_metric_indicator.append(
                ('average_metrics', np.mean(list(eval_metrics_all.values())), False)
            )
            early_stop = early_stopping.step(val_metric_indicator, model, args)

            if early_stop:
                break

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                degree_save_path = os.path.join(save_result_folder, f'train_degree.pt')
                test_losses, test_metrics, test_metrics_all = evaluate_model_node_regression_v2(model_name=args.model_name,
                                                                          model=model,
                                                                          evaluate_data=train_data_ctdg,
                                                                          mode = 'train',
                                                                          loss_func=criterion,
                                                                          quantile_mapping=args.quantile_mapping,
                                                                          args = args,
                                                                          save_degree_path=degree_save_path)
                
                degree_save_path = os.path.join(save_result_folder, f'val_degree.pt')
                test_losses, test_metrics, test_metrics_all = evaluate_model_node_regression_v2(model_name=args.model_name,
                                                                          model=model,
                                                                          evaluate_data=train_data_ctdg,
                                                                          mode = 'val',
                                                                          loss_func=criterion,
                                                                          quantile_mapping=args.quantile_mapping,
                                                                          args = args,
                                                                          save_degree_path=degree_save_path)

                degree_save_path = os.path.join(save_result_folder,f'test_degree.pt')
                test_losses, test_metrics, test_metrics_all = evaluate_model_node_regression_v2(model_name=args.model_name,
                                                                          model=model,
                                                                          evaluate_data=test_data_ctdg,
                                                                          mode = 'test',
                                                                          loss_func=criterion,
                                                                          quantile_mapping=args.quantile_mapping,
                                                                          args = args,
                                                                          save_degree_path=degree_save_path)
                
                
                
                

                logger.info(f'test loss: {np.mean(test_losses)}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics])}')
                for metric_name in test_metrics_all.keys():
                    logger.info(f'test {metric_name}, {test_metrics_all[metric_name]}')
        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.data_name}...')


        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'test loss: {np.mean(test_losses)}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric}')
            test_metric_dict[metric_name] = average_test_metric
            
        for metric_name in test_metrics_all.keys():
            logger.info(f'test {metric_name}, {test_metrics_all[metric_name]}')
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
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]}' for metric_name in test_metric_dict},
            }
        result_json = json.dumps(result_json, indent=4)

        
        save_result_path = os.path.join(save_result_folder, f"metrics.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')
    logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
    logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs])} '
                f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1)}')

    
    sys.exit()
