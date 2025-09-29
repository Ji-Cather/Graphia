# 导入必要的库和模块
from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy

from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider

from typing import Union, Dict, List

from roll.utils.logging import get_logger

from .utils_edge import *


logger = get_logger()  # 获取日志记录器实例


from Graphia.load_gnn_judger import create_edge_classification_model
from Graphia.utils.utils import get_neighbor_sampler


class EdgeGNNRewardWorker(Worker):
    """
    一个示例 Reward Worker，用于执行 ifeval 验证并把每个 func 的结果放到 output.tensors 中。
    在此示例里，ground_truths的str
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.mapper = EdgeMapper(worker_config.environment_data_args,
                                worker_config.reward_args
                                )
        self.reward_type = worker_config.reward_type
        # assert self.reward_type in ["weight_reward",
        #                              "overlap_reward",
        #                              "gnn_reward"], "unsupported reward type"
        
        self.bert_args = worker_config.bert_args
        self.gnn_args = worker_config.gnn_args
        self.recall_query = worker_config.recall_query
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, 
                    pipeline_config):
        # gpus = self.get_visible_gpus()
        # gpu_index = gpus[0]
        device = "cpu"
        self.bert_embedder = BertEmbedder(self.bert_args,
                                          logger=self.logger)
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num = \
            self.mapper.get_link_prediction_data()
            
        full_neighbor_sampler = get_neighbor_sampler(
                data=full_data, 
                sample_neighbor_strategy=self.gnn_args.sample_neighbor_strategy,
                time_scaling_factor=self.gnn_args.time_scaling_factor, 
                seed=self.gnn_args.seed)
        
        self.gnn_judger = create_edge_classification_model(
            model_name=self.gnn_args.model_name,
            save_model_path=self.gnn_args.save_model_path,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            data=full_data,
            neighbor_sampler=full_neighbor_sampler,
            device = device,
            cat_num = cat_num
        )

        

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.load_states()
        else:
            self.logger.warning("worker has not strategy")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.offload_states()
        else:
            self.logger.warning("worker has not strategy")


    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_rewards(self, data: DataProto):

        global_step = data.meta_info.get("global_step", 0)

        prompts = data.non_tensor_batch["prompt"]
        src_idxs = data.non_tensor_batch["src_idx"]
        dst_idxs = data.non_tensor_batch["dst_idx"]
        gt_edge_labels = data.non_tensor_batch["gt_label"]
        gt_edge_texts = data.non_tensor_batch["output"]
        # dx_srcs = data.non_tensor_batch["gt_dx_src_unique"]
        # gt_dst_idxs = data.non_tensor_batch["gt_dst_idxs_unique"]
        tags = data.non_tensor_batch["tag"]

        response_rewards:List[dict] = [] 
        repetition_penalty_rewards = []
        for i, (resp_tokens, 
                prompt,
                gt_edge_text, 
                gt_edge_label,
                src_idx,
                dst_idx,
                tag) in enumerate(
            zip(data.batch["responses"], 
            prompts,
            gt_edge_texts, 
            gt_edge_labels,
            src_idxs,
            dst_idxs,
            tags)
        ):
            ori_resp_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=False)
            resp_text_without_sptoken = (
                ori_resp_text.replace("<|endoftext|>", "").replace("<|end▁of▁sentence|>","").replace("<｜end▁of▁sentence｜>","").replace("<pad>", "").replace("<|im_end|>", "")
            )
            resp_text_without_sptoken = resp_text_without_sptoken.strip(
                "<|end▁of▁sentence|>").strip("<pad>").strip("<｜end of sentence｜>").strip("<endoftext>")
            # answer_text = extract_after_last_think(resp_text_without_sptoken)
            resp_text_without_sptoken = resp_text_without_sptoken.replace("<edge>","").replace("</edge>","")
            
            parsed_resp, reward_dict = self.mapper.parse_response(resp_text_without_sptoken, 
                                                    self.logger) # dict

            parsed_resp.update({
                "src_idx": int(src_idx),
                "dst_idx": int(dst_idx),
                "gt_edge_text": gt_edge_text, 
                "gt_edge_label":int(gt_edge_label),
                "tag": tag,
            })
            try:
                reward_dict_acc, parsed_resp = self.mapper._collect_gnn_reward(
                    parsed_resp, 
                    self.bert_embedder, 
                    self.gnn_args.model_name,  
                    self.gnn_judger,
                    global_step,
                    self.logger,
                    self.recall_query)
                reward_dict.update(reward_dict_acc)
                self.logger.info(f"debug {src_idx}", reward_dict)
                
            except Exception as e:
                reward_dict = {
                    "format_reward": reward_dict["format_reward"],
                    "gt_reward": 0,
                    "gnn_reward": 0,
                    "llm_reward": 0,
                    "score":0
                    }
                self.logger.error("reward calculation", e)

            response_rewards.append(reward_dict)
            try:
                parsed_resp.update({
                    "tag": tag,
                    "prompt": prompt,
                    "src_idx": int(src_idx),
                    "ori_response": str(resp_text_without_sptoken),
                    **reward_dict
                })
                
                checkkeys = ["src_idx", *reward_dict.keys()]
                parsed_resp = {k: v for k, v in parsed_resp.items() if k in checkkeys}
                outputs = json.dumps(parsed_resp,
                    ensure_ascii=False,
                )
                self.logger.debug(outputs)
            except Exception as e:
                self.logger.error(f"answer check except: {e}")

        # Convert scores to binary labels: 1 if score > 0, else 0
        scores = torch.tensor([_.get("score",0) for _ in response_rewards], dtype=torch.float16) 
        score_labels = (scores> 0).float() # 存在overlap就认为是correct？
        
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.tensor([_.get(self.reward_type,0) for _ in response_rewards], dtype=torch.float16)
        # repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        # response_level_rewards = scores + repetition_penalty_rewards
        
        # 5) 将这些张量打包进同一个字典
        output_tensors = {
            "scores": scores, # float
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
        }

        # 6) 用 DataProto.from_dict(...) 构造返回值
        output = DataProto.from_dict(tensors=output_tensors)
        return output
