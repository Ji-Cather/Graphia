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

logger = get_logger()  # 获取日志记录器实例

from .utils_dst import *


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    """
    # 如果 max_penalty 是正的，这里直接抛出错误，说明要用负值来做惩罚
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    # 内部函数 zipngram，用于切分文本为 ngram
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    # repetition_penalty_reward 函数用于计算在给定 response 中，n-gram 的重复程度
    def repetition_penalty_reward(response, **kwargs) -> float:
        """
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
        """
        # 如果回复为空或不足 ngram 大小，则直接返回 0
        if response == "" or len(response.split()) < ngram_size:
            return 0.0

        # 遍历所有 ngram，统计 unique ngram 和 total ngram 的数量
        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1

        # scaling = 1 - (不重复的 ngram / 总的 ngram 数量)
        # 不重复的越少（重复越多）scaling 越大
        scaling = 1 - len(ngrams) / total
        # reward 是 scaling 乘以 max_penalty
        reward = scaling * max_penalty
        return reward

    return repetition_penalty_reward





class DstSEQRewardWorker(Worker):
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
        self.mapper = DstMapper(worker_config.environment_data_args,
                                worker_config.reward_args
                                )
        self.reward_type = worker_config.reward_type
        # assert self.reward_type in ["overlap_reward",
        #                              "LIKR_reward"], "unsupported reward type"
        
        self.bert_args = worker_config.bert_args
        


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, 
                    pipeline_config):
        # gpus = self.get_visible_gpus()
        # gpu_index = gpus[0]
        device = "cpu"
        self.bert_embedder = BertEmbedder(self.bert_args,
                                          logger=self.logger)
        


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


        prompts = data.non_tensor_batch["prompt"]
        src_idxs = data.non_tensor_batch["src_id"]
        dst_idxs = data.non_tensor_batch["dst_id"]

        response_rewards:List[dict] = [] 
        for i, (resp_tokens, prompt,
                    src_idx, dst_idx) in enumerate(
            zip(data.batch["responses"], 
                    prompts,
                    src_idxs, dst_idxs)
        ):
            
            self.logger.info(f"debug 1: src_idx {src_idx}")
            resp_text_without_sptoken = self.tokenizer.decode(resp_tokens, skip_special_tokens=True)
            # debug 
            # self.logger.info(f"prompt, {prompt}")
            self.logger.info(f"resp_text_without_sptoken, {resp_text_without_sptoken}")

            parsed_resp, reward_dict = self.mapper.parse_response(
                resp_text_without_sptoken, 
                                                    self.logger) # dict
          

            parsed_resp.update({
                "src_idx": int(src_idx),
                "gt_dst_idx": int(dst_idx)
            })
            
            reward_dict_acc, parsed_resp = self.mapper._collect_seq_reward(
                parsed_resp, 
                self.bert_embedder, 
                self.logger)
                
            reward_dict.update(reward_dict_acc)
            self.logger.info(f"debug 2: src_idx {src_idx}", reward_dict)
            
            response_rewards.append(reward_dict)

        # Convert scores to binary labels: 1 if score > 0, else 0
        scores = torch.tensor([_.get("score",0) for _ in response_rewards], dtype=torch.float16) 
       
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.tensor([_.get(self.reward_type,0) + _.get("format_reward",0) 
                                                for _ in response_rewards], dtype=torch.float16)
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


