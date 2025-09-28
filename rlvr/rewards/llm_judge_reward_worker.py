from typing import Optional, Union, Dict, List, Any
import json
import re
import torch
import requests
import time
import traceback
import numpy as np
from functools import partial
import tensordict
from tensordict import TensorDict
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_reward_model_provider
from roll.utils.logging import get_logger
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *
from roll.datasets.chat_template import get_chat_template
import os

from LLMGGen.utils.bwr_ctdg import (
                                    Dataset_Template)

def select_to_last_period(s, upper_token = 4e3):
    upper_token = int(upper_token)
    if len(s) <= upper_token:
        return s
    if upper_token <=0:
        return ""
    s = s[-upper_token:]
    # 查找最后一个句号的位置
    last_period_index = s.rfind('.')
    # 如果找到句号，返回从开始到最后一个句号之前的部分
    if last_period_index != -1:
        return s[:last_period_index]
    else:
        # 如果没有找到句号，返回整个字符串
        return s
    
class LLMJudgeRewardWorker(Worker):
    """
    Reward Worker that uses LLM-as-judge to compute rewards.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # LLM judge相关配置
        self.judge_prompt_name = self.worker_config.judge_prompt if hasattr(self.worker_config, "judge_prompt") else None
        self.judge_prompt = prompt_maps[self.judge_prompt_name]
        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "api"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None
        self.data_name = os.path.basename(self.worker_config.environment_data_args.root.strip("/"))


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            print(f"{self.worker_name} initialized with API model")

        elif self.judge_model_type == "inference":
            self.strategy = create_strategy(worker=self)
            self.strategy.initialize(model_provider=default_reward_model_provider)
            self.tokenizer = self.strategy.tokenizer
            print(f"{self.worker_name} initialized with inference model")
            self.strategy.offload_states()
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

    def _call_api_model(self, messages: Dict, retry_times=3) -> str:
        from openai import OpenAI

        ouput = ""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API model type")
        while retry_times > 0:
            retry_times -= 1
            try:
                client = OpenAI(
                    api_key=self.judge_api_key,
                    base_url=self.judge_api_url,
                )
                completion = client.chat.completions.create(model=self.judge_model_name, messages=messages)
                output = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens
                prompt_token = completion.usage.prompt_tokens
                completion_token = completion.usage.completion_tokens
                token_info = {
                    "total_tokens": total_tokens,
                    "prompt_token": prompt_token,
                    "completion_token": completion_token,
                }
                print(token_info)
                if output != None and output != "":
                    break
            except Exception as e:
                print(e)
                continue
        self.logger.info(f"judge model api output: {str(output)}")
        return output

    def _run_local_inference(self, messages: Dict) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")

        template_name = self.worker_config.data_args.template
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        text = chat_template_func(messages)

        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to("cuda")
        attention_mask = tokenized["attention_mask"].to("cuda")

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        data = DataProto(
            batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
        )
        data = data.to("cuda")
        data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            if isinstance(output, torch.Tensor):
                generate_ids = output[:, len(input_ids[0]) :]
            else:
                generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]

        output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        self.logger.info(f"judge model inference output: {str(output)}")
        return output.strip()

    def _extract_score(self, response: str) -> float:
        try:
            match = re.search("Score: ([0-9.]+)", response)
            if match:
                score = float(match.group(1))
                normalized_score = score / 10
                return normalized_score
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0.5
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0.5

    def _extract_score_v2(self, response: str) -> float:
        response = response.lower()
        try:
            if "yes" in response:
                return 1
            elif "no" in response:
                return 0
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0

    def _format_judge_prompt(self, prompt: str, response: str, reference: str = None, goal: str = None) -> str:
        if "user\n" in prompt:
            prompt = prompt.split("user\n")[-1].strip()
        if not self.judge_prompt:
            formatted_prompt = f"""
            You are an expert judge evaluating the quality of a response to a given prompt.
            
            Prompt: {prompt}
            
            Response: {response}
            
            Reference: {reference}
            
            Please evaluate the response on a scale from 0 to 10.
            Consider factors such as correctness, completeness, clarity, and relevance to the prompt.
            Your evaluation should be a single number between 0 and 10.
            Note output your score in the following format: Score: your score.
            """
        if self.judge_prompt_name == "sotopia_judge":
            formatted_prompt = self.judge_prompt.format(
                                                       
                                                        goal = goal,
                                                        history=select_to_last_period(prompt, upper_token=2048),
                                                        response=select_to_last_period(response, upper_token=512),
                                                        )
        else:
            formatted_prompt = self.judge_prompt.format(
                                                        goal = goal,
                                                        prompt=select_to_last_period(prompt, upper_token=2048),
                                                        response=select_to_last_period(response, upper_token=512),
                                                        reference=select_to_last_period(reference, upper_token=512),
                                                        
                                                        )
        messages = [{"role": "user", "content": formatted_prompt}]
        return messages
    
   

    def _extract_score_v3(self, llm_output: str):
        # 修正正则表达式：允许方括号完全缺失
        pattern = r"(GF|CF|PD|DA|IQ|CR):\s*\[?\s*(\d+)\s*\]?.*?"
        
        matches = re.findall(pattern, llm_output, re.IGNORECASE)
        
        # 初始化默认值（全部设为0）
        scores = {'GF': 0, 'CF': 0, 'PD': 0, 'DA': 0, 'IQ': 0, 'CR': 0}
        
        # 更新匹配到的键值
        for key, value in matches:
            scores[key.upper()] = int(value)  # 转换为大写并存储为整数
        
        # 计算平均分
        total = sum(scores.values())
        average = total / (5*len(scores)) # 0-1
        return average
    
    def compute_reward_instant(self, scores: dict, gamma=None):
        """
        即时计算 reward
        
        Args:
            scores: dict, e.g., {'GOAL': 5, 'REL': 4, 'KNO': 3}
            gamma: dict, 权重，默认为 [0.5, 0.25, 0.25]
        
        Returns:
            float: normalized scalar reward in [0, 1]
        """
        if gamma is None:
            gamma = {'GOAL': 0.33, 'REL': 0.33, 'KNO': 0.33}
        
        total = 0.0
        for d in gamma:
            raw_score = scores.get(d, 1)  # 默认 1 分
            normalized_score = (raw_score - 1) / 4.0  # 映射到 [0, 1]
            total += gamma[d] * normalized_score
        
        return total   # 平均加权得分


    def _extract_score_v4(self, llm_output: str):
        # 修正正则表达式：允许方括号完全缺失
        pattern = r"(GF|REL|KNO):\s*\[?\s*(\d+)\s*\]?.*?"
        
        matches = re.findall(pattern, llm_output, re.IGNORECASE)
        
        # 初始化默认值（全部设为0）
        scores = {'GF': 0, 'REL': 0, 'KNO': 0}
        
        # 更新匹配到的键值
        for key, value in matches:
            scores[key.upper()] = int(value)  # 转换为大写并存储为整数
        average = self.compute_reward_instant(scores)
        return average

    def _get_llm_judgment(self, prompt_id: str, prompt: str, response: str, reference: str = None, goal: str = None) -> float:
        messages = self._format_judge_prompt(prompt, response, reference, goal)

        if self.judge_model_type == "api":
            llm_response = self._call_api_model(messages)
        elif self.judge_model_type == "inference":
            llm_response = self._run_local_inference(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

        if self.judge_prompt_name == "actor_judge":
            score = self._extract_score_v3(llm_response)
        elif self.judge_prompt_name == "sotopia_judge":
            score = self._extract_score_v4(llm_response)
        else:
            score = self._extract_score_v2(llm_response)
            
        info = {
            "prompt_id": prompt_id,
            "score": score,
            "prompt": prompt,
            "response": response,
            # "reference": reference,
            # "messages": messages,
            # "llm_response": llm_response,
        }
        return score, info

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}

        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        else:
            return self._compute_rewards_impl(data, metrics)

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict):
        prompts_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        scores = []
        goal = Dataset_Template[self.data_name]["goal"]
        if "output" in data.non_tensor_batch.keys():
            for prompt_id, prompt_txt, response, reference in zip(
                data.non_tensor_batch["id"], 
                prompts_text_list, 
                response_text_list, 
                data.non_tensor_batch["output"]
            ):
                score, info = self._get_llm_judgment(prompt_id, prompt_txt, response, reference, goal)
                scores.append(score)
                self.logger.info(f"{json.dumps(info, ensure_ascii=False)}")
        else:
            for prompt_id, prompt_txt, response in zip(
                data.non_tensor_batch["id"], 
                prompts_text_list, 
                response_text_list
            ):
                score, info = self._get_llm_judgment(prompt_id, prompt_txt, response, goal=goal)
                scores.append(score)
                self.logger.info(f"{json.dumps(info, ensure_ascii=False)}")

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        output.meta_info = {"metrics": metrics}
        print(f"Computed rewards for {len(scores)} samples")
        return output
