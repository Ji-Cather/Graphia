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
from .utils_dst import *

EVAL_RULE_PROMPT = """
Your task is to predict whether the dst nodes queried by LLM interact with the src node based on the evaluation rules:                    

1. **Node Text (NT):**
- Textual information of each node.

2. **Historical Interaction Count (HI):** (HI)
- The total number of past interactions the two nodes have had.
- High interaction counts indicate active historical interactions between the two nodes and a higher likelihood of future interactions.

3. **Common Neighbor Count (CN):**
- The number of shared neighbors between the two nodes.
- A higher number of common neighbors suggests a stronger community bond and a higher probability of interaction.

4. **Node-Specific Metrics:**
- **For src-node:**
    - **Self Frequency(SF):** Total number of interactions.
     - **Average Frequency of Neighbors(AFN):** Average number of interactions per historical neighbor, indicating whether the node's neighbors are predominantly new or well-established.

- **For dst-node:**
    - **Self Frequency(SF):** Total number of interactions.
     - **Average Frequency of Neighbors(AFN):** Average number of interactions per historical neighbor, indicating whether the node's neighbors are predominantly new or well-established.

- If the destination node has participated in many interactions overall but has rarely been a destination in past interactions, it is less likely for this pair to form a link. 
- If a node demonstrates a preference for neighbors with high or low interaction frequency, it is likely to exhibit the same preference for forming links with new nodes.
"""

EVAL_RULE_PROMPT = """
Task: Predict if dst nodes interact with src node using these rules:
Node Text (NT): Node attributes (e.g., user profile).
Historical Interaction (HI): Total past interactions between src-dst pairs.
    Higher HI: stronger historical ties, higher interaction likelihood.
Common Neighbors (CN): Shared neighbors between src-dst nodes.
    Higher CN: tighter community bonds, higher interaction probability.
Node-Specific Metrics:
    Self Frequency (SF): Total interactions per node (src/dst).
    Neighbor Frequency (AFN): Average interaction count of a node's historical neighbors.
    High AFN = neighbors are established; low AFN = neighbors are new.
"""
    
REFERENCE_PROMPT = """
    Prompt: {prompt}

    Ground-truth Observation: {reference_observation}
    
    LLM Query: {llm_response} \n LLM Observation: {llm_observation}
"""


NOREFERENCE_PROMPT = """
    Prompt: {prompt}
    
    LLM Query: {llm_response} \n LLM Observation: {llm_observation}
"""

REWARD_SYS_PROMPT = """
You're a critique for the LLM query for dst node prediction. Give numeric score between 0 and 100 for accuracy reward.

{eval_rule_prompt}

{reference_prompt}
"""


FORMAT_PROMPT = f"""
{chr(10).join([
        f"<reward> Int [0-100], reward score for the LLM query, which evaluates the overlap between gt dst nodes and the candidate dst nodes</reward>",
        f"<think> short_text_analysis </think>"
        ])}"""

# "<suggestion>Suggestion for modification to the query agent, the suggestion optionally can include the filter rule, similar to {'SF': '>1', 'AFN': '<1', 'HI': '==0', 'CN': '>=0'}</suggestion>",
# Suggestion for modification of LLM Query. The suggestion optionally can include the filter rule, similar to {{'SF': '>1', 'AFN': '<1', 'HI': '==0', 'CN': '>=0'}}
#  f"<think>Give the logical reason for the reward score. 100 words. </think>",
    





JUDGE_PROMPT_TYPE_MAP = {
    "judge_with_reference_rule": {
        "prompt":REWARD_SYS_PROMPT.format(
            eval_rule_prompt = EVAL_RULE_PROMPT,
            reference_prompt = REFERENCE_PROMPT
        ) + "\n" + FORMAT_PROMPT,
        "required_keys": ["prompt", "llm_response", "llm_observation","reference_observation"
        ],
    },
    "judge_with_reference":{
        "prompt":REWARD_SYS_PROMPT.format(
            eval_rule_prompt = "",
            reference_prompt = REFERENCE_PROMPT
        ) + "\n" + FORMAT_PROMPT,
        "required_keys": ["prompt", "llm_response", "llm_observation","reference_observation"],
    },
    "judge_with_rule":{
        "prompt":REWARD_SYS_PROMPT.format(
            eval_rule_prompt = EVAL_RULE_PROMPT,
            reference_prompt = NOREFERENCE_PROMPT
        ) + "\n" + FORMAT_PROMPT,
        "required_keys": ["prompt", "llm_response", "llm_observation"],
    },
    "judge_with_nothing":{
        "prompt":REWARD_SYS_PROMPT.format(
            eval_rule_prompt = "",
            reference_prompt = NOREFERENCE_PROMPT
        ) + "\n" + FORMAT_PROMPT,
        "required_keys": ["prompt", "llm_response", "llm_observation"],
    }
}

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


class DstLLMJudgeRewardWorker(Worker):
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
        dst_judge_prompt_type = self.worker_config.dst_judge_prompt_type
        self.judge_prompt = JUDGE_PROMPT_TYPE_MAP[dst_judge_prompt_type]["prompt"]
        self.judge_prompt_keys = JUDGE_PROMPT_TYPE_MAP[dst_judge_prompt_type]["required_keys"]

        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "api"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None

        self.mapper = DstMapper(worker_config.environment_data_args,
                                worker_config.reward_args
                                )
        self.bert_args = worker_config.bert_args
        # to be modified, suggestion modified to prompt for multi-turn rl ?
        self.res_parser: RegexTaggedContentParser = RegexTaggedContentParser(
            required_keys=["reward"])



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

        
        self.bert_embedder = BertEmbedder(self.bert_args,
                                          logger=self.logger)
        # self.bert_embedder.offload_states()

    def _call_api_model(self, messages: Dict, retry_times=3) -> str:
        from openai import OpenAI

        output = ""
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


    def _format_judge_prompt(self, 
                        prompt: str, 
                        llm_response: str, 
                        llm_observation: str = None,
                        reference_observation: str = None) -> str:
        prompt_head = "```"
        if prompt_head in prompt:
            prompt = prompt.split(prompt_head)[-1].strip()
        
        inputs = {
            "prompt": select_to_last_period(prompt,upper_token = 2e3),
            "llm_response": llm_response,
            "llm_observation": select_to_last_period(llm_observation, upper_token = 5e2),
            "reference_observation": select_to_last_period(reference_observation, upper_token = 5e2),
        }
        self.logger.debug(json.dumps(inputs,
                    ensure_ascii=False,
                ))

        # check valid inputs
        for key in self.judge_prompt_keys:
            assert inputs[key] is not None, f"{key} should not be None"

        formatted_prompt = self.judge_prompt.format_map(inputs)
        messages = [{"role": "user", "content": formatted_prompt}]
        return messages

    
    def _parse_reward_score(self, reward_response: str) -> float:
        reward_score = 0.0
        try:
            parsed = self.res_parser.parse(ModelResponse(reward_response)).parsed
            reward_score = float(parsed["reward"])
            self.logger.debug(f"parsed: {e}, reward_response: {reward_response}")
        except Exception as e:
            self.logger.error(f"parse error: {e}, reward_response: {reward_response}")
        return reward_score


    def _get_llm_judgment(self, 
                        prompt_id: str, 
                        gt_dst_idx: list[int],
                        src_idx:int, 
                        dx_src: int,
                        prompt: str, 
                        llm_response: str,
                        tag: str) -> float:
        # llm_observation: str = None,
        # reference_observation: str = None
        rewards = {
            "score":0.0,
        }
        try:
            parsed_llm_resp = self.mapper.parse_response(
                                llm_response, 
                                self.logger) # dict
            # parsed_resp
            assert isinstance(parsed_llm_resp, dict)
            try:
                candidate_dst_ids = parsed_llm_resp["candidate_dst_ids"]
            except Exception as e:
                candidate_dst_ids = []

            parsed_llm_resp.update({
                "src_idx": int(src_idx),
                "gt_dx_src_unique": int(dx_src),
                "gt_dst_idxs_unique": np.array(gt_dst_idx, dtype=int).tolist(),
                "candidate_dst_ids": candidate_dst_ids,
                "tag": tag,
            })
            
            rewards, parsed_llm_obs = self.mapper._collect_reward(parsed_llm_resp, 
                                                    self.bert_embedder,   
                                                    self.logger)
            llm_observation = """**Searched candidate dst nodes by LLM:\n{observation}""".format(observation=parsed_llm_obs["candidate_dst_texts"])
            query_text = parsed_llm_obs.get("query_text", "")
            filter_rule = parsed_llm_obs.get("filter_rule", "")

            llm_response_filtered = f"{query_text}\n<filter_rule>{filter_rule}</filter_rule>" 

        except Exception as e:
            self.logger.error(f"Error parsing llm response: {e}")

            # return {"llm_reward": 0,"score":0}, {"llm_response_raw": llm_response, "state": "fail to parse llm response"}

            # ### for debug
            # llm_response_filtered = """<filter_rule></filter_rule><profile>test_profile</profile>"""
            # llm_observation = "test_observation"

        reference_obs = self.mapper._collect_reference_obs(
                        src_idx,
                        gt_dst_idx,
                        tag=tag
        )
        reference_observation = f"""**Ground truth dst nodes for the src node to interact:\n{reference_obs}"""

        messages = self._format_judge_prompt(
                            prompt, 
                            llm_response_filtered, 
                            llm_observation,
                            reference_observation)

        if self.judge_model_type == "api":
            reward_response = self._call_api_model(messages)
        elif self.judge_model_type == "inference":
            reward_response = self._run_local_inference(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

        llm_reward = self._parse_reward_score(reward_response)
        # self.logger.debug(f"reward_response: {reward_response}")
        info = {
            "prompt_id": prompt_id,
            "llm_reward": llm_reward,
            "src_idx": src_idx,
            **rewards
        }
        self.logger.debug(json.dumps(info,
                    ensure_ascii=False,
                ))
        try:
            reward_dict = {"llm_reward": llm_reward,
            "score":rewards["score"]}
        except Exception as e:
            reward_dict = {"llm_reward": 0,
            "score":0}
            self.logger.error("reward dict err", e)

        return reward_dict, info

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
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

        gt_dst_idxs = data.non_tensor_batch["gt_dst_idxs_unique"]
        tags = data.non_tensor_batch["tag"]
        src_idxs = data.non_tensor_batch["src_idx"]
        dx_srcs = data.non_tensor_batch["gt_dx_src_unique"]

        reward_dicts = []
        for prompt_id, prompt_txt, response, gt_dst_idx, tag, src_idx, dx_src in zip(
            data.non_tensor_batch["id"], 
            prompts_text_list, 
            response_text_list, 
            gt_dst_idxs, 
            tags, 
            src_idxs,
            dx_srcs
        ):
            reward_dict, info = self._get_llm_judgment(
            prompt_id, 
            gt_dst_idx, 
            src_idx, 
            dx_src, 
            prompt_txt,
            response,
            tag
            )
            reward_dicts.append(reward_dict)
            self.logger.debug(f"{json.dumps(info, ensure_ascii=False)}")

        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        scores_tensor = torch.tensor([_.get("score", 0) for _ in reward_dicts], dtype=torch.float16)
        response_level_rewards = torch.tensor([_.get("llm_reward", 0) for _ in reward_dicts], dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        output.meta_info = {"metrics": metrics}
        print(f"Computed rewards for {len(reward_dicts)} samples")
        return output
