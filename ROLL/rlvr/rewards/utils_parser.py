import re
import json
from typing import Optional, List, Union, Sequence, Dict
from abc import ABCMeta
import logging
import numpy as np
import torch.nn as nn


class ModelResponse:
    """A mock class for model response."""
    def __init__(self, text: str):
        self.text = text
        self.parsed = None

class TagNotFoundError(Exception):
    """Exception raised when a tag is not found in the response."""
    def __init__(self, message: str, response_text: str):
        super().__init__(message)
        self.response_text = response_text

class DictFilterMixin(metaclass=ABCMeta):
    """A mixin class for filtering dictionary content based on keys."""
    def __init__(
        self,
        keys_to_memory: Union[str, bool, Sequence[str]] = True,
        keys_to_content: Union[str, bool, Sequence[str]] = True,
        keys_to_metadata: Union[str, bool, Sequence[str]] = False,
    ) -> None:
        self.keys_to_memory = keys_to_memory
        self.keys_to_content = keys_to_content
        self.keys_to_metadata = keys_to_metadata

class RegexTaggedContentParser(DictFilterMixin):
    """A regex tagged content parser, which extracts tagged content according
    to the provided regex pattern. Different from other parsers, this parser
    allows to extract multiple tagged content without knowing the keys in
    advance. The parsed result will be a dictionary within the parsed field of
    the model response.

    Compared with other parsers, this parser is more flexible and can be used
    in dynamic scenarios where
    - the keys are not known in advance
    - the number of the tagged content is not fixed

    Note: Without knowing the keys in advance, it's hard to prepare a format
    instruction template for different scenarios. Therefore, we ask the user
    to provide the format instruction in the constructor. Of course, the user
    can construct and manage the prompt by themselves optionally.

    Example:
        By default, the parser use a regex pattern to extract tagged content
        with the following format: 
    """
    def __init__(
        self,
        tagged_content_pattern: str = r"<(?P<name>[^>]+)>"
        r"(?P<content>.*?)"
        r"</\1?>",
        format_instruction: Optional[str] = None,
        try_parse_json: bool = False,
        required_keys: Optional[List[str]] = None,
        keys_to_memory: Union[str, bool, Sequence[str]] = True,
        keys_to_content: Union[str, bool, Sequence[str]] = True,
        keys_to_metadata: Union[str, bool, Sequence[str]] = False,
    ) -> None:
        """Initialize the regex tagged content parser.

        Args:
            tagged_content_pattern (`Optional[str]`, defaults to
            `"<(?P<name>[^>]+)>(?P<content>.*?)</\1?>"`):
                The regex pattern to extract tagged content. The pattern should
                contain two named groups: `name` and `content`. The `name`
                group is used as the key of the tagged content, and the
                `content` group is used as the value.
            format_instruction (`Optional[str]`, defaults to `None`):
                The instruction for the format of the tagged content, which
                will be attached to the end of the prompt messages to remind
                the LLM to follow the format.
            try_parse_json (`bool`, defaults to `True`):
                Whether to try to parse the tagged content as JSON. Note
                the parsing function won't raise exceptions.
            required_keys (`Optional[List[str]]`, defaults to `None`):
                The keys that are required in the tagged content.
            keys_to_memory (`Union[str, bool, Sequence[str]]`,
            defaults to `True`):
                The keys to save to memory.
            keys_to_content (`Union[str, bool, Sequence[str]]`,
            defaults to `True`):
                The keys to save to content.
            keys_to_metadata (`Union[str, bool, Sequence[str]]`,
            defaults to `False`):
                The key or keys to be filtered in `to_metadata` method. If
                it's
                - `False`, `None` will be returned in the `to_metadata` method
                - `str`, the corresponding value will be returned
                - `List[str]`, a filtered dictionary will be returned
                - `True`, the whole dictionary will be returned
        """

        DictFilterMixin.__init__(
            self,
            keys_to_memory=keys_to_memory,
            keys_to_content=keys_to_content,
            keys_to_metadata=keys_to_metadata,
        )

        assert (
            "<name>" in tagged_content_pattern
        ), "The tagged content pattern should contain a named group 'name'."
        assert (
            "<content>" in tagged_content_pattern
        ), "The tagged content pattern should contain a named group 'content'."

        self.tagged_content_pattern = tagged_content_pattern

        self._format_instruction = format_instruction
        self.try_parse_json = try_parse_json
        self.required_keys = required_keys or []

    @property
    def format_instruction(self) -> str:
        """The format instruction for the tagged content."""
        if self._format_instruction is None:
            raise ValueError(
                "The format instruction is not provided. Please provide it in "
                "the constructor of the parser.",
            )
        return self._format_instruction

    def parse(self, response: ModelResponse) -> ModelResponse:
        """Parse the response text by the regex pattern, and return a dict of
        the content in the parsed field of the response.

        Args:
            response (`ModelResponse`):
                The response to be parsed.

        Returns:
            `ModelResponse`: The response with the parsed field as the parsed
            result.
        """
        assert response.text is not None, "The response text is None."

        matches = re.finditer(
            self.tagged_content_pattern,
            response.text,
            flags= re.DOTALL,
        )


        results = {}
        for match in matches:
            results[match.group("name")] = match.group("content")

        keys_missing = [
            key for key in self.required_keys if key not in results
        ]

        if len(keys_missing) > 0:
            raise TagNotFoundError(
                f"Failed to find tags: {', '.join(keys_missing)}",
                response.text,
            )

        if self.try_parse_json:
            keys_failed = []
            for key in results:
                try:
                    results[key] = json.loads(results[key])
                except json.JSONDecodeError:
                    keys_failed.append(key)

            if keys_failed:
                logger.debug(
                    f'Failed to parse JSON for keys: {", ".join(keys_failed)}',
                )

        response.parsed = results
        return response

    def _filter_content_by_names(
        self,
        parsed_response: dict,
        keys: Union[str, bool, Sequence[str]],
        allow_missing: bool = False,
    ) -> Union[str, dict, None]:
        """Filter the parsed response by keys. If only one key is provided, the
        returned content will be a single corresponding value. Otherwise,
        the returned content will be a dictionary with the filtered keys and
        their corresponding values.

        Args:
            keys (`Union[str, bool, Sequence[str]]`):
                The key or keys to be filtered. If it's
                - `False`, `None` will be returned in the `to_content` method
                - `str`, the corresponding value will be returned
                - `List[str]`, a filtered dictionary will be returned
                - `True`, the whole dictionary will be returned
            allow_missing (`bool`, defaults to `False`):
                Whether to allow missing keys in the response. If set to
                `True`, the method will skip the missing keys in the response.
                Otherwise, it will raise a `ValueError` when a key is missing.

        Returns:
            `Union[str, dict]`: The filtered content.
        """

        if isinstance(keys, bool):
            if keys:
                return parsed_response
            else:
                return None

        if isinstance(keys, str):
            return parsed_response[keys]

        # check if the required names are in the response
        for name in keys:
            if name not in parsed_response:
                if allow_missing:
                    logger.warning(
                        f"Content name {name} not found in the response. Skip "
                        f"it.",
                    )
                else:
                    raise ValueError(f"Name {name} not found in the response.")
        return {
            name: parsed_response[name]
            for name in keys
            if name in parsed_response
        }

    def to_memory(
        self,
        parsed_response: dict,
        allow_missing: bool = False,
    ) -> Union[str, dict, None]:
        """Filter the fields that will be stored in memory."""
        return self._filter_content_by_names(
            parsed_response,
            self.keys_to_memory,
            allow_missing=allow_missing,
        )

    def to_content(
        self,
        parsed_response: dict,
        allow_missing: bool = False,
    ) -> Union[str, dict, None]:
        """Filter the fields that will be fed into the content field in the
        returned message, which will be exposed to other agents.
        """
        return self._filter_content_by_names(
            parsed_response,
            self.keys_to_content,
            allow_missing=allow_missing,
        )

    def to_metadata(
        self,
        parsed_response: dict,
        allow_missing: bool = False,
    ) -> Union[str, dict, None]:
        """Filter the fields that will be fed into the returned message
        directly to control the application workflow."""
        return self._filter_content_by_names(
            parsed_response,
            self.keys_to_metadata,
            allow_missing=allow_missing,
        )


class Data:

    def __init__(self, 
    src_node_ids: np.ndarray, 
    dst_node_ids: np.ndarray, 
    node_interact_times: np.ndarray, 
    edge_ids: np.ndarray, 
    labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


from transformers import AutoTokenizer, AutoModel
import torch
# search for edge types/ dst node idxs
class BertEmbedder:
    def __init__(self, 
                model_args, 
                logger):
        """
        Args:
            model_args: 包含模型路径等信息的对象或 dict
            device_mapping (list): 指定模型加载的设备列表，如 [0] 表示 GPU 0
        """
        
        self.device = f"cpu"
        logger.info(f"Loading bert model to {self.device}")
        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                       device_map = self.device)
        self.model = AutoModel.from_pretrained(model_args.model_name_or_path,
                                                        device_map = self.device)

        # # 将模型加载到指定设备
        # self.model.to(self.device)

    def get_embedding(self, text: str) -> torch.Tensor:
        """获取文本的嵌入向量"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 使用 [CLS] token 的表示作为句子嵌入
        return outputs.last_hidden_state[:, 0, :].cpu()

    def offload_states(self):
        """卸载模型状态到 CPU"""
        self.model.cpu()
        torch.cuda.empty_cache()

    def load_states(self):
        """重新加载模型到指定 GPU"""
        self.model.to(self.device)


if __name__=="__main__":
    text = """.0</price_range></edge>

<edge><dst_node>4876</dst_node>
<ts_str>2025-03-17 08:00:00</ts_str>
<label>Tops/Womens</label>
<cate_level2_name>Womens clothing/Tops</cate_level2_name>
<cate_level3_name>Tops/Womens</cate_level3_name>
<price_range>2.0</price_range></edge>

<edge><dst_node>4920</dst_node>
<ts_str>2025-03-17 08:00:00</ts_str>
<label>Body spray.</label>
<cate_level2_name>Body scents</cate_level2_name>
<cate_level3_name>Body spray.</cate_level3_name>
<price_range>1.0</price_range></edge>
</think>

```xml
<edge>
    <dst_node>4697</dst_node>
    <ts_str>2025-03-17 08:00:00</ts_str>
    <label>Sanitary towels.</label>
    <cate_level2_name>Sanitary towels/principal care</cate_level2_name>
    <cate_level3_name>Sanitary towels.</cate_level3_name>
    <price_range>1.0</price_range>
</edge>
```"""
    text = text.replace("<edge>","").replace("</edge>","")
    parser = RegexTaggedContentParser(
        required_keys = ["label", "cate_level2_name", "cate_level3_name", "price_range", "ts_str"]
    )

    re.finditer(
            r"<(?P<name>[^>]+)>"
        r"(?P<content>.*?)"
        r"</\1?>", text,flags= re.DOTALL )

    parsed = parser.parse(ModelResponse(text)).parsed
    parsed