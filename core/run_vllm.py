from dataclasses import dataclass
from typing import Dict, List, AsyncGenerator, Tuple
from uuid import uuid4

import torch
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput

from helpers.chroma_utils import vectorstore
from helpers.constants import HF_TOKEN, SYSTEM_PROMPT, PROMPT, GLOBAL_CONFIG

NUM_GPUS = torch.cuda.device_count()


@dataclass
class ModelConfig:
    """Configuration for loading the model."""

    name: str
    model_id: str
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    max_model_len: int = 8912


@dataclass
class GenerationConfig:
    """Configuration for generation parameters."""

    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float


class DocQA:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config["retriever"]["k"]}
        )
        self.system_dict = {"role": "system", "content": SYSTEM_PROMPT}
        self.sampling_params = self._get_sampling_params()
        self._init_models()

    def _init_models(self) -> None:
        """Initialize model artifacts as specified in the configuration."""
        self.models = {}
        for model_config in self.config["models"]:
            model_cfg = ModelConfig(
                **{
                    k: v
                    for k, v in model_config.items()
                    if k in ModelConfig.__annotations__
                }
            )

            engine_args = AsyncEngineArgs(
                model=model_cfg.model_id,
                tensor_parallel_size=NUM_GPUS,
                gpu_memory_utilization=model_cfg.gpu_memory_utilization,
                trust_remote_code=model_cfg.trust_remote_code,
                max_model_len=model_cfg.max_model_len,
                token=HF_TOKEN,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)
            model = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.models[model_cfg.name] = {
                "model": model,
                "tokenizer": tokenizer
            }

    def _get_sampling_params(self) -> SamplingParams:
        gen_config = self.config["generation"]
        return SamplingParams(
            max_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            repetition_penalty=gen_config["repetition_penalty"],
        )

    async def __call__(
        self, query: str, model_name: str, messages: List[Dict[str, str]]
    ) -> Tuple[AsyncGenerator[RequestOutput, None], str]:
        """
        Generate responses based on the query and specified model.
        """
        if model_name not in self.models:
            raise KeyError(
                f"Model {model_name} not found. Available models: {list(self.models.keys())}"
            )

        model = self.models[model_name]["model"]
        tokenizer = self.models[model_name]["tokenizer"]
        # Retrieve relevant documents
        retrieved_docs = await self.retriever.ainvoke(query)
        retrieved_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        # Prepare input messages
        messages_copy = messages.copy()
        messages_copy.insert(0, self.system_dict)
        messages_copy.append(
            {
                "role": "user",
                "content": PROMPT.format(question=query, context=retrieved_docs),
            }
        )
        input_text = tokenizer.apply_chat_template(
            messages_copy, tokenize=False, add_generation_prompt=True
        )
        request_id = str(uuid4())
        request_generator = model.generate(
            prompt=input_text,
            sampling_params=self.sampling_params,
            request_id=request_id,
        )

        return request_generator, retrieved_docs
