from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple, Any
from threading import Thread
from vllm import LLM, SamplingParams
import torch
from helpers.chroma_utils import vectorstore
from helpers.constants import HF_TOKEN, SYSTEM_PROMPT, PROMPT, GLOBAL_CONFIG


NUM_GPUS = torch.cuda.device_count()


@dataclass
class ModelConfig:
    name: str
    model_id: str
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    max_model_len: int = 101000

@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float

class DocQA:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config['retriever']['k']}
        )
        self.models = self._load_models()
        self.system_dict = {"role": "system", "content": SYSTEM_PROMPT}

    def _load_models(self) -> Dict[str, Any]:
        models = {}
        for model_config in self.config['models']:
            model_cfg = ModelConfig(**{k: v for k, v in model_config.items() if k in ModelConfig.__annotations__})
            
            llm = LLM(
                model=model_cfg.model_id,
                tensor_parallel_size=NUM_GPUS,
                gpu_memory_utilization=model_cfg.gpu_memory_utilization,
                trust_remote_code=model_cfg.trust_remote_code,
                max_model_len=model_cfg.max_model_len,
                token=HF_TOKEN
            )
            
            models[model_cfg.name] = llm
                
        return models

    def _get_sampling_params(self) -> SamplingParams:
        gen_config = self.config['generation']
        return SamplingParams(
            max_tokens=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
            repetition_penalty=gen_config['repetition_penalty'],
        )

    def __call__(
        self,
        query: str,
        model_name: str,
        messages: List[Dict[str, str]]
    ) -> Generator[Tuple[str, str, str], None, None]:
    
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        model = self.models[model_name]

        retrieved_docs = self.retriever.invoke(query)
        retrieved_docs = '\n'.join([doc.page_content for doc in retrieved_docs])

        messages_copy = messages.copy()
        messages_copy.insert(0, self.system_dict)
        messages_copy.append({
            "role": "user",
            "content": PROMPT.format(question=query, context=retrieved_docs)
        })

        sampling_params = self._get_sampling_params()

        def generation_thread():
            outputs = model.chat(
                messages=messages_copy, 
                sampling_params=sampling_params
            )
            return outputs

        # Start generation in separate thread
        thread = Thread(target=generation_thread)
        thread.start()

        collected_tokens = []
        outputs = generation_thread()
        for output in outputs:
            text = output.outputs[0].text
            collected_tokens.append(text)
            yield text, ''.join(collected_tokens), retrieved_docs