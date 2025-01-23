from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Any
from threading import Thread
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from helpers.chroma_utils import vectorstore
from helpers.constants import HF_TOKEN, SYSTEM_PROMPT, PROMPT, GLOBAL_CONFIG
from core.streamer import ResponseTokensStreamer

@dataclass
class ModelConfig:
    name: str
    model_id: str
    low_cpu_mem_usage: bool = True

@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float
    no_repeat_ngram_size: int
    early_stopping: bool

class DocQA:
    def __init__(self):
        """Initialize DocQA with configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = GLOBAL_CONFIG
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config['retriever']['k']}
        )
        self.models = self._load_models()
        self.system_dict = {"role": "system", "content": SYSTEM_PROMPT}


    def _load_models(self) -> Dict[str, Dict[str, Any]]:
        models = {}
        model_cfg = None
        for model_config in self.config['models']:
            # try:
            model_cfg = ModelConfig(**model_config)
            model = AutoAWQForCausalLM.from_pretrained(
                    model_path=model_cfg.model_id,
                    device_map='auto',
                    low_cpu_mem_usage=model_cfg.low_cpu_mem_usage,
                    token=HF_TOKEN,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)
            models[model_cfg.name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            # except Exception as e:
            #     raise ValueError(f"Failed to load model {model_cfg.name}: {str(e)}")
                
        return models

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(**self.config['generation'])

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

        model = self.models[model_name]['model']
        tokenizer = self.models[model_name]['tokenizer']

        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(query)
        retrieved_docs = '\n'.join([doc.page_content for doc in retrieved_docs])

        # Prepare messages
        messages_copy = messages.copy()
        messages_copy.insert(0, self.system_dict)
        messages_copy.append({
            "role": "user",
            "content": PROMPT.format(question=query, context=retrieved_docs)
        })

        input_text = tokenizer.apply_chat_template(
            messages_copy,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([input_text], return_tensors="pt").to('cuda')

        streamer = ResponseTokensStreamer(
            tokenizer,
            model_inputs.input_ids,
            skip_special_tokens=True
        )

        gen_config = self._get_generation_config()
        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            **gen_config.__dict__
        }

        # Start generation in separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        collected_tokens = []
        for text in streamer:
            collected_tokens.append(text)
            yield text, ''.join(collected_tokens), retrieved_docs