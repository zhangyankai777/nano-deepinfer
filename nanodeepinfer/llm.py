from nanodeepinfer.engine.llm_engine import LLMEngine, config_from_kwargs
from nanodeepinfer.sampling_params import SamplingParams


class LLM:
    """离线推理入口，对应原版 deepinfer/entrypoints/llm.py 的 LLM。"""

    def __init__(self, model: str, **kwargs):
        self.llm_engine = LLMEngine(config_from_kwargs(model, kwargs))

    def generate(self, prompts=None, sampling_params: SamplingParams | list[SamplingParams] | None = None,
                 prompt_token_ids=None, use_tqdm: bool = True) -> list[dict]:
        return self.llm_engine.generate(prompts, prompt_token_ids, sampling_params, use_tqdm)
