import time
from dataclasses import fields

from tqdm import tqdm
from transformers import AutoTokenizer

from nanodeepinfer.config import Config
from nanodeepinfer.engine.model_runner import ModelRunner
from nanodeepinfer.engine.scheduler import Scheduler
from nanodeepinfer.engine.sequence import Sequence
from nanodeepinfer.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=False)
        if config.eos == -1:
            config.eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
        self.model_runner = ModelRunner(config)  # 先定 num_kvcache_blocks，Scheduler 再建 BlockManager
        self.scheduler = Scheduler(config)
        self.outputs: dict[int, Sequence] = {}

    def add_request(self, prompt_tokens: list[int], sampling_params: SamplingParams) -> Sequence:
        assert 0 < len(prompt_tokens) <= self.config.max_model_len
        seq = Sequence(prompt_tokens, sampling_params, self.config.kvcache_block_size)
        self.scheduler.add(seq)
        self.outputs[seq.seq_id] = seq
        return seq

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

    def generate(self, prompts: list[str] | None = None,
                 prompt_token_ids: list[list[int]] | None = None,
                 sampling_params: SamplingParams | list[SamplingParams] | None = None,
                 use_tqdm: bool = True) -> list[dict]:
        assert prompts or prompt_token_ids, "prompts or prompt_token_ids is required"
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * (len(prompts) if prompts else len(prompt_token_ids))
        seqs = []
        if prompts:
            for prompt, sp in zip(prompts, sampling_params):
                ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                seqs.append(self.add_request(ids, sp))
        else:
            for ids, sp in zip(prompt_token_ids, sampling_params):
                seqs.append(self.add_request(list(ids), sp))
        total = sum(s.max_tokens for s in seqs)
        bar = tqdm(total=total, desc="Generating", disable=not use_tqdm)
        done = 0
        t0 = time.time()
        while not self.scheduler.is_finished():
            self.step()
            now = sum(s.num_completion_tokens for s in seqs)
            bar.update(now - done)
            done = now
        bar.close()
        elapsed = time.time() - t0
        outputs = []
        for seq in seqs:
            ids = seq.token_ids[seq.num_prompt_tokens:]
            outputs.append({
                "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                "token_ids": ids,
                "num_prompt_tokens": seq.num_prompt_tokens,
                "elapsed": elapsed,
            })
            del self.outputs[seq.seq_id]
        return outputs


def config_from_kwargs(model: str, kwargs: dict) -> Config:
    names = {f.name for f in fields(Config)} - {"hf_config", "eos"}
    return Config(model=model, **{k: v for k, v in kwargs.items() if k in names})
