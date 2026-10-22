import json
import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """DeepSeek V2/V3 家族的 HF config 子集，字段名与 config.json 保持一致。"""

    hidden_size: int = 2048
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 10944
    vocab_size: int = 102400
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False
    # MLA
    q_lora_rank: int | None = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    attention_bias: bool = False
    # MoE
    first_k_dense_replace: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 1408
    n_group: int = 4
    topk_group: int = 2
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    moe_layer_freq: int = 1

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class Config:
    model: str
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16384
    max_model_len: int | None = None
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1  # -1 表示由 ModelRunner 按可用内存推导
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = True
    device: str = "auto"
    seed: int = 0
    eos: int = -1
    hf_config: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if os.path.isdir(self.model):
            with open(os.path.join(self.model, "config.json")) as f:
                raw = json.load(f)
        else:
            from huggingface_hub import hf_hub_download

            with open(hf_hub_download(self.model, "config.json")) as f:
                raw = json.load(f)
        known = {k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__}
        self.hf_config = ModelConfig(**known)
        if self.max_model_len is None:
            self.max_model_len = self.hf_config.max_position_embeddings
        assert self.max_model_len >= 1
        assert self.kvcache_block_size >= 1
