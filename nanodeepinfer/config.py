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
    # ---- V4 家族（以下字段缺省时即 V2/V3 路径）----
    # 稀疏注意力：index_topk>0 开启；索引器从主压缩 KV 投影（use_kproj 变体）
    index_topk: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_use_knorm: bool = False
    use_qnorm: bool = False
    o_groups: int = 0          # V 侧分组低秩；0 时退化为 V3 的 kv_b/o_proj
    o_lora_rank: int = 0
    window_size: int = -1      # SWA 滑窗；-1 关闭
    attn_sink_init_value: float | None = None
    # mHC（多路 Hyper-Connections）
    hc_enabled: bool = False
    hc_mult: int = 1
    hc_dynamic_scale: float = 1.0
    hc_pre_init_value: float = 1.0
    hc_post_init_value: float = 1.0
    hc_res_init_value: float = 1.0
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 10
    hc_post_mult_value: float = 1.0
    # Engram（n-gram 条件记忆，依赖 HC）
    engram_layer_ids: list[int] | None = None
    engram_max_ngram_size: int = 2
    engram_vocab_size: list[int] | None = None      # 每阶哈希表基数（取素数）
    n_embed_per_ngram: list[int] | None = None      # 每阶嵌入维（= n_head*head_dim）
    n_head_per_ngram: list[int] | None = None
    engram_eps: float = 1e-6
    engram_clamp_value: float = 15.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def is_v4(self) -> bool:
        return self.index_topk > 0 or self.hc_enabled


@dataclass
class Config:
    model: str
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16384
    max_model_len: int | None = None
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1  # -1 表示由 ModelRunner 按可用内存推导
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False  # False 时 decode 用 CUDA graph 捕获重放（仅 CUDA）
    device: str = "auto"
    dtype: str = "auto"  # auto: CUDA 用 bfloat16、CPU 用 float32
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
