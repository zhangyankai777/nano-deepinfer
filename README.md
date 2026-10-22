# Nano-DeepInfer

一个极简、可读的 [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) 教学实现，仿照 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的做法：只保留推理框架的核心链路，用约 900 行纯 PyTorch 代码把 DeepInfer 的请求生命周期、调度、KV cache 与 DeepSeek 标志性的 MLA + MoE 模型结构串起来，方便新手顺着一条主线读懂一个生产级推理框架是怎么工作的。

原版 DeepInfer 是 DeepSeek 支撑大规模线上服务的推理框架，包含 4 万余行 Python、C++/TileLang 内核、DeepEP/DeepGEMM/FlashMLA 等厂商级优化与 DP/EP/CP 并行。本项目不追求性能对标，而是把其中最值得学习的设计压缩成可以在一台普通机器上读完、跑通、改得动的版本。

## 核心链路

一次 `generate` 调用在本项目中的完整路径，也就是阅读代码的主线：

```text
LLM.generate(prompts, SamplingParams)
  └─ LLMEngine        分词、组装请求、驱动主循环            engine/llm_engine.py
      └─ Scheduler    连续批处理：prefill 准入 / decode / 抢占   engine/scheduler.py
          ├─ BlockManager   分页 KV cache 分配 + 前缀缓存复用     engine/block_manager.py
          └─ ModelRunner    备张量 → 前向 → 采样                engine/model_runner.py
              └─ DeepseekForCausalLM                            models/deepseek.py
                  ├─ MLAAttention   低秩潜向量 KV 缓存 + 解耦 RoPE  layers/attention.py
                  └─ MoE            分组受限 topk 路由 + 共享专家    layers/moe.py
```

## 保留了什么，简化了什么

| DeepInfer 的设计 | 本项目 | 说明 |
|---|---|---|
| 离线 `LLM` + `SamplingParams` 入口 | ✅ 保留 | API 形态与原版 `deepinfer/entrypoints/llm.py` 对齐 |
| 引擎主循环与异步调度 | ✅ 简化 | 保留准入/批处理/抢占决策，去掉 CPU 调度与设备执行的异步重叠 |
| 连续批处理、prefill/decode 两阶段 | ✅ 保留 | 额外实现分块 prefill，长 prompt 可分多个 step 灌入 |
| KV cache 管理（原版未开源） | ✅ 补齐 | 分页块管理 + 链式哈希前缀缓存，思路与 vLLM 系一脉相承 |
| MLA（多头潜注意力） | ✅ 保留 | KV cache 只存 `kv_lora_rank + qk_rope_head_dim` 维潜向量，注意力以吸收式在潜空间计算（`kv_b_proj` 的 K 部分吸收进 Q、V 部分在加权后作用），不还原完整 K/V |
| MoE 路由 | ✅ 保留 | sigmoid 打分、校正偏置（noaux_tc 无辅助损失）、分组受限 topk、共享专家、前几层 dense |
| 确定性采样（`seq_seed`） | ✅ 保留 | 温度/top-k/top-p 与原版同名字段 |
| DeepGEMM / FlashMLA / TileLang 内核 | ❌ 简化为纯 PyTorch | 不写自定义内核，但保留其核心变换：MLA 吸收式潜空间注意力、MoE 专家批量矩阵乘；数学与原版等价 |
| DP / EP / CP 并行、EPLB | ❌ 省略 | 单机单卡路径；`Gate` 与专家的组织方式即为 EP 切分的天然位置 |
| DSpark / MTP 投机解码 | ❌ 省略 | 原版在 `ModelRunner` 之上做草稿-验证，接口位置见下方扩展点 |
| SWA bounded replay、结构化输出 | ❌ 省略 | 属于特性层，不影响主链路理解 |
| CUDA graph | ✅ 保留 | decode 走整批静态形状前向并按 (批量, 长度桶) 捕获重放；prefill 与 CPU 走 eager，`enforce_eager=True` 可关闭 |
| FP8 检查点 | ✅ 保留 | 加载时按 128×128 分块 scale 反量化，见 `utils/loader.py` |

## 适配硬件

本项目只有一个版本，不需要为不同显卡分档。原版 DeepInfer 要求 NVIDIA SM100 / 华为 Ascend 950，是因为它依赖 FlashMLA、DeepGEMM 等只为新架构编写的内核；本项目用纯 PyTorch 复刻这些内核的数学、CUDA 上统一用 bfloat16 计算，因此对卡型没有特殊要求：

| 环境 | 要求 |
|---|---|
| NVIDIA GPU | 计算能力 SM80 及以上均可，包括 RTX 3060（SM86）/3070/3090、RTX 40 系、A100/H100 以及 SM100 数据中心卡；有卡自动使用 |
| CPU | 无显卡时自动退回 CPU（float32），适合小模型对照实验 |

曾短暂提供过一个 `rtx3060` 配置档（float16 + 收紧的批量上限），在 RTX 3060 上与默认 bfloat16 配置做了同负载对照实测（3.4 亿参数同构模型、8 条序列 × 256 token、交替顺序多轮）：默认档 89.9 tok/s、3060 档 87.0~87.9 tok/s，差异在测量噪声范围内——因为两者共用同一套算子实现，分档并无性能收益，故已删除，只保留实测不慢且数值范围更好的默认配置。

需要注意的是，能否装得下只取决于模型权重体积：例如 DeepSeek-V2-Lite 的 bfloat16 权重约 31GB，超出 RTX 3060 的 12GB，这类模型需 24GB 以上显存的卡；在 3060 上请改用更小的同构检查点（按 `config.json` 缩小层数/维度自训或蒸馏的小模型均可直接加载）。显存余量紧张时，可通过 `max_num_seqs`、`max_num_batched_tokens`、`max_model_len`、`gpu_memory_utilization` 调小批量与 KV cache 占用，见下方参数表。

## 快速开始

### 安装

```bash
pip install git+https://github.com/zhangyankai777/nano-deepinfer.git
# 或克隆后本地安装
pip install .
```

依赖仅 `torch`、`transformers`、`safetensors`、`tqdm`，支持 CPU 与 CUDA（有卡自动用卡）。

### 运行

```python
from nanodeepinfer import LLM, SamplingParams

llm = LLM("deepseek-ai/DeepSeek-V2-Lite-Chat", max_num_seqs=16)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["请用一句话介绍深度学习推理框架。"], sampling_params)
print(outputs[0]["text"])
```

也可以直接运行仓库自带的脚本，任意 SM80 及以上显卡（包括 RTX 3060）上命令完全相同：

```bash
python example.py --model deepseek-ai/DeepSeek-V2-Lite-Chat
python bench.py --model /path/to/model --num-seqs 16
```

在 RTX 3060 等 12GB 显存的卡上加载小模型时，建议把批量与上下文收紧以免显存溢出：

```python
llm = LLM("/path/to/model", max_num_seqs=8, max_num_batched_tokens=2048, max_model_len=4096)
```

无可用显卡时自动退回 CPU 运行（`device="cpu"`），适合对照代码做小模型实验。

支持的检查点为 HuggingFace 格式的 DeepSeek-V2 / V3 家族模型（`config.json` + safetensors，含 FP8 权重），例如 DeepSeek-V2-Lite、DeepSeek-V3、DeepSeek-R1。模型结构由 `config.json` 字段驱动，改几个维度参数即可构造任意大小的同构模型做实验。

### `LLM` 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `dtype` | `auto` | `auto` 时 CUDA 用 bfloat16、CPU 用 float32；也可指定 `float16`/`bfloat16`/`float32` |
| `max_num_seqs` | 256 | 单个 batch 内最大序列数 |
| `max_num_batched_tokens` | 16384 | 单步最大 token 预算，决定分块 prefill 的切分 |
| `max_model_len` | 取自 config | 最大上下文长度 |
| `kvcache_block_size` | 16 | 分页 KV cache 的块大小 |
| `num_kvcache_blocks` | -1（自动） | KV cache 块数，自动时按可用显存/内存预算推导 |
| `gpu_memory_utilization` | 0.9 | 自动推导块数时使用的显存比例 |
| `device` | `auto` | `auto` / `cuda` / `cpu` |

### `SamplingParams` 字段

`temperature`（0 为贪心）、`top_p`、`top_k`、`max_tokens`、`ignore_eos`、`stop_token_ids`、`seq_seed`（给定后采样可复现）。

## 代码结构

```text
nanodeepinfer/
├─ llm.py                  # LLM 入口，对应原版 entrypoints/llm.py
├─ config.py               # 引擎配置 + DeepSeek config 字段子集
├─ sampling_params.py
├─ engine/
│  ├─ llm_engine.py        # 分词、请求组装、生成主循环
│  ├─ scheduler.py         # 连续批处理、分块 prefill、重算式抢占
│  ├─ sequence.py          # 单条请求的全部状态
│  ├─ block_manager.py     # 分页块分配、链式哈希前缀缓存
│  └─ model_runner.py      # 输入准备、KV cache 分配、执行与采样调度
├─ models/deepseek.py      # DeepseekForCausalLM：embedding + N 层 + lm_head
├─ layers/
│  ├─ attention.py         # MLA：潜向量缓存、解耦 RoPE、吸收式潜空间注意力
│  ├─ moe.py               # Gate（noaux_tc）+ routed/shared experts
│  ├─ layernorm.py         # RMSNorm
│  ├─ rotary_embedding.py  # DeepSeek 交错式 RoPE
│  └─ sampler.py           # 温度 / top-k / top-p / 贪心
└─ utils/
   ├─ context.py           # 一步之内的全局执行上下文（块表、槽位映射）
   └─ loader.py            # safetensors 加载与 FP8 反量化
```

## 三个值得细读的实现点

**1. MLA 如何省 KV cache（`layers/attention.py`）。** 每个 token 不存各头的完整 K/V，而是存一个 `kv_lora_rank` 维的压缩潜向量加一份共享的 RoPE 键。缓存占用与头数无关，这是 DeepSeek 模型能把 KV cache 压到传统 MHA 几十分之一的关键。计算时不还原 K/V：`kv_b_proj` 生成 K 的部分被吸收进 Q、生成 V 的部分在注意力加权之后才作用于潜向量，这正是 FlashMLA 采用的等价变换。读代码时重点看 `kv_a_proj_with_mqa` 的写缓存与 `_projections` 的吸收。

**2. 前缀缓存如何复用（`engine/block_manager.py`）。** 每个写满的块以 `(父块哈希, 本块 token)` 做链式哈希，相同前缀的请求可以直接引用已有块并跳过这部分 prefill。`can_allocate` 返回可命中的块数，`Scheduler` 据此只计算未缓存的后缀。

**3. 调度如何处理显存不足（`engine/scheduler.py`）。** decode 阶段某条序列需要新块而空闲块不足时，最晚运行的序列被抢占：释放其全部块、回到等待队列，之后从头重算。这与原版所依赖的思路相同——用重算换显存，实现简单且不会死锁。

## 与 DeepInfer 的对应关系

| 本项目文件 | DeepInfer 对应位置 |
|---|---|
| `llm.py` | `deepinfer/entrypoints/llm.py` |
| `engine/llm_engine.py` | `deepinfer/engine/llm_engine.py` |
| `engine/scheduler.py` | `deepinfer/engine/` 内的异步调度器（同步简化版） |
| `engine/model_runner.py` | `deepinfer/worker/model_runner.py` |
| `models/deepseek.py` | `deepinfer/model_executor/models/deepseek.py` |
| `layers/attention.py` | `deepinfer/model_executor/models/v4_modules/mla_attn.py` + FlashMLA |
| `layers/moe.py` | `deepinfer/model_executor/models/v4_modules/moe.py` + DeepEP |
| `layers/sampler.py` | `deepinfer/model_executor/layers/sampler.py` |
| `utils/loader.py` | `deepinfer/model_executor/model_loader.py` |

注：DeepInfer 现服务于 DeepSeek-V4 家族，其注意力与专家系统由 V2/V3 的 MLA + MoE 演进而来。本项目以 V2/V3 家族检查点为载体复刻这条主线，因为同构的小模型（如 V2-Lite）可以在普通设备上真正跑起来验证理解；引擎侧的调度、KV 管理与采样链路与版本无关。

## 适合继续练手的扩展点

- 把 `MLAAttention` 的潜空间注意力换成真正的 FlashMLA 内核，对照两者在显存访问上的差异；
- 在 `ModelRunner.run` 之上加一层草稿-验证循环，即是 DSpark/MTP 投机解码的骨架；
- 在 `Gate` 的专家循环处按 `expert_parallel_size` 切分专家并插入 all-to-all，即是 DeepEP 的位置；
- 给 `BlockManager` 加块淘汰策略，把前缀缓存做成真正的 LRU KV 缓存池。

## 致谢与引用

衷心感谢 DeepSeek 将 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 开源——把支撑其大规模线上服务的推理框架完整公开，极大地降低了普通开发者学习工业级推理系统的门槛。本项目正是在这一开源成果之上、为学习其设计而做的简化再实现，组织方式同时借鉴了 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。DeepInfer 的设计与引用信息如下：

```bibtex
@misc{deepinfer2026,
  title = {DeepInfer: High-performance, large-scale inference framework for DeepSeek models},
  author = {DeepSeek-AI},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/deepseek-ai/DeepInfer}}
}
```

## License

Apache License 2.0，见 [LICENSE](LICENSE)。
