# Nano-DeepInfer

一个极简、可读的 [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) 教学实现，仿照 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的做法：只保留推理框架的核心链路，用约 1600 行纯 PyTorch 代码把 DeepInfer 的请求生命周期、调度、KV cache 与 DeepSeek 标志性的模型结构串起来——V2/V3 的 MLA + MoE，以及 V4 的稀疏注意力 + mHC + Engram 三大新结构，方便新手顺着一条主线读懂一个生产级推理框架是怎么工作的。

原版 DeepInfer 是 DeepSeek 支撑大规模线上服务的推理框架，包含 4 万余行 Python、C++/TileLang 内核、DeepEP/DeepGEMM/FlashMLA 等厂商级优化与 DP/EP/CP 并行。本项目在保持核心架构与链路完全对齐的前提下，用纯 PyTorch 复刻了其中的关键数学变换与调度决策，使其可以在一台普通机器上读完、跑通、改得动。

## 性能基准（RTX 3060，同条件对照）

在双 RTX 3060 12GB 上，用 3.4 亿参数的同构模型（MLA+MoE，vocab=256）与同条件 prompt（64 token）实测：

| 负载 | nano-deepinfer | nano-vllm | 相对吞吐 |
|---|---|---|---|
| 8 条序列 × 256 token | 1440 tok/s | 1891 tok/s | 76% |
| 32 条序列 × 128 token | 4203 tok/s | 4597 tok/s | 91% |

- 32 条时已接近 nano-vllm（差距 <10%），瓶颈在 prefill 阶段；decode 步纯图重放上限约 5600 tok/s，已高于 nano-vllm 的 decode 吞吐。
- 8 条时差距主要来自 prefill 的小算子链——这与原版 DeepInfer 依赖 FlashMLA/DeepGEMM 的场景一致，本项目的定位是让读者看清这些优化的数学本质，而非替代它们。

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
                  ├─ MoE            分组受限 topk 路由 + 共享专家    layers/moe.py
                  └─ V4 分支（config 含 index_topk 时启用）
                      ├─ V4SparseAttention  索引器稀疏 MLA   layers/attention_v4.py
                      ├─ HCMix / HCHead     多路超连接 mHC   layers/hc.py
                      └─ Engram             n-gram 条件记忆  layers/engram.py
```

## 链路详解：一个 token 从请求到输出的完整旅程

下面按时间顺序讲解一次 `llm.generate(["你好"], SamplingParams(max_tokens=256))` 内部发生的每一件事。每一步都标注了对应的代码位置，建议对照源码阅读。

### 第 0 步：引擎初始化（`LLM(...)`，只发生一次）

```python
llm = LLM("deepseek-ai/DeepSeek-V2-Lite-Chat")
```

1. 读模型的 `config.json`，解析出 MLA/MoE 的全部维度参数（`config.py` 的 `ModelConfig`）；
2. 搭建 `DeepseekForCausalLM`，从 safetensors 加载权重——FP8 检查点在这一步按 128×128 分块反量化成 bf16（`utils/loader.py`）；
3. **分配 KV cache**：按显存预算算出物理块数，申请 `num_blocks × 16槽 × 288维 × 层数` 的显存池（`model_runner.py` 的 `_allocate_kv_cache`）。注意 288 = `kv_lora_rank(256) + qk_rope_head_dim(32)`，与注意力头数无关——这是 MLA 省显存的关键；
4. `Scheduler` 拿到块数，建 `BlockManager` 管理这个池子。

### 第 1 步：请求进入（`llm_engine.py` 的 `add_request`）

文本被分词器转成 token id 列表，包装成一个 `Sequence` 对象——它记录这条请求的一切：token 序列、`block_table`（逻辑块→物理块的映射，此时为空）、已缓存 token 数、采样参数。然后放入 `Scheduler.waiting` 队列等待调度。

### 第 2 步：调度决策（`scheduler.py` 的 `schedule`，每个 step 开头执行一次）

调度器每一步只回答一个问题：**这一步算谁、算多少？** 规则只有两条：

- **prefill 优先**：`waiting` 里有新请求时，本步是 prefill 步。逐个询问 `BlockManager.can_allocate`：前缀缓存能命中几个整块？空闲块够不够装下整条序列？命中部分的 prefill 直接跳过；预算（`max_num_batched_tokens`）不够算完整个 prompt 时，只调度前一段（**分块 prefill**），剩下的下个 step 继续；
- **无新请求才 decode**：`waiting` 空了，本步是 decode 步——`running` 中每条序列各推进 1 个 token。若某条序列需要新块而空闲块不足，**抢占**队尾序列：释放它的全部块、退回 `waiting`，之后从头重算（用重算换显存，实现简单且不死锁）。

### 第 3 步：KV cache 分页与前缀缓存（`block_manager.py`）

KV cache 像操作系统的内存一样分页管理：

- 物理池切成固定大小的块（默认每块 16 个 token 槽位），每条序列持有一张 `block_table`，记录自己的第 i 个逻辑块放在哪个物理块上——序列逻辑上连续，物理上离散；
- **前缀缓存**：每个写满的块计算链式哈希 `(父块哈希, 本块token)` 登记进哈希表。新请求若与已有序列共享前缀（比如相同的 system prompt），`can_allocate` 会命中这些块：引用计数 +1 直接复用，命中部分不用重算 prefill。

### 第 4 步：prefill 执行（`model_runner.py` 的 `_run_prefill_batched`）

本步调度的所有序列被**填充成矩形** `[B, S]`（B 条序列、S 为本步最长调度长度）一次算完：

- 每行真实 token 的 KV 槽位指向自己的 `block_table`；填充行的槽位全部指向一块公共"草稿区"，其输出算完即弃——这样既不用变长 kernel，又不会污染缓存；
- 一次前向后，取每行最后一个真实位置的 logits 送去采样，得到每条序列的第 1 个生成 token。

### 第 5 步：模型前向内部（`models/deepseek.py`，每层 DecoderLayer 做两件事）

```text
x = x + MLA(RMSNorm(x))        # 注意力子层
x = x + MoE(RMSNorm(x))        # 前馈子层（前 first_k_dense_replace 层是普通 MLP）
```

**MLA 注意力**（`layers/attention.py`，详见下文"值得细读的实现点"）：

1. 输入经**一次融合 GEMM**同时得到 q 的低秩投影与 kv 投影（q_a 与 kv_a 权重已拼接）；
2. kv 侧分出 `c_kv`（256 维潜向量）与 `k_pe`（32 维 RoPE 键），拼接后写入 KV cache 的本 token 槽位——**缓存里只有这 288 维，没有各头的 K/V**；
3. q 侧升维后分两段：nope 段与 `kv_b_proj` 的 K 部分相乘得到 `q_abs`（把上投影"吸收"进 Q），rope 段加位置编码；
4. 注意力直接在潜空间进行：`q_abs·c_kv + q_pe·k_pe` 得分数，softmax 加权 `c_kv`，最后才过 V 部分与 `o_proj` 回到隐藏维度。

**MoE 前馈**（`layers/moe.py`）：

1. `Gate` 对每个 token 打分（sigmoid + 校正偏置，即 DeepSeek 的无辅助损失路由），先按"组内 top2 之和"选出几个专家组，再在组内做 topk 选出 4 个专家；
2. 16 个专家的权重预先堆叠成大三维张量，一次批量矩阵乘算完所有专家，再按路由权重加权求和——无逐专家循环、无动态形状，因此 decode 可以被 CUDA graph 捕获；
3. 另有 1 个共享专家（普通 MLP）无条件参与，输出与路由专家相加。

### 第 6 步：采样（`layers/sampler.py`）

拿到 logits 后：全批贪心时只做一次 `argmax`；否则逐行做 温度缩放 → top-k 截断 → top-p 截断 → `multinomial`。指定 `seq_seed` 时用独立随机发生器，保证同参数结果可复现。

### 第 7 步：后处理（`scheduler.py` 的 `postprocess`）

采出的 token 追加进序列，然后判断结束条件：`max_tokens` 达到 / 采出 EOS / 命中 `stop_token_ids`。

- 结束的序列：释放全部物理块（引用计数归零的块归还空闲池）；
- 未结束的序列：留在 `running` 等下一个 decode 步；
- 顺手把本步刚写满的块登记进前缀缓存哈希表——后续同前缀请求因此受益。

### 第 8 步：decode 循环（`model_runner.py` 的 `_run_decode_graph`）

接下来的每个 step 重复"调度 → 执行 → 采样 → 后处理"，区别在于输入形状完全静态（每条序列只有上一步采出的 1 个 token，即 `[B]`），因此整个前向被 **CUDA graph** 按 `(批大小, 块数桶)` 捕获，之后只更新输入缓冲、整条图一次重放——这就是 decode 快的原因。有人结束、有新序列 prefill 完毕加入时批大小变化，切换到对应的图即可（**连续批处理**）。

### 第 9 步：输出

所有序列结束后主循环退出，token id 经分词器解码成文本，连同 `token_ids`、`num_prompt_tokens` 一起返回。

```text
 step 0        step 1        step 2 ...      step N
┌──────────┐  ┌──────────┐  ┌──────────┐   ┌──────────┐
│ prefill  │  │ decode   │  │ decode   │...│ decode   │
│ 8条×64tok│→ │ 8条×1tok │→ │ 8条×1tok │   │ 全部结束 │
└──────────┘  └──────────┘  └──────────┘   └──────────┘
   eager       CUDA graph    CUDA graph
```

## 保留了什么，简化了什么

| DeepInfer 的设计 | 本项目 | 说明 |
|---|---|---|
| 离线 `LLM` + `SamplingParams` 入口 | ✅ 保留 | API 形态与原版 `deepinfer/entrypoints/llm.py` 对齐 |
| 引擎主循环与异步调度 | ✅ 简化 | 保留准入/批处理/抢占决策，去掉 CPU 调度与设备执行的异步重叠 |
| 连续批处理、prefill/decode 两阶段 | ✅ 保留 | 额外实现分块 prefill，长 prompt 可分多个 step 灌入 |
| KV cache 管理（原版未开源） | ✅ 补齐 | 分页块管理 + 链式哈希前缀缓存，思路与 vLLM 系一脉相承 |
| MLA（多头潜注意力） | ✅ 保留 | KV cache 只存 `kv_lora_rank + qk_rope_head_dim` 维潜向量，注意力以吸收式在潜空间计算（`kv_b_proj` 的 K 部分吸收进 Q、V 部分在加权后作用），不还原完整 K/V |
| MoE 路由 | ✅ 保留 | sigmoid 打分、校正偏置（noaux_tc 无辅助损失）、分组受限 topk、共享专家、前几层 dense；专家权重堆叠成批量矩阵乘，无逐专家循环、无动态形状，decode 可被 CUDA graph 捕获 |
| V4 稀疏注意力 | ✅ 保留 | 索引器对全量缓存打分选 topk，主注意力只算选中位置；q 直接与潜向量内积（FlashMLA 布局），V 侧分组低秩；含 SWA 滑窗与 attention sink 选项 |
| V4 mHC（多路超连接） | ✅ 保留 | 残差扩成 m 路 trunk，每子层 fn-GEMM + PolyNorm + sigmoid/sinkhorn 生成混合系数，pre 混单路、post 注回；含收尾 HCHead |
| V4 Engram | ✅ 保留 | n-gram 哈希查稀疏嵌入表（随机奇数乘法因子 + 素数表长），fused 投影后与 HC trunk 门控融合；原版跨卡 DeepEP 取表简化为本地查表 |
| 确定性采样（`seq_seed`） | ✅ 保留 | 温度/top-k/top-p 与原版同名字段 |
| DeepGEMM / FlashMLA / TileLang 内核 | ❌ 简化为纯 PyTorch | 不写自定义内核，但保留其核心变换：MLA 吸收式潜空间注意力（`kv_b_proj` K 吸收进 Q、V 加权后作用）、q/kv 输入投影融合为一次 GEMM、MoE 专家权重堆叠批量矩阵乘；数学与原版等价 |
| V4 索引器 FP4 量化、跨层共享、DFlash、YOCO | ❌ 省略 | 属内核与特性层优化，不影响对三大新结构主线的理解 |
| DP / EP / CP 并行、EPLB | ❌ 省略 | 单机单卡路径；`Gate` 与专家的组织方式即为 EP 切分的天然位置 |
| DSpark / MTP 投机解码 | ❌ 省略 | 原版在 `ModelRunner` 之上做草稿-验证，接口位置见下方扩展点 |
| SWA bounded replay、结构化输出 | ❌ 省略 | 属于特性层，不影响主链路理解 |
| CUDA graph | ✅ 保留 | decode 走整批静态形状前向并按 (批量, 长度桶) 捕获重放；prefill 与 CPU 走 eager，`enforce_eager=True` 可关闭 |
| FP8 检查点 | ✅ 保留 | 加载时按 128×128 分块 scale 反量化，见 `utils/loader.py` |

## 支持的模型

只要检查点是 **DeepSeek-V2 / V3 架构**（MLA + MoE）且为 HuggingFace 格式（`config.json` + safetensors），即可直接加载推理：

| 家族 | 代表检查点 | 规模 | 说明 |
|---|---|---|---|
| DeepSeek-V2 | `deepseek-ai/DeepSeek-V2-Lite(-Chat)` | 16B 总参 / 2.4B 激活 | 最小的官方 MLA+MoE 检查点，教学演示首选 |
| DeepSeek-V2 | `deepseek-ai/DeepSeek-V2(-Chat)` | 236B / 21B 激活 | |
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3(-0324)` | 671B / 37B 激活 | |
| DeepSeek-V3 | `deepseek-ai/DeepSeek-R1(-0528)` 及同架构蒸馏版 | 671B / 37B 激活 | R1 与 V3 同架构 |
| DeepSeek-V4 | 同构 `config.json`（含 `index_topk` 等字段） | — | 稀疏注意力 + mHC + Engram 全链路已实现；官方尚未发布公开检查点，可先按 V4 字段构造同构模型跑通链路 |
| FP8 版 | 上述模型的 FP8 检查点 | 权重视体积减半 | 加载时自动按 128×128 分块反量化，无需离线转换 |

两类补充说明：

- **同构自定义小模型**：模型结构完全由 `config.json` 字段驱动，把层数、维度、专家数改小保存即可构造任意大小的 MLA+MoE 模型（随机权重也能跑通全链路）。上方性能基准使用的 3.4 亿参数模型就是这样构造的，适合无大显存时验证理解与测速；
- **不支持的模型**：非 DeepSeek 架构（Qwen、Llama 等 dense/GQA 模型，请用 nano-vllm）。

显存门槛只取决于权重体积：bf16 权重 ≈ 2 字节/参数，V2-Lite 约 31GB 需 24GB 以上的卡；FP8 版减半。同构小模型任意显卡乃至纯 CPU 均可运行。

## 适配硬件

本项目只有一个版本，不需要为不同显卡分档。原版 DeepInfer 要求 NVIDIA SM100 / 华为 Ascend 950，是因为它依赖 FlashMLA、DeepGEMM 等只为新架构编写的内核；本项目用纯 PyTorch 复刻这些内核的数学、CUDA 上统一用 bfloat16 计算，因此对卡型没有特殊要求：

| 环境 | 要求 |
|---|---|
| NVIDIA GPU | 计算能力 SM80 及以上均可，包括 RTX 3060（SM86）/3070/3090、RTX 40 系、A100/H100 以及 SM100 数据中心卡；有卡自动使用 |
| CPU | 无显卡时自动退回 CPU（float32），适合小模型对照实验 |

曾短暂提供过一个 `rtx3060` 配置档（float16 + 收紧的批量上限），在 RTX 3060 上与默认 bfloat16 配置做了同负载对照实测（3.4 亿参数同构模型、8 条序列 × 256 token、交替顺序多轮）：默认档 89.9 tok/s、3060 档 87.0~87.9 tok/s，差异在测量噪声范围内——因为两者共用同一套算子实现，分档并无性能收益，故已删除，只保留实测不慢且数值范围更好的默认配置。

当前版本在 RTX 3060 上的实测吞吐见上方"性能基准"节：32 条并发时达到 4203 tok/s（同条件 nano-vllm 为 4597 tok/s），decode 步纯图重放上限约 5600 tok/s。

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

无可用显卡时自动退回 CPU 运行（`device="cpu"`），适合对照代码做小模型实验。可加载的模型清单见上方"支持的模型"一节。

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
│  ├─ attention_v4.py      # V4 稀疏注意力：索引器 topk 选路 + 潜空间主注意力
│  ├─ hc.py                # V4 mHC：多路残差 trunk 的 pre/post 混合与 HCHead
│  ├─ engram.py            # V4 Engram：n-gram 哈希查表 + 门控注入
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

## V4 三大新结构

`config.json` 中出现 `index_topk` 时模型自动走 V4 路径（`ModelConfig.is_v4`），与 V2/V3 共用同一套引擎。三者对应原版 `deepinfer/model_executor/models/v4_modules/`。

**1. 稀疏注意力（`layers/attention_v4.py`）。** V3 的 MLA 让每个 query 与全部缓存潜向量计算；V4 在其上加了一个"索引器"：每个 query 先从 `q_B` 输出尾部分裂出几头低维 index q，与全量缓存潜向量投影出的 index k 打分（`relu(iq·ik)` 再按头加权求和），只保留 `index_topk` 个位置参与主注意力。长上下文下主注意力的计算量从 O(L) 降为 O(topk)。三个细节值得对照原版看：index k 由主潜向量投影而来（`use_kproj` 变体），不额外占缓存；RoPE 只加在尾部 `qk_rope_head_dim` 维；V 侧是分组低秩变换（`o_groups`/`o_lora_rank`），加权潜向量先过 Wv 再回 hidden。可选 SWA 滑窗（`window_size`）与 attention sink（softmax 多一列恒零"垃圾桶"位）也已实现。

**2. mHC 多路超连接（`layers/hc.py`）。** 把单路残差 `x` 扩成 m 路 trunk `[n, m, h]` 在层间流动。每个子层（attn/ffn）配一组混合参数：trunk 拉平过一次 GEMM + PolyNorm（rms 归一），切出 pre/post/comb 三段系数——pre 经 sigmoid 加权把 m 路混成单路作为子层输入，子层输出 y 按 post 系数注入，trunk 自身按 comb 矩阵混合；comb 经 sinkhorn 迭代成双随机矩阵，保证多路混合时信息守恒（对应原版 `mhc_ops`）。初始化让残差近似恒等传递，混合交给 fn 学习。层末由 `HCHead` 把 trunk 混回单路接 lm_head。教学版未做原版的 hc_shifted 流水线变体。

**3. Engram n-gram 条件记忆（`layers/engram.py`）。** 一个可学习的"查表记忆"：每个位置取以当前 token 结尾的 n-gram 窗口（2 阶、3 阶……），按每层随机奇数乘法因子做多项式哈希、对互不相同的素数表长取模，查出稀疏嵌入拼接后过 `fused_kv_projection`，再与 HC trunk 门控融合（门由 hidden 与嵌入共同算 sigmoid）。哈希乘法因子由 `seed + 10007 × layer_id` 的发生器导出、表长全局取素数去重——这两个约定与原版 `engram_metadata.py` 完全一致。原版的跨卡 DeepEP 取表与 CompressedTokenizer 词表压缩在此简化为本地查表与原始词表。

**V4 路径的性能注记。** 教学版用纯 PyTorch 表达稀疏打分与多路混合：同尺寸随机模型实测（RTX 3060，8 条 × 128 token）V4 路径吞吐约为 V3 路径的 47%——原版靠 TileLang 内核（topk 内核、融合 mHC GEMM、FP4 索引缓存）才获得实际加速，这里保留的是结构与数学，读代码时重点看 `_sparse_mask` 的打分选路、`HCMix.pre/post` 的两次混合与 `NgramHash.hashes` 的哈希约定。

## 与 DeepInfer 的对应关系

| 本项目文件 | DeepInfer 对应位置 |
|---|---|
| `llm.py` | `deepinfer/entrypoints/llm.py` |
| `engine/llm_engine.py` | `deepinfer/engine/llm_engine.py` |
| `engine/scheduler.py` | `deepinfer/engine/` 内的异步调度器（同步简化版） |
| `engine/model_runner.py` | `deepinfer/worker/model_runner.py` |
| `models/deepseek.py` | `deepinfer/model_executor/models/deepseek.py` |
| `layers/attention.py` | `deepinfer/model_executor/models/v4_modules/mla_attn.py` + FlashMLA |
| `layers/attention_v4.py` | `deepinfer/model_executor/models/v4_modules/{attn,index_attn,mla_attn}.py` 的稀疏主线 |
| `layers/hc.py` | `deepinfer/model_executor/models/v4_modules/{hc,mhc_ops,hc_blocks}.py` |
| `layers/engram.py` | `deepinfer/model_executor/models/v4_modules/{engram,engram_metadata}.py` |
| `layers/moe.py` | `deepinfer/model_executor/models/v4_modules/moe.py` + DeepEP |
| `layers/sampler.py` | `deepinfer/model_executor/layers/sampler.py` |
| `utils/loader.py` | `deepinfer/model_executor/model_loader.py` |

注：DeepInfer 现服务于 DeepSeek-V4 家族。本项目同时复刻 V2/V3 的 MLA+MoE 主线与 V4 的三大新结构——前者有公开检查点（如 V2-Lite）可在普通设备上跑起来验证理解，后者按 `config.json` 字段构造同构模型即可走通全链路；引擎侧的调度、KV 管理与采样链路与版本无关。

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
