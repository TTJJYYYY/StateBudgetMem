# StateBudgetMem：端侧 MemoryBank 复现审计与差距分析

审计日期：2026-07-11  
审计对象：当前工作树实际代码、测试、配置、结果文件，以及 `C:\Users\admin\Downloads\Memorybank.pdf`（AAAI 2024 MemoryBank 原论文）

## 1. 执行摘要

当前项目最准确的定位是：**C. 部分复现**，更具体地说，是“MemoryBank 核心存储/检索/遗忘思想的可离线重新实现，加上 StateBudgetMem 自己的版本化、双视图、路由和预算实验框架”。它还不是完整论文复现，也还不能严格称为已经完成“端侧 MemoryBank baseline”。

建议完成度估计：

- MemoryBank 核心机制复现完成度：**58%**。已有三类存储容器、FAISS dense retrieval、`R=exp(-t/S)`、初始强度、召回强化、last-accessed 更新和 prompt 组装；但摘要与人格演化没有由对话自动生成，论文实验没有复跑，遗忘处理含非论文规则。
- 端侧 MemoryBank baseline 完成度：**46%**。核心路径设计为 local-only，hash embedding 可离线，FAISS/本地 sentence-transformer 可接入，也有资源采集脚本；但本机缺少 numpy/FAISS/pytest，正式本地闭环未在本次环境跑通，本地 LLM 提取/摘要/人格/回答没有后端，隐私边界与端侧规模实验不足。

对外结论：

- “已经实现 MemoryBank baseline”：**不宜无限定地说**；可说“已实现可测试的 MemoryBank-inspired 离线机制原型”。
- “已经完成 MemoryBank 核心机制复现”：**不宜说**；关键的自动摘要、人格演化和论文实验仍缺。
- “已经在端侧复现 MemoryBank”：**不能说**；目前只证明了若干本地组件和脚本设计，未形成经本机实跑验证的完整端侧闭环。
- 最稳妥表述：**“我们已完成 MemoryBank 的本地存储、向量检索、Ebbinghaus retention 与召回强化等核心子机制的离线重新实现，并搭建了端侧评测脚本；尚未完成论文数据、自动摘要/人格演化、原指标人工评测和本地 LLM 全闭环。”**

## 2. 原论文机制与实验基准

原论文的 MemoryBank 有三根支柱：memory storage、memory retrieval、memory updating。存储包含按时间记录的多轮对话、每日事件摘要及其全局摘要、每日人格/情绪分析及其全局人格画像。每日与全局内容由 LLM 按论文 prompt 逐级综合，不只是预留字符串字段。

检索把每轮对话和事件摘要视为 memory piece，使用可替换 encoder 预编码，通过 FAISS 做 dense retrieval；论文实现中英文分别使用 MiniLM 与 Text2vec。召回后，相关记忆与 global portrait、global event summary 一同进入生成 prompt。

更新采用简化 Ebbinghaus 模型 `R = exp(-t/S)`。首次提及时 `S=1`；一次记忆在对话中被召回后 `S += 1`，并将 elapsed time 重置为 0。论文只描述按 retention 概率遗忘以及召回强化，没有规定“遗忘后 strength 减半”或固定阈值硬删除。

论文定量实验是 15 个虚拟用户、10 天对话、中英各 97 个 probing questions（共 194）。评价由人工标注：Memory Retrieval Accuracy（0/1）、Response Correctness（0/0.5/1）、Contextual Coherence（0/0.5/1）和三模型 Ranking Score。论文还比较 ChatGPT、ChatGLM、BELLE 版本，并对开源模型使用 38k 心理对话、LoRA rank 16、3 epochs、A100 的 SiliconFriend 调优。后者属于完整 SiliconFriend 复现，不是仅复现 MemoryBank 数据结构的必要条件，但若宣称完整论文复现则不可省略。

## 3. 论文机制对齐表

| MemoryBank 机制 | 论文实现 | 当前项目实现 | 代码位置 | 测试 | 实验日志 | 对齐等级 | 关键缺口 |
|---|---|---|---|---|---|---|---|
| Memory Writing | 持续记录人机多轮对话及时间 | `store_dialog`/`store` 可写入本地内存 | `src/statebudgetmem/baselines/memorybank/system.py` | 有 | 仅 paper snapshot | 基本复现 | 默认不是持久数据库；正式写入闭环未实跑 |
| Memory Extraction | 论文以完整对话与 LLM 摘要/人格分析形成记忆 | 可选 `llm_extractor`；无后端时每条消息直接成为 memory | 同上 `_extract` | 部分 | 无正式结果 | 近似实现 | 流程更接近 Mem0 风格事实抽取，并非论文流程 |
| Memory Storage | dialog、daily/global event summary、daily/global personality | dialog/summary memory + 两个全局字符串 | `system.py`, `paper_storage.py` | 有 | `results/memorybank/paper_storage/` | 部分复现 | 缺 daily personality 层；fixture 注入不等于演化 |
| Memory Representation | 每轮对话/事件摘要为可编码 memory piece | `MemoryPiece`，含 id/type/time/strength/vector | `core/online.py`, `system.py` | 有 | snapshot | 基本复现 | 项目额外字段不影响；全局对象未统一索引 |
| Memory Retrieval | dual-tower dense encoder + FAISS | normalized embedding + FAISS inner product | `system.py` | 有（含 fake/hash） | paper snapshot | 基本复现 | 未用论文中英 encoder 跑正式结果 |
| Semantic Similarity / Embedding | English MiniLM，Chinese Text2vec，可替换 | 默认 all-MiniLM-L6-v2；hash CI；可注入模型 | `embeddings.py`, `system.py` | 有 | hash 结果声明但正式目录缺失 | 部分复现 | 中文 Text2vec 与真实 sentence-transformer 实验缺失 |
| Memory Strength | 首次 `S=1` | `MemoryPiece` 默认强度需由核心模型初始化，store 后测试覆盖 | `core/online.py`, `system.py` | 有 | 日志脚本存在 | 基本复现 | 本次因依赖缺失未实跑验证 |
| Reinforcement after Recall | recall 后 `S += 1` | selected Top-K 后 `mem.strength += 1` | `system.py:493` | 有 | 正式日志目录缺失 | 基本复现 | 所有 selected 均强化，是否等同“对话中实际使用”未验证 |
| Last-accessed Update | recall 后 elapsed time 重置为 0 | `last_accessed = now` | `system.py:494` | 有 | 脚本计划记录 before/after | 基本复现 | 未把“实际用于回答”与“被检索”分开 |
| Forgetting Curve | `R=exp(-t/S)` | 同式；t 为可配置时间单位 | `system.py:528-530, 743-745` | 有 | 当前正式遗忘日志缺失 | 基本复现 | 论文未明确固定 24h 单位，需敏感性分析 |
| Retention Calculation | elapsed since learning/last recall 与 strength | 基于 `last_accessed` 和 `decay_interval_hours` | `system.py` | 有 | raw schema 有字段 | 基本复现 | 时间单位属实现假设，应明确为近似 |
| Forgotten Memory Handling | 按 retention 概率选择性遗忘，细节探索性 | 默认仅标记；可阈值排除；update 时 `strength *= 0.5` | `system.py:679-758` | 有 | 无本次成功日志 | 近似实现 | 阈值 hard exclusion 和减半均非论文规则 |
| Memory Update | 论文主要指 forgetting/reinforcement 与层级摘要更新 | 另有 ADD/MERGE/NOOP 语义更新 | `system.py` | 有 | 无 | 近似实现 | 语义 merge 是项目扩展；未实现论文持续摘要更新 |
| User Profile / Personality Evolution | 每日分析，再综合为全局画像 | `update_user_portrait` 直接覆盖字符串；数据集预填 | `system.py:671`, `datasets.py` | 只测赋值 | 无演化日志 | 尚未实现 | 没有从每日对话生成、累积、更新的算法 |
| Reflection | 论文没有独立“reflection”模块；摘要/人格综合承担反思功能 | 无独立 reflection | 无 | 无 | 无 | 不适用/部分 | 不应把论文没有命名的模块硬列为缺失；其综合能力仍未实现 |
| Prompt Construction | relevant memory + global portrait + global summary + query | `build_augmented_prompt` 输出结构化 sections | `system.py:764+`, `agents.py` | 有 | paper snapshot | 基本复现 | 未用论文模型生成并做人评 |
| LLM Dependency | 摘要、画像、回答；ChatGPT/ChatGLM/BELLE | optional OpenAI-compatible answer path；离线用 fixture/template | `agents.py`, `evaluator.py`, CLI | mock/部分 | 历史结果非本次复跑 | 部分复现 | 无真正本地 LLM adapter；完整 demo 在线模式会传数据到 API |
| Evaluation Dataset | 15 users × 10 days，194 bilingual probes | 自建 5 users × 5-10 days，约 50 probes；另有 demo/Memora adapter | `data/memorybank_reproduction/`, `datasets.py` | schema 有 | phase1 报告宣称结果但目录缺失 | 近似实现 | 不是论文数据规模、题目或语言配比 |
| Evaluation Metrics | 四项人工指标 | keyword proxy、gold retrieval、stale/resource metrics | `metrics.py`, `evaluator.py` | 有 | 部分历史结果 | 近似实现 | proxy 不等于人工 correctness/coherence/ranking |
| Paper Experiment Setup | 3 LLM variants、双语、人工评价；开源模型心理数据 LoRA | 未复现 | 无统一 runner | 无 | 无 | 尚未实现 | 无模型对齐、人工标注、LoRA、194 题结果 |
| Paper Results | 论文 Table 2 六组结果 | 无可比较复现表 | 文档仅列项目 proxy | 无 | 无 | 尚未实现 | 不能验证或反驳论文结果 |

## 4. 当前项目实际能力

MemoryBank 包的实质能力包括：本地内存对象存储、pickle/FAISS 持久化接口、可注入 embedding、dense retrieval、语义分数与时间/强度组合排序、召回强化、遗忘预览与事件日志、可选遗忘候选排除、prompt sections、固定数据适配器、mock/template 回答及资源记录脚本。

TF-IDF baseline、versioning、views、routing 和 evaluation 均有真实代码与测试，而非空目录。它们主要是 StateBudgetMem 方法或对照框架，不应计入 MemoryBank 论文复现百分比。尤其 versioning 的 `SUPERSEDE/TEMP_INVALIDATE/RESTORE`、current/history views 和 query routing 是本项目研究贡献方向，不能反向当作原论文已复现的证据。

README 与实际状态存在两类不一致：一是 README 仍称 views 为 next stage，而仓库已有 views runner、方法、测试和结果；二是阶段报告称 phase1、ondevice、budget sweep、forgetting logs 已有完整结果，但本次工作树的 `results/memorybank/` 只看到 `obsolete_analysis.json`、`paper_storage/` 和 `reproduction_storage/summary.json`，没有报告引用的 phase1/ondevice/budget_sweep/forgetting_demo 正式目录。因此报告中的数字只能视为历史/外部运行陈述，不能视为当前仓库可审计结果。

## 5. 本地运行验证

运行环境：Windows，PowerShell，系统 Python 3.13.5；项目未安装到该 Python。

| 命令 | 结果 | 输出/证据 | 失败原因 |
|---|---|---|---|
| `pytest -q` | 失败 | 无测试执行 | 当前 PATH 无 pytest |
| bundled Python `-m pytest -q` | 失败 | 无测试执行 | bundled Python 同样无 pytest |
| `PYTHONPATH=src python -m statebudgetmem.cli --help` | 成功 | 显示 run/route/evaluate-memorybank/analyze-staleness | 无 |
| `python tools/memorybank/run_phase1_baseline.py --smoke --run-id audit_20260711` | 失败 | 未生成结果 | 核心可选依赖 numpy 缺失 |
| `python tools/memorybank/run_memorybank_forgetting_demo.py --run-id audit_20260711` | 失败 | 未生成结果 | numpy 缺失 |
| `python tools/memorybank/run_budget_sweep.py --quick --run-id audit_20260711` | 失败 | 未生成结果 | numpy 与 faiss-cpu 缺失 |

结论必须区分：CLI 成功说明最小包代码可导入；三个 MemoryBank 脚本失败是 **memorybank optional dependencies 未安装**，不是代码功能不存在；pytest 失败是 **test dependency/environment**。但在端侧复现审计中，“本机尚未按文档安装后跑通”仍意味着当前不能提供新的端侧成功证据。

本次没有联网下载依赖，也没有改动代码或手工补结果。工作树原本已有大量未跟踪文档与结果，审计未覆盖或删除这些用户文件。

## 6. 端侧复现评价

### 6.1 本地独立运行链路

- 可本地完成（从代码设计判断）：固定/规则化 writing、内存与 pickle 存储、hash 或本地 sentence-transformer embedding、FAISS index、retrieval、strength/forgetting 更新、versioning、views、rule routing、离线 proxy evaluation。
- 尚无完整本地实现：LLM memory extraction、daily/global summary 自动生成、daily/global personality 自动演化、自然语言回答生成。
- 可选择云端：OpenAI-compatible extractor/router/judge/answer；一旦启用，相关对话、检索记忆或评测内容可能进入云端 prompt。
- 不依赖云数据库；默认数据与结果路径均在本地工作树。sentence-transformer 首次按模型名加载可能需要网络下载，只有预下载模型或本地路径才是真正离线。

### 6.2 轻量模型

接口上可以注入本地 encoder/LLM，但仓库没有 llama.cpp、ONNX、MLC、Ollama 等本地 4B 以下生成后端，也没有端侧量化模型配置和验证。hash embedding 只是确定性 smoke encoder，不是可用于论文结论的语义模型。MemoryBank 的非生成核心可以本地跑；完整交互 demo 的高质量 extraction/summary/profile/answer 仍没有本地闭环。

### 6.3 本地存储与隐私

memory metadata、embedding 与 FAISS index 可通过 pickle/FAISS 写到本地；实验 JSON/JSONL/CSV 也默认写入 `results/`。但缺少明确的数据保留/删除策略、敏感标签、字段级脱敏、云 prompt 审批或 local-only 强制开关。当前只能说“支持本地运行路径”，不能宣称严格 privacy-preserving。个人记忆默认 local-only 的产品约束尚未被策略和测试强制执行。

### 6.4 资源评估

脚本覆盖 retrieval latency、tracemalloc peak、估计 storage、index count、prompt token proxy 和 memory-count/top-k/threshold sweep，这是良好框架。但缺少真实 index 文件字节数、进程 RSS、CPU/GPU 利用、写入与 forgetting latency、真实 tokenizer token、真实语义 encoder/本地 LLM 的冷启动与稳态成本、多次重复及置信区间。资源曲线脚本本次也未成功运行。

端侧复现必需项：可重复安装锁定、本地 semantic embedding、完整 local-only 数据流、真实存储/index/RSS/latency、规模 sweep、隐私边界、本地生成方案或明确把生成排除出 baseline。加分项：GPU 能耗、移动端、量化对比、漂亮 UI。

## 7. 缺口分类与优先级

| 类别 | 缺口 | 必要性 | 工作量 | 优先级 | 影响端侧结论 | 本周期 |
|---|---|---:|---:|---:|---:|---:|
| A 论文机制 | 自动 daily/global event summary | 必须 | 中 | P0 | 是 | 是 |
| A 论文机制 | daily/global personality evolution | 必须 | 中 | P0 | 是 | 是 |
| A 论文机制 | 论文一致的 forgetting policy（区分原生与扩展） | 必须 | 小 | P0 | 是 | 是 |
| A 论文机制 | MiniLM/Text2vec 或明确等价 encoder | 必须 | 中 | P1 | 是 | 是 |
| B 实验 | 15×10、194 双语问题或可追溯同构数据 | 必须 | 大 | P0 | 间接 | 是 |
| B 实验 | 原四项人工标注与 ranking | 必须 | 大 | P0 | 否 | 是 |
| B 实验 | 三 LLM 变体/模型差异控制 | 完整论文复现必须 | 大 | P2 | 否 | 视周期 |
| B 实验 | 论文 Table 2 可比结果 | 必须 | 大 | P1 | 否 | 是 |
| C 端侧运行 | 可复现安装与依赖锁定 | 必须 | 小 | P0 | 是 | 是 |
| C 端侧运行 | 本地 semantic encoder 实跑 | 必须 | 小/中 | P0 | 是 | 是 |
| C 端侧运行 | 4B 以下本地 LLM adapter 或边界声明 | 必须 | 中 | P1 | 是 | 是 |
| C 端侧运行 | local-only 强制模式与网络禁用测试 | 必须 | 中 | P0 | 是 | 是 |
| C 端侧运行 | 敏感记忆云发送控制/脱敏 | 必须 | 中 | P1 | 是 | 是 |
| D 端侧评估 | RSS、磁盘 index、各阶段 latency | 必须 | 中 | P0 | 是 | 是 |
| D 端侧评估 | 规模与 budget-performance 曲线 | 必须 | 中 | P1 | 是 | 是 |
| D 端侧评估 | CPU/GPU/能耗 | 加分 | 中 | P2 | 否 | 可后置 |

## 8. 风险

最大科学风险是把“fixture 能进入三个字段”写成“三层记忆已复现”，以及把 keyword proxy 写成论文人工指标。第二个风险是 hash embedding 的低质量结果与 semantic encoder 结果混报。第三个风险是把 StateBudgetMem 自己的版本化/视图优势归入 MemoryBank，从而造成不公平 baseline。第四个风险是脚本存在但当前结果目录缺失，报告数字不可追溯。第五个风险是 local-capable 被表述成 local-verified 或 privacy-preserving。

## 9. Roadmap

### Phase 1：可验证的 MemoryBank baseline

冻结 paper-compatible profile；将项目扩展策略放入 separate profile。补齐自动分层摘要/人格演化，明确 strength/last-accessed/retention 事件，建立 194 题同构数据与人工评价协议，跑通 tests、smoke、formal 命令并提交机器可读结果。

### Phase 2：端侧运行闭环

提供离线 semantic encoder 资产说明与本地 4B 以下 LLM adapter（或明确 baseline 只评 retrieval）；加入 `local_only=true` 强制网络禁用，落盘 memory/embedding/index/metadata，建立敏感字段与云发送策略。

### Phase 3：端侧资源评价

在 100/500/1k/5k/10k memories 上重复测量写入、检索、遗忘、prompt 构造、RSS、磁盘字节和真实 token；报告均值、P50/P95、方差与硬件信息，画 accuracy-resource 曲线。

### Phase 4：接入 StateBudgetMem 改进

固定同数据、同 encoder、同 top-k/token/storage budget，比较 native MemoryBank、TF-IDF、MemoryBank+Versioning、+Views、+Routing 和完整 StateBudgetMem；报告 valid recall、stale retrieval/use、answer accuracy、latency、storage、tokens、peak RSS。

### Phase 5：最终 Demo

最后再展示 baseline、strength/forgetting timeline、current/history、routing、local-only 状态和资源面板。Demo 不作为论文复现证据，只展示已由实验验证的能力。

## 10. Top 15 TODO

| # | 任务目标 | 修改文件 | 验收标准 | 依赖 | 工作量 | 建议负责人 | 影响答辩 | 类型 |
|---:|---|---|---|---|---|---|---|---|
| 1 | 建立 paper-compatible 配置并隔离项目扩展 | `configs/`, `system.py` | 原生/扩展策略可一键切换，结果记录 profile | 无 | 1-2d | baseline 负责人 | 是 | 论文复现 |
| 2 | 固化可安装环境 | `pyproject.toml`, lock 文件, README | 新环境一条命令安装且 tests 可跑 | 1 | 1d | 工程负责人 | 是 | 端侧运行 |
| 3 | 补 daily event summary generator | 新 `summarization.py`, interfaces | 对固定对话确定性/本地生成并留 provenance | 本地 LLM | 3-5d | baseline 负责人 | 是 | 论文复现 |
| 4 | 补 global event synthesis | 同上 | 多日摘要可增量更新且测试通过 | 3 | 2-3d | baseline 负责人 | 是 | 论文复现 |
| 5 | 补 daily/global personality evolution | 新 `profile.py` | 每日与全局层均可追踪版本 | 本地 LLM | 3-5d | NLP 负责人 | 是 | 论文复现 |
| 6 | 校正 forgetting paper profile | `system.py`, tests | 论文模式无未声明 strength×0.5；事件字段完整 | 1 | 1-2d | baseline 负责人 | 是 | 论文复现 |
| 7 | 区分 retrieved 与 actually-used reinforcement | `agents.py`, `system.py` | 只强化进入回答上下文的记忆，含边界测试 | 6 | 2d | baseline 负责人 | 是 | 论文复现 |
| 8 | 准备 15×10、194 双语同构数据 | `data/memorybank_paper/`, docs | schema、来源、hash、标注说明完整 | 人工 | 5-8d | 数据负责人 | 是 | 论文实验 |
| 9 | 实现原论文人工评价表 | `evaluation/`, annotation docs | 四指标、双人标注、一致性、ranking 可导出 | 8 | 3-5d | 评测负责人 | 是 | 论文实验 |
| 10 | 跑本地 MiniLM/Text2vec 对齐实验 | configs/tools/results | 模型版本、缓存路径、双语结果可复现 | 2,8 | 2-3d | 实验负责人 | 是 | 论文实验/端侧 |
| 11 | 加 local-only 网络禁用守卫 | core/online, adapters, tests | local-only 模式下任何网络 adapter 均失败关闭 | 2 | 2-3d | 工程负责人 | 是 | 端侧运行 |
| 12 | 接入 4B 以下本地生成 adapter | interfaces + local adapter | extraction/summary/profile/answer 全离线 smoke | 模型资产 | 4-7d | 模型负责人 | 是 | 端侧运行 |
| 13 | 实测持久存储与 index | storage/tool/results | memory、metadata、embedding、FAISS 文件可恢复且字节准确 | 2,10 | 2d | 工程负责人 | 是 | 端侧实验 |
| 14 | 完整资源 benchmark | benchmark tool/config/results | 写入/检索/遗忘/RSS/P95/token/规模曲线 | 10,13 | 3-5d | 实验负责人 | 是 | 端侧实验 |
| 15 | 固定公平消融矩阵 | configs/experiment, tools/results | 六方法同预算、同数据、同 seed、同模型 | 8-14 | 4-6d | 研究负责人 | 是 | 我们的方法 |

## 11. 推荐的对外表述

如果明天向老师汇报，建议准确地说：

> 我们已经完成了 MemoryBank 若干关键子机制的本地重新实现：对话/摘要/画像存储容器、FAISS 向量检索、Ebbinghaus retention、召回后的 strength 与 last-accessed 更新，以及结构化 prompt 和端侧资源记录框架；同时，StateBudgetMem 的版本化、双视图和路由模块已有独立实现。我们尚未完成 MemoryBank 原论文级复现：自动的每日/全局摘要和人格演化仍主要由 fixture 代替，15 用户×10 天×194 双语问题及原四项人工指标未复跑，本地语义 embedding 与本地 LLM 的全闭环、隐私强制边界和正式资源曲线也没有在当前环境验证。因此目前应称为“MemoryBank-inspired 的部分核心机制离线原型”，而不是“已在端侧完整复现 MemoryBank”。

不应使用的夸大表述包括：“完整复现 MemoryBank 论文”“论文指标已复现”“三层记忆演化已完成”“完整端侧闭环已完成”“已证明隐私优势”。

## 12. 审计限制

本次严格以当前工作树为准。由于现有环境没有 pytest、numpy 和 faiss-cpu，未能执行测试套件和 MemoryBank 正式脚本；因此对相关代码只能给出静态实现与既有测试覆盖判断，不能标成“本次运行通过”。论文 PDF 已完成全文文本核对；本报告没有联网核对论文官方仓库实现细节，因而凡论文正文未规定的遗忘阈值、概率抽样和具体持久化行为，均不擅自归为论文要求。
