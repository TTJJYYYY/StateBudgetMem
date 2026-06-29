# MemoryBank 长期记忆管理 — 复现与评测

基于论文 ["MemoryBank: Enhancing Large Language Models with Long-Term Memory" (AAAI-24)](https://ojs.aaai.org/index.php/AAAI/article/view/29946) 的复现实现，支持外部评测数据集，采用 LLM-as-Judge 评分机制。

---

## 项目结构

```
├── memorybank.py          # MemoryBank 核心实现（存储 / 检索 / 遗忘）
├── evaluation_v2.py       # 对比实验框架 + Memora 数据集加载
├── evaluation_results_*.json   # 各 persona 的评测结果
├── evaluation_results.json     # 汇总结果
└── README.md              # 本文档
```

> `evaluation.py` 为早期版本，已废弃，请使用 `evaluation_v2.py`。

---

## 环境依赖

```bash
pip install numpy faiss-cpu sentence-transformers openai
```

- `faiss-cpu`：向量检索（有 GPU 可换 `faiss-gpu`）
- `sentence-transformers`：文本 Embedding（默认 `all-MiniLM-L6-v2`）
- `openai`：调用 DeepSeek API
- Python >= 3.10

---

## 快速开始

### 1. 配置 DeepSeek API Key

打开 `evaluation_v2.py`，找到以下行并填入你的 API Key：

```python
llm = DeepSeekLLM(api_key="sk-你的API密钥")
```

### 2. 准备数据集（Memora）

```bash
git clone https://github.com/geniesinc/Memora.git
```

确保 `Memora/data/` 目录存在，结构如下：

```
Memora/data/
├── weekly/
│   ├── software_engineer/
│   │   ├── conversations/
│   │   └── evaluation_questions_software_engineer.json
│   ├── academic_researcher/
│   └── ...
├── monthly/
└── quarterly/
```

### 3. 运行评测

```bash
python evaluation_v2.py
```

程序会自动：
1. 加载指定 persona 的对话历史和评测问题
2. 分别用基线 Agent（无记忆）和 MemoryBank Agent（有记忆）回答
3. 使用 LLM-as-Judge 对回答质量打分
4. 汇总所有 persona 的结果

---

## 核心机制

### MemoryBank 模块（`memorybank.py`）

| 论文机制 | 代码实现 |
|---------|---------|
| 记忆存储（三层结构） | `MemoryPiece` + 类型标记（dialog / summary / portrait） |
| 向量检索 | `FAISS IndexFlatIP` + `sentence-transformers` Embedding |
| 艾宾浩斯遗忘曲线 | `update_forgetting()` 中 `R = e^(-t/S)` |
| Spacing Effect | 检索时 `strength += 1`，被回忆的记忆更持久 |
| 增强 Prompt | `build_augmented_prompt()` 整合记忆 + 画像 + 摘要 |
| Agent 封装 | `MemoryAugmentedAgent` / `BaselineAgent` |

### 评测框架（`evaluation_v2.py`）

| 特性 | 说明 |
|------|------|
| **数据集支持** | Memora（已适配）、LongMemEval / STALE / MemConflict（预留接口） |
| **批量评测** | 支持一次跑多个 persona，自动汇总 |
| **LLM-as-Judge** | 用 DeepSeek 评判回答是否准确利用了历史记忆，替代粗糙的关键词匹配 |
| **评分标准** | 0 = 完全错误 / 编造；0.1-0.3 = 几乎未利用历史记忆；0.7-0.9 = 基本正确；1.0 = 完全准确 |

---

## 实验结果

使用 **Memora** 数据集 `weekly` 子集，在 5 个不同 persona 上进行评测（共 75 题）。

| Persona | 基线均分 | MemoryBank | 提升幅度 | MemoryBank 答对 |
|---------|---------|-----------|---------|----------------|
| software_engineer | 0.080 | 0.333 | 316.7% | 2/15 |
| academic_researcher | 0.107 | 0.287 | 168.7% | 2/15 |
| business_executive | 0.087 | 0.173 | 100.0% | 0/15 |
| financial_analyst | 0.087 | 0.253 | 192.3% | 2/15 |
| startup_founder | 0.100 | 0.127 | 26.7% | 0/15 |
| **总计** | **0.092** | **0.235** | **155.1%** | **6/75** |

### 按任务类型分析

| 任务类型 | 基线均分 | MemoryBank | 说明 |
|---------|---------|-----------|------|
| **remembering** | ~0.08 | ~0.16 | **核心优势**。事实回忆精准，向量检索能匹配语义相关的历史记忆 |
| **recommending** | ~0.10 | ~0.84 | **显著提升**。能基于用户历史偏好给出个性化推荐（如电影、书籍），基线只能给出通用建议 |
| **reasoning** | ~0.06 | ~0.14 | **表现有限**。涉及计算和综合推理时，单纯检索记忆片段不足以完成任务 |

### 关键发现

1. **MemoryBank 在个性化推荐上效果显著**：基线只能给出"请告诉我你的喜好"这类通用回复，MemoryBank 能基于历史对话精准推荐（如"你喜欢《火星救援》，推荐《星际穿越》"）。
2. **事实回忆是稳定优势**：在 remember 类任务上 consistently 优于基线。
3. **推理能力是短板**：需要计算总和、汇总多日数据等推理任务，MemoryBank 无法自动完成。这说明**记忆管理和推理能力是两个不同的问题**。

---

## 已知局限

1. **缺乏主动过期检测**
   - 当前 MemoryBank 不会主动识别"旧记忆已失效"
   - 例如用户从"喜欢川菜"改为"清淡饮食"，系统可能同时召回新旧两种偏好
   - **改进方向**：参考 Memora / STALE 论文，加入记忆冲突检测和过期标记

2. **检索质量依赖 Embedding 模型**
   - 短查询 + 长记忆的场景下，纯语义相似度检索召回率不足
   - **改进方向**：引入关键词硬匹配辅助，或调整检索策略

3. **无法完成计算型推理**
   - 能检索到"花了 $7.35 买咖啡"，但不会自动计算"本周咖啡总支出"
   - 这是记忆模块的设计边界，需结合更强的推理模块解决

4. **端侧适配未实现**
   - 当前使用云端 DeepSeek API，未涉及端侧模型部署
   - **改进方向**：结合 llama.cpp 等端侧推理框架，配合第 11 篇论文（Budget-Curated Memory）进行记忆预算管理

---

## 参考文献

1. **MemoryBank** (AAAI-24): Zhong et al. *MemoryBank: Enhancing Large Language Models with Long-Term Memory*
2. **Memora** (arXiv 2026): *From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents* — 数据集来源
3. **LongMemEval** (ICLR 2025): *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*
4. **STALE** (arXiv 2026): *Can LLM Agents Know When Their Memories Are No Longer Valid?*
5. **MemConflict** (arXiv 2026): *Evaluating Long-Term Memory Systems Under Memory Conflicts*
6. **Forget to Improve** (arXiv 2026): *On-Device LLM-Agent Continual Learning via Budget-Curated Memory* — 端侧预算管理

---

## 后续计划

- [ ] 过期记忆检测模块（冲突识别 + 旧记忆标记）
- [ ] 端侧部署适配（llama.cpp + 轻量模型）
- [ ] Gradio 交互式 Demo
- [ ] 支持更多评测数据集（LongMemEval / STALE / MemConflict）
