# 当前状态与历史版本双视图

本目录实现 StateBudgetMem 项目中的 `views/` 模块，负责维护和检索两种记忆视图：

* `Current View`：当前状态视图，只保留当前有效状态；
* `History View`：历史版本视图，保存完整历史版本；
* `Dual View`：双视图检索，根据问题类型动态选择当前视图、历史视图或两者结合。

本模块主要解决长期记忆系统中的一个核心问题：

> 只保存当前状态会丢失历史，但保存全部历史又容易让旧记忆干扰当前问题。

因此，`views/` 的目标是在提高当前问题准确率的同时，仍然保留回答过去状态和变化过程的能力。

---

## 一、模块目标

本模块主要完成以下功能：

1. 实现 `Current View`，只保留当前有效状态；
2. 实现 `History View`，保存完整历史版本；
3. 实现双视图的维护和检索；
4. 比较三种方法的效果：

   * 统一记忆库；
   * 仅当前状态；
   * 当前状态与历史版本双视图。

其中：

* 当前问题优先使用 `Current View`，避免旧记忆干扰；
* 历史问题优先使用 `History View`，保证过去状态可追溯；
* 变化类问题同时使用 `Current View` 和 `History View`，回答状态变化过程。

---

## 二、设计思想

长期记忆中经常会出现同一属性不断变化的情况，例如：

```text
1月：用户住在上海。
3月：用户搬到了北京。
5月：用户又搬到了深圳。
```

如果只保存当前状态：

```text
用户现在住在深圳。
```

系统可以正确回答“用户现在住在哪里”，但无法回答“用户以前住在哪里”。

如果保存全部历史：

```text
用户住在上海。
用户住在北京。
用户住在深圳。
```

系统虽然保留了完整历史，但在回答“用户现在住在哪里”时，可能错误召回旧状态。

因此，本模块将记忆分为两个视图：

```text
Current View
└── 只保留当前有效状态

History View
└── 保存完整历史版本
```

查询时根据问题类型选择不同视图：

```text
CURRENT 查询
└── 使用 Current View

HISTORICAL 查询
└── 使用 History View

CHANGE 查询
└── 使用 Current View + History View

GENERAL 查询
└── 使用 Current View + History View
```

---

## 三、与项目统一接口的关系

本模块不重新定义数据格式，而是复用项目已有的统一接口。

输入记忆格式：

```python
MemoryRecord
```

输入查询格式：

```python
QueryRecord
```

输出结果格式：

```python
MethodResult
```

统一方法接口：

```python
MemoryMethod
```

也就是说，`views/` 中的方法和 `baselines/` 中的方法一样，都可以被 evaluation 模块统一调用和比较。

---

## 四、与其他模块的关系

本模块主要依赖以下已有模块：

```text
schemas/
└── 定义 MemoryRecord、QueryRecord、MethodResult 等实验数据结构

versioning/
└── 判断记忆版本关系，并维护当前状态和历史版本

baselines/
└── 提供基础 TF-IDF 检索器

evaluation/
└── 计算 recall、valid recall、stale retrieval rate 等指标
```

整体流程为：

```text
MemoryRecord 数据
        ↓
versioning 处理版本关系
        ↓
views 构建 Current View / History View
        ↓
根据 QueryRecord.query_type 选择检索视图
        ↓
返回 MethodResult
        ↓
evaluation 计算指标
```

如果接入 `preprocessing/`，则需要先将自然语言解析结果转换为 `MemoryRecord`：

```text
raw messages
        ↓
preprocessing
        ↓
ParsedMemory
        ↓
record_adapter.py
        ↓
MemoryRecord
        ↓
versioning
        ↓
views
```

其中 `record_adapter.py` 不是 `views/` 必须依赖的文件，但如果要把 `preprocessing` 和 `views` 串起来，建议补充。

---

## 五、目录结构

```text
src/statebudgetmem/views/
├── __init__.py
├── README.md
├── models.py
├── selectors.py
├── manager.py
├── methods.py
└── runner.py
```

各文件作用如下：

```text
__init__.py
└── 对外导出 views 模块中的主要类和函数

models.py
└── 定义 ViewName、ViewPolicy、ViewDecision 等视图相关模型

selectors.py
└── 根据 versioning 的结果选择当前状态记忆和历史记忆

manager.py
└── 实现 RecordViewManager，负责维护 Current View 和 History View

methods.py
└── 实现可被 evaluation 调用的 MemoryMethod：
    - FlatViewMemoryMethod
    - CurrentOnlyMemoryMethod
    - HistoryOnlyMemoryMethod
    - DualViewMemoryMethod

runner.py
└── 提供 views 实验运行入口，用于比较三种方法
```

---

## 六、核心方法说明

### 1. FlatViewMemoryMethod

统一记忆库方法。

所有记忆都放在同一个检索池中，不区分当前状态和历史状态。

优点：

```text
历史信息保留完整，召回范围大。
```

缺点：

```text
当前问题容易召回已经过期的旧记忆。
```

适合作为对照组。

---

### 2. CurrentOnlyMemoryMethod

仅当前状态方法。

该方法只检索 `Current View` 中的当前有效记忆。

优点：

```text
回答当前状态问题时，过期记忆干扰较少。
```

缺点：

```text
无法很好回答历史问题和变化过程问题。
```

适合作为另一个对照组。

---

### 3. HistoryOnlyMemoryMethod

仅历史视图方法。

该方法检索完整历史版本，主要用于调试和分析。

一般不作为主要方法参与对比，但可以用于观察历史版本是否被完整保留。

---

### 4. DualViewMemoryMethod

双视图方法。

该方法根据查询类型动态选择视图：

```text
CURRENT
└── 只使用 Current View

HISTORICAL
└── 只使用 History View

CHANGE
└── 使用 Current View + History View

GENERAL
└── 使用 Current View + History View
```

优点：

```text
既能减少当前问题中的旧记忆干扰，又能保留历史问题和变化问题的回答能力。
```

这是本模块的主要方法。

---

## 七、实验比较

本模块重点比较三种方法：

| 方法    | 类名                        | 说明         |
| ----- | ------------------------- | ---------- |
| 统一记忆库 | `FlatViewMemoryMethod`    | 所有记忆一起检索   |
| 仅当前状态 | `CurrentOnlyMemoryMethod` | 只检索当前有效记忆  |
| 双视图   | `DualViewMemoryMethod`    | 根据查询类型选择视图 |

主要评价指标包括：

| 指标                     | 含义               |
| ---------------------- | ---------------- |
| `recall_at_k`          | 检索结果中命中相关记忆的比例   |
| `valid_recall_at_k`    | 检索结果中命中当前有效记忆的比例 |
| `stale_retrieval_rate` | 检索结果中召回过期记忆的比例   |
| `total_token_cost`     | 检索结果消耗的 token 数  |
| `latency_ms`           | 检索耗时             |

理想结果是：

```text
FlatViewMemoryMethod
└── 召回较高，但 stale_retrieval_rate 也较高

CurrentOnlyMemoryMethod
└── stale_retrieval_rate 较低，但历史问题召回较弱

DualViewMemoryMethod
└── 在 valid_recall_at_k 和 stale_retrieval_rate 之间取得更好的平衡
```

---

## 八、运行方式

在项目根目录运行：

```bash
PYTHONPATH=src python tools/views/run_views_experiment.py \
  --dataset data/controlled/temporal_challenge_v1.jsonl \
  --top-k 3 \
  --seed 42 \
  --results-dir results/views
```

默认比较三种方法：

```text
flat
current
dual
```

也可以手动指定：

```bash
PYTHONPATH=src python tools/views/run_views_experiment.py \
  --dataset data/controlled/temporal_challenge_v1.jsonl \
  --methods flat current dual \
  --top-k 3
```

---

## 九、测试方式

只测试 `views/` 模块：

```bash
PYTHONPATH=src pytest tests/views -q
```

如果同时加入了 `preprocessing/record_adapter.py`，可以运行：

```bash
PYTHONPATH=src pytest tests/views tests/preprocessing/test_record_adapter.py -q
```

---
## 十、本模块的作用

`views/` 模块在项目中的作用可以概括为：

```text
versioning 负责判断新旧记忆的版本关系；
views 负责把版本关系组织成可检索的当前视图和历史视图；
evaluation 负责比较不同视图策略的效果。
```

因此，本模块不是单独做一个检索器，而是在已有版本管理结果的基础上，实现更适合时态一致性长期记忆的检索视图。

最终目标是证明：

```text
相比统一记忆库，双视图能够减少旧记忆对当前问题的干扰；
相比仅当前状态，双视图仍然能够回答历史状态和变化过程问题。
```
