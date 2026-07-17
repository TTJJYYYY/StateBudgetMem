# Router 小修方案 Proposal

## 1. 文档目的

本文根据正式 fair comparison 的 Rule Routing 误差分析，提出一个轻量 Router 修复方案。

目标是在不修改以下模块的前提下，提高 Rule Router 对 CURRENT、HISTORICAL 和 CHANGE 查询的识别能力：

- Dense Retrieval；
- MiniLM embedding；
- FAISS；
- MemoryBank 评分；
- Versioning；
- Current View；
- History View；
- Token Budget；
- Candidate Filtering 接口。

本文只提出修改建议，不直接修改正式 Router 或 Adapter 逻辑。

任何对正式 Router 的修改，都需要先由组长确认。

---

## 2. 当前问题证据

正式实验使用：

```yaml
dataset_path: data/controlled/temporal_challenge_v1.jsonl
top_k: 3
candidate_k: 20
token_budget: 256
random_seed: 42
repeat: 1
embedding_backend: sentence_transformer
embedding_model: all-MiniLM-L6-v2
```

正式结果显示：

| 指标 | 数值 |
|---|---:|
| 总体路由准确率 | 46.875% |
| CURRENT 路由准确率 | 78.125% |
| HISTORICAL 路由准确率 | 3.125% |
| CHANGE 路由准确率 | 59.375% |
| 错误回退 GENERAL | 50 条 |
| Rule Recall@K | 0.2506 |
| Oracle Recall@K | 0.4758 |
| Rule Valid Recall@K | 0.3455 |
| Oracle Valid Recall@K | 0.7335 |

混淆矩阵如下：

| Gold Query Type | Pred CURRENT | Pred HISTORICAL | Pred CHANGE | Pred GENERAL | Total |
|---|---:|---:|---:|---:|---:|
| CURRENT | 25 | 0 | 0 | 7 | 32 |
| HISTORICAL | 0 | 1 | 0 | 31 | 32 |
| CHANGE | 1 | 0 | 19 | 12 | 32 |
| GENERAL | 0 | 0 | 0 | 0 | 0 |

主要问题是：

1. 31/32 的 HISTORICAL 查询被预测为 GENERAL；
2. 12/32 的 CHANGE 查询被预测为 GENERAL；
3. 7/32 的 CURRENT 查询被预测为 GENERAL；
4. 未命中规则时直接回退 GENERAL；
5. GENERAL 路径导致个人记忆候选集合为空。

---

## 3. 修改范围

建议只修改：

```text
src/statebudgetmem/routing/router.py
```

可以配套新增或修改：

```text
tests/routing/test_router.py
tests/routing/test_router_temporal_patterns.py
```

暂时不修改：

```text
src/statebudgetmem/baselines/memorybank/statebudgetmem_adapter.py
src/statebudgetmem/views/
src/statebudgetmem/versioning/
src/statebudgetmem/baselines/memorybank/
src/statebudgetmem/unified_runner.py
```

原因是当前 Oracle 结果已经说明 Dense Retrieval 和 scoped retrieval 接口能够正常工作，主要问题集中在 Router 的 Query Type 判断。

---

## 4. 建议一：扩展 CURRENT 时间表达

当前错误案例表明，“今晚”“这周”等表达没有稳定识别为 CURRENT。

建议增加以下 CURRENT 信号：

```text
今天
今晚
今早
今天早上
今天下午
今天晚上
这周
本周
本月
现在
目前
当前
最近
这两天
眼下
现阶段
```

推荐使用集合或正则统一管理，例如：

```python
CURRENT_KEYWORDS = {
    "今天",
    "今晚",
    "今早",
    "这周",
    "本周",
    "本月",
    "现在",
    "目前",
    "当前",
    "最近",
    "这两天",
    "眼下",
    "现阶段",
}
```

需要覆盖的测试样例：

```text
我今晚应该服用多少毫克？
这周我还能按原计划跑五公里吗？
我目前主要使用什么手机？
我现在工作日怎么通勤？
最近我每天喝多少咖啡？
```

---

## 5. 建议二：增加 HISTORICAL 时间表达

HISTORICAL 当前只有 1/32 识别正确，是最严重的问题。

错误案例大量包含月份、日期和过去时间段，例如：

```text
三月份我更偏好哪种书籍形式？
二月份我工作日通常怎么去公司？
五月份我通常每天喝多少咖啡？
六月二十日我的长期饮食限制有哪些？
去年我住在哪里？
```

建议增加以下 HISTORICAL 信号：

```text
过去
以前
之前
曾经
当时
那时候
原来
最开始
起初
去年
前年
上个月
上周
过去一段时间
之前那段时间
那段时间
```

同时增加月份和日期正则。

### 5.1 中文月份

```python
MONTH_PATTERN = re.compile(
    r"(一|二|三|四|五|六|七|八|九|十|十一|十二)月份"
)
```

### 5.2 数字月份

```python
NUMERIC_MONTH_PATTERN = re.compile(
    r"(?<!\d)(1[0-2]|0?[1-9])月份?"
)
```

### 5.3 中文日期

```python
CHINESE_DATE_PATTERN = re.compile(
    r"[一二三四五六七八九十]+月"
    r"[一二三四五六七八九十]+日"
)
```

### 5.4 数字日期

```python
NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(1[0-2]|0?[1-9])月"
    r"(3[01]|[12]\d|0?[1-9])日"
)
```

### 5.5 相对历史时间

```python
HISTORICAL_KEYWORDS = {
    "过去",
    "以前",
    "之前",
    "曾经",
    "当时",
    "那时候",
    "原来",
    "最开始",
    "起初",
    "去年",
    "前年",
    "上个月",
    "上周",
    "过去一段时间",
    "之前那段时间",
    "那段时间",
}
```

---

## 6. 建议三：扩展 CHANGE 表达

当前 CHANGE 识别率为 59.375%，主要遗漏自然语言中的变化表达。

错误案例：

```text
我的暑假旅行目的地是怎么改的？
```

虽然包含明显变化含义，但没有被当前规则识别。

建议增加以下 CHANGE 信号：

```text
变化
改变
改了
改成
怎么改
变成
后来
不再
是否还
还保持
原来和现在
之前和现在
从……到……
从……变成……
先……后来……
```

推荐正则：

```python
CHANGE_PATTERNS = [
    re.compile(r"怎么改"),
    re.compile(r"改成"),
    re.compile(r"变成"),
    re.compile(r"从.+到"),
    re.compile(r"从.+变成"),
    re.compile(r"原来.+现在"),
    re.compile(r"之前.+现在"),
    re.compile(r"先.+后来"),
    re.compile(r"是否还"),
    re.compile(r"还保持"),
    re.compile(r"不再"),
]
```

当前类似：

```text
从...到
还...吗
```

如果使用普通字符串包含判断，只能匹配字面量的三个点，无法匹配真实句子。

因此需要改为正则表达式。

---

## 7. 建议四：调整 Router 的判断优先级

建议优先级为：

```text
CHANGE
→ HISTORICAL
→ CURRENT
→ 明确 GENERAL
→ Personal Query Fallback
→ GENERAL
```

原因如下。

### 7.1 CHANGE 优先

包含“原来和现在”“从……变成……”等表达的查询，可能同时包含“现在”和过去时间词。

如果 CURRENT 先判断，会错误地只选择 Current View。

例如：

```text
我原来开车，现在改成坐地铁了吗？
```

该问题应分类为 CHANGE，而不是 CURRENT。

### 7.2 HISTORICAL 优先于 CURRENT

某些历史问题可能包含“现在”作为比较背景，但主要询问过去状态。

需要根据完整模式判断，而不是只要出现“现在”就判 CURRENT。

### 7.3 GENERAL 只匹配明确通用问题

GENERAL 不应作为所有未匹配问题的立即 fallback。

---

## 8. 建议五：限制过宽的 GENERAL 关键词

如果 GENERAL 规则中包含以下单字：

```text
加
减
```

会误判很多个人状态变化问题，例如：

```text
我的运动量减少了吗？
咖啡量增加了吗？
药量从多少增加到多少？
最近的工作压力减轻了吗？
```

这些问题虽然包含“加”或“减”，但语义上属于 CHANGE 或 CURRENT。

建议：

1. 删除单字级 GENERAL 关键词；
2. 改用明确的数学表达；
3. 使用算式、数字和计算动词组合识别 GENERAL。

例如：

```python
MATH_PATTERNS = [
    re.compile(r"\d+\s*[+\-×÷*/]\s*\d+"),
    re.compile(r"\d+\s*加\s*\d+"),
    re.compile(r"\d+\s*减\s*\d+"),
    re.compile(r"\d+\s*乘\s*\d+"),
    re.compile(r"\d+\s*除以\s*\d+"),
]
```

明确 GENERAL 关键词可以包括：

```text
等于多少
计算
换算
定义
什么是
为什么
天气
百科
```

但也需要防止“什么是我以前的习惯”这类个人问题被误判。

---

## 9. 建议六：增加 Personal Query Fallback

当前未命中规则后直接返回 GENERAL，会导致大量个人状态查询空检索。

建议先判断查询是否包含个人状态信号。

个人信号：

```python
PERSONAL_MARKERS = {
    "我",
    "我的",
    "本人",
}
```

个人属性词：

```python
PERSONAL_STATE_TERMS = {
    "喜欢",
    "偏好",
    "习惯",
    "工作",
    "通勤",
    "住",
    "住址",
    "饮食",
    "药",
    "服用",
    "计划",
    "手机",
    "设备",
    "宠物",
    "咖啡",
    "跑步",
    "旅行",
    "限制",
    "状态",
}
```

建议逻辑：

```python
def looks_personal_query(text: str) -> bool:
    has_personal_marker = any(
        marker in text
        for marker in PERSONAL_MARKERS
    )

    has_personal_state = any(
        term in text
        for term in PERSONAL_STATE_TERMS
    )

    return has_personal_marker and has_personal_state
```

如果问题具有明显个人状态特征，但没有匹配 HISTORICAL 或 CHANGE，可考虑默认分类为 CURRENT。

伪代码：

```python
if matches_change(text):
    return QueryType.CHANGE

if matches_historical(text):
    return QueryType.HISTORICAL

if matches_current(text):
    return QueryType.CURRENT

if matches_explicit_general(text):
    return QueryType.GENERAL

if looks_personal_query(text):
    return QueryType.CURRENT

return QueryType.GENERAL
```

是否允许 personal query 默认回退到 CURRENT，需要先由组长确认。

---

## 10. 容易混淆的反例

新增规则时不能只让当前 96 条数据通过，还需要加入容易混淆的负例。

## 10.1 HISTORICAL 反例

```text
我之前说过的计划现在还有效吗？
```

虽然包含“之前”，但核心问题可能是 CURRENT 或 CHANGE。

```text
三月份发布的手机现在值得买吗？
```

这里“三月份”描述产品发布时间，不一定是用户历史状态。

## 10.2 CURRENT 反例

```text
现在和以前有什么区别？
```

虽然包含“现在”，但应分类为 CHANGE。

```text
我现在想知道去年住在哪里。
```

虽然包含“现在”，核心查询仍然是 HISTORICAL。

## 10.3 CHANGE 反例

```text
怎么修改文件名？
```

包含“修改”，但属于 GENERAL，不是个人状态变化。

```text
什么是状态改变？
```

属于知识问答 GENERAL。

## 10.4 GENERAL 反例

```text
我的咖啡量增加了吗？
```

包含“加”，但属于 CHANGE。

```text
我减少药量了吗？
```

包含“减”，但属于 CHANGE。

---

## 11. 建议测试文件结构

建议新增：

```text
tests/routing/test_router_temporal_patterns.py
```

测试内容至少包括以下四组。

### 11.1 CURRENT

```python
@pytest.mark.parametrize(
    "query",
    [
        "我今晚应该服用多少毫克？",
        "这周我还能跑五公里吗？",
        "我目前主要使用什么手机？",
        "最近我每天喝多少咖啡？",
    ],
)
def test_current_temporal_patterns(query):
    ...
```

### 11.2 HISTORICAL

```python
@pytest.mark.parametrize(
    "query",
    [
        "三月份我更偏好哪种书籍形式？",
        "二月份我工作日怎么通勤？",
        "六月二十日我的饮食限制是什么？",
        "去年我住在哪里？",
    ],
)
def test_historical_temporal_patterns(query):
    ...
```

### 11.3 CHANGE

```python
@pytest.mark.parametrize(
    "query",
    [
        "我的旅行目的地是怎么改的？",
        "我从开车变成坐地铁了吗？",
        "原来和现在有什么区别？",
        "我还保持以前的习惯吗？",
    ],
)
def test_change_patterns(query):
    ...
```

### 11.4 GENERAL 负例

```python
@pytest.mark.parametrize(
    "query",
    [
        "3 加 5 等于多少？",
        "一公里等于多少米？",
        "今天北京天气如何？",
        "什么是操作系统？",
        "怎么修改文件名？",
    ],
)
def test_general_negative_examples(query):
    ...
```

---

## 12. 修改后的实验安排

如果组长批准修改 Router，建议按以下步骤执行。

### 第一步：创建独立分支

```text
feature/router-analysis-or-fix
```

不要直接在当前正式 fair comparison 分支修改 Router。

### 第二步：增加测试

先补 Router 单元测试，再修改实现。

### 第三步：修改 Router

只修改路由匹配规则和 fallback，不修改 Dense Retrieval。

### 第四步：运行测试

至少运行：

```powershell
python -m pytest tests/routing -q
```

以及全量：

```powershell
python -m pytest -q
```

### 第五步：重新运行 Rule

使用与原实验完全一致的配置：

```yaml
top_k: 3
candidate_k: 20
token_budget: 256
random_seed: 42
repeat: 1
embedding_model: all-MiniLM-L6-v2
```

只重新运行：

```text
statebudgetmem_rule
```

必要时同时重新运行 Oracle 作为环境一致性检查。

### 第六步：保存为补充实验

建议结果目录：

```text
results/router_fix_supplementary/
```

或者：

```text
results/fair_comparison_router_fix/
```

不得覆盖：

```text
results/fair_comparison/
```

---

## 13. 评价指标

修改前后至少比较：

1. Overall Routing Accuracy；
2. Routing Accuracy by Query Type；
3. Recall@K；
4. Valid Recall@K；
5. Stale Retrieval Rate；
6. Empty Retrieval Rate；
7. GENERAL fallback 数量；
8. Rule 与 Oracle 的差距；
9. Router 负例错误率。

重点目标：

- 显著降低 HISTORICAL → GENERAL；
- 显著降低 CHANGE → GENERAL；
- 降低空检索率；
- 提高 Recall@K 和 Valid Recall@K；
- 不显著增加错误个人记忆检索；
- 不破坏真正 GENERAL 问题。

---

## 14. 风险

## 14.1 过拟合当前 96 条查询

如果只根据当前错误句子添加完整字符串规则，会得到表面提升，但不能泛化。

应使用：

- 时间表达正则；
- 语义类别词；
- 同义改写；
- 负例测试。

## 14.2 Personal Query Fallback 过宽

将所有包含“我”的问题默认分类为 CURRENT，可能错误处理：

```text
我想知道什么是操作系统？
```

因此 personal fallback 需要同时检查个人状态属性，而不能只检查第一人称。

## 14.3 CHANGE 优先级过高

“修改文件”“改变变量”等通用问题可能被误判为 CHANGE。

CHANGE 规则需要结合个人状态信号。

## 14.4 时间词歧义

月份和日期可能描述外部事件，而非用户历史状态。

例如：

```text
三月份发布的手机现在能买吗？
```

需要结合“我”“我的”和个人属性词判断。

---

## 15. 推荐的最小修改顺序

建议按以下优先级实施：

1. 增加月份、日期和历史时间段识别；
2. 增加“今晚”“这周”等 CURRENT 表达；
3. 将 CHANGE 的字面模板改为正则；
4. 删除或限制 GENERAL 中单字“加”“减”；
5. 增加 Personal Query Fallback；
6. 增加负例测试；
7. 重新运行补充实验。

其中前四项风险较低，Personal Query Fallback 对正式逻辑影响更大，需要组长单独确认。

---

## 16. 预期结果

在不修改 Dense Retrieval 的情况下，合理预期：

1. HISTORICAL Routing Accuracy 明显提升；
2. GENERAL fallback 数量明显下降；
3. HISTORICAL Empty Retrieval Rate 明显下降；
4. CHANGE Recall@K 提升；
5. CURRENT Recall@K 小幅提升；
6. Rule 与 Oracle 的差距缩小；
7. Stale Retrieval Rate 不应明显上升。

本 proposal 不预先承诺具体提升数值，实际效果必须通过补充实验验证。

---

## 17. 待组长确认

在修改正式 Router 前，需要组长确认以下事项：

1. 是否允许修改 `router.py`；
2. 是否允许使用正则时间表达；
3. 是否限制 GENERAL 中单字“加”“减”；
4. 是否允许 Personal Query 默认回退 CURRENT；
5. 是否创建独立 Router Fix 分支；
6. 修复结果是否进入最终报告；
7. 修复结果是否作为补充实验；
8. 是否保留修复前后两组 Rule 结果。

---

## 18. 结论

当前 Rule Router 的主要问题是规则覆盖不足和 GENERAL fallback 过度，而不是 Dense Retrieval 能力不足。

建议优先完成：

1. 时间表达扩展；
2. CHANGE 正则模式；
3. GENERAL 规则收紧；
4. Router 负例测试；
5. Personal Query Fallback 评估。

所有修复应在独立分支中完成，并作为补充实验保存，不能覆盖当前正式 fair comparison 结果。