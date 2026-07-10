# MemoryBank 复现数据集

## 1. 数据集简介

本目录用于存放 MemoryBank 风格长期记忆系统复现数据集。

该数据集用于支持 StateBudgetMem 项目中的 MemoryBank baseline，
用于评估个人智能体长期记忆能力。

数据集主要用于验证智能体是否能够：

- 保存用户长期信息；
- 根据历史对话生成记忆摘要；
- 构建用户画像；
- 检索相关历史记忆；
- 基于历史交互回答问题。


该数据集参考 MemoryBank 的记忆写入流程：

原始对话
|
v
每日事件摘要（Daily Event Summary）
|
v
全局事件总结（Global Event Summary）


原始对话
|
v
每日人格观察（Daily Personality）
|
v
用户长期画像（Global User Portrait）


该数据集主要服务于：

src/statebudgetmem/baselines/memorybank/

中的 MemoryBank baseline 实现。


---

## 2. 目录结构


memorybank_reproduction/

├── users/
│   ├── user_001.json
│   ├── user_002.json
│   ├── user_003.json
│   ├── user_004.json
│   └── user_005.json
│
├── probing_questions.jsonl
│
└── README.md



---

## 3. 用户数据格式


每个用户文件包含：

- 用户基本信息（profile）
- 多天历史对话（days）
- 每日事件总结（daily_event_summary）
- 每日人格观察（daily_personality）
- 长期事件总结（global_event_summary）
- 长期用户画像（global_user_portrait）


数据结构：

{
    "user_id": "user_001",

    "profile": {
        "name": "",
        "personality": "",
        "interests": []
    },

    "days": [
        {
            "date": "",
            "dialogues": [],

            "daily_event_summary": "",

            "daily_personality": ""
        }
    ],

    "global_event_summary": "",

    "global_user_portrait": ""
}



---

## 4. 数据集规模


| 项目 | 数量 |
| --- | --- |
| 虚拟用户数量 | 5 |
| 每个用户天数 | 7 |
| 总对话天数 | 35 |
| 测试问题数量 | 50 |
| 每个用户问题数量 | 10 |


每一天包含：

- 多轮用户与助手对话；
- 至少两个不同主题；
- 一个每日事件总结；
- 一个每日人格观察。



---

## 5. 用户设计说明


### user_001

主题：

学习方向与职业规划变化。


主要内容：

- Python学习；
- 数据分析；
- 考研规划；
- AI方向发展。


主要测试：

长期目标变化记忆。


变化过程：

准备考研

|

v

软件开发方向

|

v

AI软件工程师



---

### user_002

主题：

生活习惯与偏好变化。


主要内容：

- 饮食习惯；
- 跑步；
- 健康管理。


主要测试：

用户偏好更新。


变化过程：

奶茶和甜食

|

v

减少糖分摄入

|

v

健康饮食和跑步



---

### user_003

主题：

技术学习与项目经历。


主要内容：

- Python；
- 数据结构；
- 快速排序；
- Flask；
- MySQL；
- GitHub。


主要测试：

具体事实记忆能力。



---

### user_004

主题：

兴趣爱好与用户画像。


主要内容：

- 摄影；
- 旅行；
- 音乐；
- 艺术。


主要测试：

长期用户画像能力。



---

### user_005

主题：

职业选择变化。


主要内容：

- 公务员考试；
- 互联网行业；
- 产品经理。


主要测试：

长期决策过程追踪能力。


变化过程：

公务员方向

|

v

互联网行业探索

|

v

产品经理方向



---

## 6. Probing Questions


probing_questions.jsonl 用于评估 MemoryBank 的检索和回答能力。


每条问题包含：

- question_id
- user_id
- question
- question_type
- reference_answer
- gold_memory_ids
- expected_keywords



字段说明：

question_id：

问题编号。


user_id：

对应用户。


question：

测试问题。


question_type：

问题类型。


reference_answer：

人工标准答案。


gold_memory_ids：

答案对应的历史记忆。


expected_keywords：

答案关键词。



---

## 7. 问题类型


### memory_recall


测试模型是否能够准确回忆历史事实。


例如：

我学习了什么编程语言？



---

### event_summary


测试模型是否能够总结用户过去的重要事件。


例如：

我完成了什么项目？



---

### user_portrait


测试模型是否能够理解用户长期兴趣和人格特点。


例如：

我是一个什么样的人？



---

### temporal_memory


测试模型是否能够理解用户状态随时间变化。


例如：

我的职业规划经历了什么变化？



---

### negative_memory


测试模型是否会生成不存在的记忆。


例如：

我是否提到过学习Java？



---

## 8. Memory ID 设计


每条对话都有唯一编号。


格式：

user_xxx_dayxx_dialogxx


例如：

user_001_day01_dialog01


Memory ID 用于：

- 标记答案来源；
- 生成 gold labels；
- 评估检索结果。



---

## 9. 与 StateBudgetMem 的关系


该数据集主要用于：

Memory Writing

+

Memory Retrieval


数据流程：

用户历史对话

|

v

MemoryBank存储

|

v

检索与回答评估


StateBudgetMem 其他模块负责：

- 时态版本管理；
- 当前状态维护；
- 历史状态保存；
- 过期记忆控制。


该数据集为 MemoryBank baseline 提供基础测试环境。



---

## 10. 使用方式


数据加载由：

src/statebudgetmem/baselines/memorybank/datasets.py


负责。


主要流程：

users/*.json

|

v

Dataset Loader

|

v

Memory Writer

|

v

MemoryBank Storage

|

v

Evaluation