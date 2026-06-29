"""
MemoryBank: 长期记忆管理模块
基于论文 "MemoryBank: Enhancing Large Language Models with Long-Term Memory" (AAAI-24)

核心功能：
1. 记忆存储：存储原始对话、事件摘要、用户画像
2. 记忆检索：基于向量相似度检索相关记忆
3. 记忆更新：基于艾宾浩斯遗忘曲线进行记忆衰减和强化

使用方法：
    from memorybank import MemoryBank
    mb = MemoryBank()
    mb.store("用户：我喜欢打篮球", "2026-06-29 10:00")
    memories = mb.retrieve("用户：推荐一些运动", top_k=3)
    mb.update_forgetting(current_time="2026-06-30 10:00")
"""

import json
import time
import math
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import faiss


@dataclass
class MemoryPiece:
    """单条记忆单元"""
    content: str                  # 记忆内容（对话或摘要）
    timestamp: float              # 创建时间戳（unix time）
    memory_type: str              # 类型：dialog / summary / portrait
    strength: float = 1.0         # 记忆强度 S（初始为1，每次被回忆+1）
    last_accessed: float = 0.0    # 上次被回忆的时间戳
    embedding: Optional[np.ndarray] = None  # 向量表示（不存入磁盘时忽略）
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['embedding'] = None  # embedding 不序列化
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryPiece':
        return cls(**{k: v for k, v in d.items() if k != 'embedding'})


class MemoryBank:
    """
    MemoryBank 核心实现
    
    三层记忆结构：
    - raw_dialogs: 原始对话记录（带时间戳）
    - daily_summaries: 每日事件摘要
    - user_portrait: 用户画像（性格特征等）
    
    检索机制：
    - 使用 FAISS 进行向量相似度检索
    - 检索时考虑语义相关性 + 记忆强度 + 时间衰减
    
    遗忘机制：
    - 基于艾宾浩斯遗忘曲线 R = e^(-t/S)
    - t: 距离上次回忆的时间
    - S: 记忆强度（被回忆时+1）
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 默认 384 维
        forgetting_threshold: float = 0.3,  # 记忆留存率低于此值则遗忘
        decay_interval_hours: float = 24.0,  # 遗忘检查的时间间隔（小时）
        embedding_model = None  # 外部注入的 embedding 模型
    ):
        self.embedding_dim = embedding_dim
        self.forgetting_threshold = forgetting_threshold
        self.decay_interval_sec = decay_interval_hours * 3600

        # Embedding 模型（可外部注入）
        self._embedding_model = embedding_model
        self._embedding_model_local = None

        # 记忆存储
        self.memories: List[MemoryPiece] = []  # 所有记忆片段
        self.user_portrait: str = ""           # 用户画像
        self.global_summary: str = ""          # 全局事件摘要

        # FAISS 向量索引（懒加载，首次使用时根据实际 embedding 维度创建）
        self.index = None
        self.id_to_memory: Dict[int, MemoryPiece] = {}  # faiss id -> memory
        self.next_id = 0

        # 统计信息
        self.access_count = 0
        self.forget_count = 0

    def _ensure_index(self):
        """懒加载 FAISS 索引：首次获取 embedding 时才创建"""
        if self.index is None:
            # 用一个 dummy text 触发 embedding 模型加载，获取真实维度
            dummy_emb = self._get_embedding("test")
            self.embedding_dim = dummy_emb.shape[0]
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"[MemoryBank] FAISS 索引已初始化，维度: {self.embedding_dim}")

    # ────────────────────────── Embedding ──────────────────────────

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        if self._embedding_model is not None:
            # 外部模型（如 sentence-transformers）
            return self._embedding_model.encode(text)

        # 使用本地 embedding（如果没有外部模型，用随机向量占位）
        if self._embedding_model_local is None:
            # 尝试加载轻量模型
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model_local = SentenceTransformer('all-MiniLM-L6-v2')
                print("[MemoryBank] 已加载 embedding 模型: all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "需要 sentence-transformers 库。请运行: pip install sentence-transformers\n"
                    "或者传入自定义的 embedding_model（需有 .encode() 方法）"
                )

        emb = self._embedding_model_local.encode(text)
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        return emb.astype(np.float32)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 归一化（用于内积近似余弦相似度）"""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    # ────────────────────────── 记忆存储 ──────────────────────────

    def store(
        self,
        content: str,
        timestamp: Optional[str] = None,
        memory_type: str = "dialog"
    ) -> MemoryPiece:
        """
        存储一条新记忆

        Args:
            content: 记忆内容
            timestamp: 时间戳字符串，如 "2026-06-29 10:00"
            memory_type: dialog（对话）/ summary（摘要）/ portrait（画像）

        Returns:
            创建的记忆单元
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        ts = self._parse_time(timestamp)

        # 生成 embedding
        emb = self._get_embedding(content)
        emb = self._normalize(emb)

        # 创建记忆单元
        memory = MemoryPiece(
            content=content,
            timestamp=ts,
            memory_type=memory_type,
            strength=1.0,
            last_accessed=ts,
            embedding=emb
        )

        # 存入列表
        self.memories.append(memory)

        # 存入 FAISS
        self._ensure_index()
        faiss_id = self.next_id
        self.index.add(emb.reshape(1, -1))
        self.id_to_memory[faiss_id] = memory
        self.next_id += 1

        return memory

    def store_dialog(self, role: str, content: str, timestamp: Optional[str] = None):
        """快捷方法：存储对话"""
        dialog_text = f"{role}: {content}"
        return self.store(dialog_text, timestamp, memory_type="dialog")

    def store_summary(self, summary: str, timestamp: Optional[str] = None):
        """快捷方法：存储每日事件摘要"""
        return self.store(summary, timestamp, memory_type="summary")

    def update_user_portrait(self, portrait: str):
        """更新用户画像"""
        self.user_portrait = portrait

    def update_global_summary(self, summary: str):
        """更新全局摘要"""
        self.global_summary = summary

    # ────────────────────────── 记忆检索 ──────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[str]] = None,
        current_time: Optional[str] = None
    ) -> List[Dict]:
        """
        检索与查询最相关的记忆

        检索时会：
        1. 计算语义相似度（FAISS）
        2. 考虑记忆强度 S（被回忆次数越多越重要）
        3. 考虑时间衰减（越久远的记忆权重越低）

        Args:
            query: 查询文本
            top_k: 返回最相关的 k 条
            memory_types: 过滤特定类型的记忆，如 ["dialog", "summary"]
            current_time: 当前时间，用于计算时间衰减

        Returns:
            检索到的记忆列表，每条包含 content, score, type, age_hours 等
        """
        self._ensure_index()
        if self.index.ntotal == 0:
            return []

        # 编码查询
        query_emb = self._get_embedding(query)
        query_emb = self._normalize(query_emb)

        # FAISS 检索（返回更多候选，后续再过滤）
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_emb.reshape(1, -1), search_k)

        # 当前时间
        now_ts = self._parse_time(current_time) if current_time else time.time()

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx not in self.id_to_memory:
                continue

            mem = self.id_to_memory[idx]

            # 类型过滤
            if memory_types and mem.memory_type not in memory_types:
                continue

            # 计算时间衰减因子
            age_hours = (now_ts - mem.timestamp) / 3600.0
            time_decay = math.exp(-age_hours / 168)  # 一周衰减到 1/e

            # 综合分数 = 语义相似度 * 记忆强度 * 时间衰减
            # FAISS IP 返回的是内积，对于归一化向量等于余弦相似度
            semantic_score = float(dist)
            composite_score = semantic_score * (1 + mem.strength * 0.3) * time_decay

            # 强化被访问的记忆（Spacing Effect）
            mem.strength += 1
            mem.last_accessed = now_ts
            self.access_count += 1

            results.append({
                "content": mem.content,
                "memory_type": mem.memory_type,
                "semantic_score": round(semantic_score, 4),
                "composite_score": round(composite_score, 4),
                "strength": mem.strength,
                "age_hours": round(age_hours, 1),
                "timestamp": datetime.fromtimestamp(mem.timestamp).strftime("%Y-%m-%d %H:%M")
            })

        # 按综合分数排序
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:top_k]

    # ────────────────────────── 遗忘更新 ──────────────────────────

    def update_forgetting(self, current_time: Optional[str] = None):
        """
        执行遗忘更新

        根据艾宾浩斯遗忘曲线 R = e^(-t/S) 计算每条记忆的留存率，
        低于阈值的记忆将被标记为遗忘（从 FAISS 索引中移除）。

        Args:
            current_time: 当前时间，用于计算时间间隔
        """
        if not self.memories:
            return []

        now_ts = self._parse_time(current_time) if current_time else time.time()
        forgotten = []

        for mem in self.memories:
            # 距离上次被回忆的时间
            t = (now_ts - mem.last_accessed) / 3600.0  # 转为小时
            S = mem.strength

            # 艾宾浩斯遗忘曲线：R = e^(-t/S)
            retention = math.exp(-t / S) if S > 0 else 0

            if retention < self.forgetting_threshold:
                # 这条记忆被遗忘了
                mem.strength *= 0.5  # 衰减但不完全删除
                forgotten.append({
                    "content": mem.content[:50] + "...",
                    "retention": round(retention, 4),
                    "age_hours": round(t, 1)
                })
                self.forget_count += 1

        return forgotten

    # ────────────────────────── 构建增强 Prompt ──────────────────────────

    def build_augmented_prompt(
        self,
        query: str,
        current_time: Optional[str] = None,
        top_k: int = 5,
        include_portrait: bool = True,
        include_summary: bool = True
    ) -> Dict[str, str]:
        """
        构建增强后的 prompt

        将检索到的相关记忆、用户画像、全局摘要整合，
        供 LLM 使用以生成个性化回复。

        Returns:
            {
                "system_context": "用户画像和摘要...",
                "relevant_memories": "相关记忆...",
                "prompt_template": "完整 prompt 模板..."
            }
        """
        memories = self.retrieve(query, top_k=top_k, current_time=current_time)

        # 相关记忆文本
        memory_text = "\n".join([
            f"[{m['memory_type']}] {m['content']}"
            for m in memories
        ]) if memories else "（无相关历史记忆）"

        # 系统上下文
        context_parts = []
        if include_portrait and self.user_portrait:
            context_parts.append(f"【用户画像】{self.user_portrait}")
        if include_summary and self.global_summary:
            context_parts.append(f"【事件摘要】{self.global_summary}")

        system_context = "\n".join(context_parts)

        # 完整的 prompt 模板
        prompt = f"""你是一个有长期记忆的AI助手。请参考以下信息来回答用户的问题。

{system_context}

【相关历史记忆】
{memory_text}

【用户当前问题】
{query}

请基于历史记忆和当前问题，给出个性化、连贯的回答。如果历史记忆中有用户的偏好或重要信息，请在回答中体现。"""

        return {
            "system_context": system_context,
            "relevant_memories": memory_text,
            "prompt_template": prompt,
            "retrieved_count": len(memories)
        }

    # ────────────────────────── 持久化 ──────────────────────────

    def save(self, filepath: str):
        """保存记忆库到文件"""
        data = {
            "memories": [m.to_dict() for m in self.memories],
            "user_portrait": self.user_portrait,
            "global_summary": self.global_summary,
            "next_id": self.next_id,
            "access_count": self.access_count,
            "forget_count": self.forget_count
        }

        # 保存 FAISS 索引
        if self.index is not None:
            faiss.write_index(self.index, filepath + ".faiss")
        else:
            # 空索引时创建一个空的保存
            empty_index = faiss.IndexFlatIP(self.embedding_dim)
            faiss.write_index(empty_index, filepath + ".faiss")

        # 保存元数据
        with open(filepath + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[MemoryBank] 已保存 {len(self.memories)} 条记忆到 {filepath}")

    def load(self, filepath: str):
        """从文件加载记忆库"""
        # 加载 FAISS 索引
        self.index = faiss.read_index(filepath + ".faiss")

        # 加载元数据
        with open(filepath + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self.memories = [MemoryPiece.from_dict(m) for m in data["memories"]]
        self.user_portrait = data["user_portrait"]
        self.global_summary = data["global_summary"]
        self.next_id = data["next_id"]
        self.access_count = data["access_count"]
        self.forget_count = data["forget_count"]

        # 重建 id 映射
        self.id_to_memory = {}
        for i, mem in enumerate(self.memories):
            self.id_to_memory[i] = mem

        print(f"[MemoryBank] 已加载 {len(self.memories)} 条记忆")

    def get_stats(self) -> Dict:
        """获取记忆库统计信息"""
        type_counts = {}
        for m in self.memories:
            type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1

        return {
            "total_memories": len(self.memories),
            "type_distribution": type_counts,
            "total_access": self.access_count,
            "total_forgotten": self.forget_count,
            "index_size": self.index.ntotal if self.index is not None else 0,
            "user_portrait_set": bool(self.user_portrait),
            "global_summary_set": bool(self.global_summary)
        }

    # ────────────────────────── 工具方法 ──────────────────────────

    @staticmethod
    def _parse_time(t: Optional[str]) -> float:
        """解析时间字符串为 unix 时间戳"""
        if t is None:
            return time.time()
        if isinstance(t, (int, float)):
            return float(t)

        formats = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(t, fmt).timestamp()
            except ValueError:
                continue

        # 如果都失败，尝试直接解析
        try:
            from dateutil import parser
            return parser.parse(t).timestamp()
        except:
            return time.time()


# ═══════════════════════════════════════════════════════════════
# 快捷封装：带 MemoryBank 的对话 Agent
# ═══════════════════════════════════════════════════════════════

class MemoryAugmentedAgent:
    """
    接入 MemoryBank 的对话 Agent

    这是 MemoryBank 的"使用示例"——展示如何把记忆模块接到 LLM 上。
    实际使用时，需要替换 call_llm 方法为你自己的 LLM 调用逻辑。
    """

    def __init__(self, memory_bank: Optional[MemoryBank] = None, llm_caller=None):
        """
        Args:
            memory_bank: MemoryBank 实例（可选）
            llm_caller: LLM 调用函数，签名应为 f(prompt: str) -> str
        """
        self.memory = memory_bank or MemoryBank()
        self.llm_caller = llm_caller or self._default_llm
        self.dialog_history: List[Dict] = []

    def _default_llm(self, prompt: str) -> str:
        """占位 LLM——实际使用时请替换为真实的 LLM 调用"""
        return "[占位回复] 请配置真实的 LLM 调用函数"

    def chat(self, user_input: str, timestamp: Optional[str] = None) -> str:
        """
        进行一次对话

        流程：
        1. 用 MemoryBank 检索相关历史记忆
        2. 构建增强 prompt
        3. 调用 LLM 生成回复
        4. 存储本次对话到记忆库
        """
        # 1. 检索相关记忆
        context = self.memory.build_augmented_prompt(
            query=user_input,
            current_time=timestamp,
            top_k=5
        )

        # 2. 调用 LLM
        response = self.llm_caller(context["prompt_template"])

        # 3. 存储对话
        self.memory.store_dialog("用户", user_input, timestamp)
        self.memory.store_dialog("AI", response, timestamp)

        # 4. 记录对话历史
        self.dialog_history.append({
            "user": user_input,
            "ai": response,
            "retrieved_memories": context["retrieved_count"],
            "timestamp": timestamp
        })

        return response

    def batch_store_history(self, dialogs: List[Tuple[str, str, str]]):
        """
        批量存储历史对话（用于初始化时灌入模拟历史）

        Args:
            dialogs: [(role, content, timestamp), ...]
        """
        for role, content, ts in dialogs:
            self.memory.store_dialog(role, content, ts)

    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        return self.memory.get_stats()


# ═══════════════════════════════════════════════════════════════
# 基线 Agent（无记忆）——用于对比实验
# ═══════════════════════════════════════════════════════════════

class BaselineAgent:
    """
    无记忆的基线 Agent

    每次对话独立处理，不保留任何历史记忆。
    用于和 MemoryAugmentedAgent 做对比实验。
    """

    def __init__(self, llm_caller=None):
        self.llm_caller = llm_caller or (lambda p: "[占位回复]")
        self.dialog_history: List[Dict] = []

    def chat(self, user_input: str, timestamp: Optional[str] = None) -> str:
        """直接调用 LLM，无记忆增强"""
        prompt = f"请回答用户的问题：\n\n{user_input}"
        response = self.llm_caller(prompt)

        self.dialog_history.append({
            "user": user_input,
            "ai": response,
            "timestamp": timestamp
        })

        return response


# ═══════════════════════════════════════════════════════════════
# 如果直接运行此文件，执行示例
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MemoryBank 演示")
    print("=" * 60)

    # 初始化 MemoryBank
    mb = MemoryBank()

    # 模拟存储一些历史对话
    print("\n[1] 存储历史对话...")
    mb.store_dialog("用户", "你好，我叫小明", "2026-06-20 10:00")
    mb.store_dialog("AI", "你好小明！很高兴认识你。", "2026-06-20 10:00")
    mb.store_dialog("用户", "我喜欢打篮球和游泳", "2026-06-20 10:05")
    mb.store_dialog("AI", "篮球和游泳都是很棒的运动！", "2026-06-20 10:05")
    mb.store_dialog("用户", "我在准备期末考试，最近压力很大", "2026-06-21 15:00")
    mb.store_dialog("AI", "考试加油！需要我帮你做复习计划吗？", "2026-06-21 15:00")
    mb.store_dialog("用户", "上次你说的那本书叫什么名字？", "2026-06-22 09:00")
    mb.store_dialog("AI", "我推荐的是《深度学习入门》。", "2026-06-22 09:00")

    # 设置用户画像
    mb.update_user_portrait("用户小明，爱好篮球和游泳，近期在准备期末考试，性格友善")

    print(f"已存储 {len(mb.memories)} 条记忆")

    # 检索测试
    print("\n[2] 检索测试...")
    query = "推荐一些运动相关的活动"
    results = mb.retrieve(query, top_k=3, current_time="2026-06-25 10:00")
    print(f"查询: {query}")
    for r in results:
        print(f"  → [{r['memory_type']}] {r['content'][:40]}... "
              f"(score={r['composite_score']}, strength={r['strength']})")

    # 构建增强 prompt
    print("\n[3] 构建增强 prompt...")
    context = mb.build_augmented_prompt("最近想放松一下", current_time="2026-06-25 10:00")
    print(context["prompt_template"][:300] + "...")

    # 遗忘测试
    print("\n[4] 遗忘测试...")
    forgotten = mb.update_forgetting(current_time="2026-07-25 10:00")  # 一个月后
    print(f"被遗忘的记忆: {len(forgotten)} 条")
    for f in forgotten:
        print(f"  → {f['content']} (留存率: {f['retention']})")

    # 统计信息
    print("\n[5] 记忆库统计:")
    stats = mb.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("演示完成。请将真实的 LLM 调用接入 MemoryAugmentedAgent 使用。")
    print("=" * 60)