"""
MemoryBank — 统一接口实现

在原有 MemoryBank 基础上：
1. 继承 interfaces.MemorySystem 抽象基类
2. 实现 Mem0 风格的两阶段流水线（提取→更新）
3. 保留艾宾浩斯遗忘曲线 + Spacing Effect
4. 支持 filters 过滤检索
5. 预留 versioning 字段（status / version / parent_id）

同时提供 TFIDFMemoryBank 基线实现。
"""

import math
import time
import json
import pickle
import hashlib
from datetime import datetime
try:  # optional dependency; required only by the FAISS MemoryBank class
    import numpy as np
except ImportError:  # pragma: no cover - optional extra
    np = None  # type: ignore[assignment]

try:  # optional dependency; required only by the FAISS MemoryBank class
    import faiss
except ImportError:  # pragma: no cover - optional extra
    faiss = None  # type: ignore[assignment]
from typing import List, Dict, Optional, Tuple, Any, Callable

from statebudgetmem.core import (
    MemorySystem, MemoryPiece, MemoryType, MemoryStatus,
    UpdateOperation, filter_memories, messages_to_memory_pieces
)


# ═══════════════════════════════════════════════════════════════
# MemoryBank 核心实现（适配统一接口）
# ═══════════════════════════════════════════════════════════════

class MemoryBank(MemorySystem):
    """
    MemoryBank 长期记忆管理

    基于论文 "MemoryBank: Enhancing Large Language Models with Long-Term Memory" (AAAI-24)

    核心机制：
    - 向量检索（FAISS + sentence-transformers）
    - 艾宾浩斯遗忘曲线 R = e^(-t/S)
    - Spacing Effect（回忆强化）
    - 两阶段流水线（提取→更新）参考 Mem0
    """

    def __init__(
            self,
            embedding_dim: int = 384,
            forgetting_threshold: float = 0.3,
            llm_extractor: Optional[Callable] = None,
            embedding_model=None,
            decay_interval_hours: float = 24.0,
    ):
        if np is None or faiss is None:
            raise ImportError(
                "FAISS MemoryBank requires optional dependencies. "
                "Install with: pip install -e '.[memorybank]'"
            )
        self.embedding_dim = embedding_dim
        self.forgetting_threshold = forgetting_threshold
        self.decay_interval_sec = decay_interval_hours * 3600

        # Embedding 模型
        self._embedding_model = embedding_model
        self._embedding_model_local = None

        # LLM 提取器（用于两阶段流水线的提取阶段，可选）
        self.llm_extractor = llm_extractor

        # 记忆存储
        self.memories: List[MemoryPiece] = []
        self.memories_by_id: Dict[str, MemoryPiece] = {}  # memory_id -> MemoryPiece
        self.user_portrait: str = ""
        self.global_summary: str = ""

        # FAISS 向量索引（懒加载）
        self.index = None
        self.faiss_id_to_mid: Dict[int, str] = {}  # faiss_id -> memory_id
        self.next_faiss_id = 0

        # 统计
        self.access_count = 0
        self.forget_count = 0

    # ── Embedding ──

    def _get_embedding(self, text: str) -> np.ndarray:
        if self._embedding_model is not None:
            emb = self._embedding_model.encode(text)
        else:
            if self._embedding_model_local is None:
                from sentence_transformers import SentenceTransformer
                self._embedding_model_local = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"[MemoryBank] 已加载 embedding 模型: all-MiniLM-L6-v2")
            emb = self._embedding_model_local.encode(text)
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 0 else emb.astype(np.float32)

    def _ensure_index(self):
        if self.index is None:
            emb = self._get_embedding("test")
            self.embedding_dim = emb.shape[0]
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    def _generate_id(self, content: str, timestamp: float) -> str:
        """生成唯一 memory_id"""
        import hashlib
        return hashlib.md5(f"{content}_{timestamp}_{time.time()}".encode()).hexdigest()[:16]

    @staticmethod
    def _parse_time(value: str | float | int | None) -> float:
        """Parse timestamps accepted by the original MemoryBank API."""
        if value is None:
            return time.time()
        if isinstance(value, (int, float)):
            return float(value)
        value = value.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError as exc:
            raise ValueError(f"Unsupported timestamp: {value}") from exc

    def store(
        self,
        content: str,
        timestamp: str | float | int | None = None,
        memory_type: str | MemoryType = MemoryType.DIALOG,
    ) -> MemoryPiece:
        """Store one memory using the original baseline's public API."""
        ts = self._parse_time(timestamp)
        mt = memory_type if isinstance(memory_type, MemoryType) else MemoryType(memory_type)
        memory = MemoryPiece(
            content=content,
            timestamp=ts,
            memory_type=mt,
            last_accessed=ts,
            memory_id=self._generate_id(content, ts),
            tags=self._auto_tag(content),
        )
        self._insert_memory(memory)
        return memory

    def store_dialog(
        self, role: str, content: str, timestamp: str | float | int | None = None
    ) -> MemoryPiece:
        """Store one raw dialog turn."""
        return self.store(f"{role}: {content}", timestamp, MemoryType.DIALOG)

    def store_summary(
        self, summary: str, timestamp: str | float | int | None = None
    ) -> MemoryPiece:
        """Store one daily/session summary."""
        return self.store(summary, timestamp, MemoryType.SUMMARY)

    # ── 核心接口：add (两阶段流水线) ──

    def add(self, messages: List[Tuple[str, str, str]], **kwargs) -> List[str]:
        """
        添加对话，执行两阶段流水线：
        Phase 1: 提取（从消息中提取候选记忆）
        Phase 2: 更新（判断操作类型并执行）
        """
        # Phase 1: 提取
        candidates = self._extract(messages)

        # Phase 2: 更新
        updated_ids = []
        for candidate in candidates:
            operation, target_id = self._classify_operation(candidate)
            mid = self._execute_operation(operation, candidate, target_id)
            if mid:
                updated_ids.append(mid)

        return updated_ids

    def _extract(self, messages: List[Tuple[str, str, str]]) -> List[MemoryPiece]:
        """
        Phase 1: 提取候选记忆

        如果有 llm_extractor，用 LLM 提取事实；
        否则直接把每条对话当作一个候选记忆。
        """
        candidates = []

        if self.llm_extractor:
            # LLM 提取模式（Mem0 风格）
            # 构造 prompt，让 LLM 提取事实
            conversation_text = "\n".join([f"{role}: {content}" for role, content, _ in messages])
            prompt = f"""从以下对话中提取关键事实（偏好、身份、事件等），每条事实一句话：

{conversation_text}

提取的事实（每行一条）："""
            try:
                result = self.llm_extractor(prompt)
                facts = [line.strip("- ").strip() for line in result.split("\n") if line.strip()]
                for fact in facts:
                    ts = time.time()
                    mp = MemoryPiece(
                        content=fact,
                        timestamp=ts,
                        memory_type=MemoryType.FACT,
                        memory_id=self._generate_id(fact, ts),
                        tags=self._auto_tag(fact),
                    )
                    candidates.append(mp)
            except Exception as e:
                print(f"[提取失败] {e}，回退到原始消息模式")
                candidates = messages_to_memory_pieces(messages)
        else:
            # 原始消息模式
            candidates = messages_to_memory_pieces(messages)

        return candidates

    def _classify_operation(self, candidate: MemoryPiece) -> Tuple[UpdateOperation, Optional[str]]:
        """
        Phase 2a: 判断操作类型

        检索语义相似的现有记忆，决定是 ADD / UPDATE / NOOP
        （简化版，不调用 LLM，用阈值判断）
        """
        if not self.memories:
            return UpdateOperation.ADD, None

        # 检索最相似的 3 条
        similar = self.retrieve(candidate.content, top_k=3)

        if not similar:
            return UpdateOperation.ADD, None

        best_match = similar[0]
        best_score = best_match.get("semantic_score", 0)
        best_mid = best_match.get("memory_id", "")

        if best_score > 0.95:
            # 高度相似，可能是重复
            return UpdateOperation.NOOP, best_mid
        elif best_score > 0.75:
            # 语义相关，可能是更新
            return UpdateOperation.UPDATE, best_mid
        else:
            return UpdateOperation.ADD, None

    def _execute_operation(self, operation: UpdateOperation,
                           candidate: MemoryPiece,
                           target_id: Optional[str]) -> Optional[str]:
        """Phase 2b: 执行操作"""
        if operation == UpdateOperation.ADD:
            return self._insert_memory(candidate)
        elif operation == UpdateOperation.UPDATE and target_id:
            # 更新：给目标记忆补充信息
            target = self.memories_by_id.get(target_id)
            if target:
                target.content += f" | {candidate.content}"
                target.version += 1
                target.strength += 0.5
                return target_id
        elif operation == UpdateOperation.NOOP:
            # 重复，不做任何事
            return target_id
        return None

    def _insert_memory(self, memory: MemoryPiece) -> str:
        """插入一条新记忆到存储和索引"""
        self._ensure_index()

        # 生成 embedding
        emb = self._get_embedding(memory.content)
        memory.embedding = emb

        # 确保有 ID
        if not memory.memory_id:
            memory.memory_id = self._generate_id(memory.content, memory.timestamp)

        # 存入列表和字典
        self.memories.append(memory)
        self.memories_by_id[memory.memory_id] = memory

        # 存入 FAISS
        faiss_id = self.next_faiss_id
        self.index.add(emb.reshape(1, -1))
        self.faiss_id_to_mid[faiss_id] = memory.memory_id
        self.next_faiss_id += 1

        return memory.memory_id

    def _auto_tag(self, content: str) -> List[str]:
        """简单自动标签（基于关键词）"""
        tags = []
        content_lower = content.lower()
        tag_keywords = {
            "饮食": ["吃", "食物", "餐厅", "辣", "清淡", "火锅", "川菜"],
            "运动": ["篮球", "游泳", "跑步", "健身"],
            "学习": ["考试", "专业", "课程", "书", "学习"],
            "健康": ["医院", "医生", "病", "胃", "受伤"],
            "身份": ["名字", "职业", "专业", "学生"],
            "偏好": ["喜欢", "爱", "偏好", "兴趣"],
        }
        for tag, keywords in tag_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)
        return tags

    # ── 核心接口：retrieve ──

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        memory_types: Optional[List[str]] = None,
        current_time: str | float | int | None = None,
    ) -> List[Dict]:
        """
        检索记忆，支持过滤

        检索流程：
        1. FAISS 向量相似度检索
        2. 应用 filters（status / tags / valid_at 等）
        3. 计算综合分数（语义相似度 × 记忆强度 × 时间衰减）
        """
        self._ensure_index()
        if self.index.ntotal == 0:
            return []

        # 编码查询
        query_emb = self._get_embedding(query)

        # FAISS 检索
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_emb.reshape(1, -1), search_k)

        now = self._parse_time(current_time) if current_time is not None else time.time()
        candidates = []

        for dist, idx in zip(distances[0], indices[0]):
            mid = self.faiss_id_to_mid.get(int(idx))
            if not mid:
                continue
            mem = self.memories_by_id.get(mid)
            if not mem:
                continue
            if memory_types and mem.memory_type.value not in set(memory_types):
                continue

            # 计算综合分数
            age_hours = (now - mem.timestamp) / 3600.0
            time_decay = math.exp(-age_hours / 168)  # 一周衰减
            semantic_score = float(dist)
            composite_score = semantic_score * (1 + mem.strength * 0.3) * time_decay

            # Spacing Effect
            mem.strength += 1
            mem.last_accessed = now
            mem.access_count += 1
            self.access_count += 1

            candidates.append({
                "memory_id": mem.memory_id,
                "content": mem.content,
                "memory_type": mem.memory_type.value,
                "semantic_score": round(semantic_score, 4),
                "composite_score": round(composite_score, 4),
                "strength": mem.strength,
                "status": mem.status.value,
                "tags": mem.tags,
                "timestamp": mem.timestamp,
                "age_hours": round(age_hours, 1),
            })

        # 按综合分数排序
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        candidates = candidates[:top_k]

        # 应用 filters
        if filters:
            # 将 candidates 转回 MemoryPiece 进行过滤
            pieces = []
            for c in candidates:
                mp = self.memories_by_id.get(c["memory_id"])
                if mp:
                    pieces.append(mp)
            filtered = filter_memories(pieces, filters)
            filtered_ids = {m.memory_id for m in filtered}
            candidates = [c for c in candidates if c["memory_id"] in filtered_ids]

        return candidates

    # ── 核心接口：update ──

    def update(self, memory_id: str, operation: UpdateOperation, **kwargs):
        """
        更新记忆状态

        支持的操作：
        - SUPERSEDE: 标记为被替代，设置 validity_period end
        - TEMP_INVALIDATE: 标记为暂时失效
        - RESTORE: 恢复为 ACTIVE
        - DELETE: 软删除
        """
        mem = self.memories_by_id.get(memory_id)
        if not mem:
            print(f"[警告] 记忆 {memory_id} 不存在")
            return

        if operation == UpdateOperation.SUPERSEDE:
            mem.status = MemoryStatus.SUPERSEDED
            mem.validity_period = (mem.timestamp, kwargs.get("end_time", time.time()))
            if kwargs.get("new_memory_id"):
                mem.parent_id = kwargs["new_memory_id"]

        elif operation == UpdateOperation.TEMP_INVALIDATE:
            mem.status = MemoryStatus.TEMP_INVALID
            mem.validity_period = (mem.timestamp, kwargs.get("end_time"))

        elif operation == UpdateOperation.RESTORE:
            mem.status = MemoryStatus.ACTIVE
            mem.validity_period = (mem.timestamp, None)

        elif operation == UpdateOperation.DELETE:
            mem.status = MemoryStatus.DELETED

    # ── 其他接口 ──

    def get(self, memory_id: str) -> Optional[MemoryPiece]:
        return self.memories_by_id.get(memory_id)

    def get_all(self, filters: Optional[Dict] = None) -> List[MemoryPiece]:
        if filters:
            return filter_memories(self.memories, filters)
        return list(self.memories)

    def delete(self, memory_id: str, soft: bool = True):
        if soft:
            self.update(memory_id, UpdateOperation.DELETE)
        else:
            # 硬删除：从 FAISS 中移除需要重建索引，这里简化处理
            mem = self.memories_by_id.pop(memory_id, None)
            if mem and mem in self.memories:
                self.memories.remove(mem)

    def get_stats(self) -> Dict:
        type_counts = {}
        status_counts = {}
        for m in self.memories:
            type_counts[m.memory_type.value] = type_counts.get(m.memory_type.value, 0) + 1
            status_counts[m.status.value] = status_counts.get(m.status.value, 0) + 1

        return {
            "total_memories": len(self.memories),
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "total_access": self.access_count,
            "index_size": self.index.ntotal if self.index else 0,
            "user_portrait_set": bool(self.user_portrait),
            "global_summary_set": bool(self.global_summary),
        }

    def save(self, filepath: str):
        data = {
            "memories": [m.to_dict() for m in self.memories],
            "user_portrait": self.user_portrait,
            "global_summary": self.global_summary,
            "next_faiss_id": self.next_faiss_id,
            "access_count": self.access_count,
            "forget_count": self.forget_count,
        }
        with open(filepath + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.index is not None:
            faiss.write_index(self.index, filepath + ".faiss")
        print(f"[MemoryBank] 已保存 {len(self.memories)} 条记忆到 {filepath}")

    def load(self, filepath: str):
        with open(filepath + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.memories = [MemoryPiece.from_dict(m) for m in data["memories"]]
        self.memories_by_id = {m.memory_id: m for m in self.memories}
        self.user_portrait = data["user_portrait"]
        self.global_summary = data["global_summary"]
        self.next_faiss_id = data["next_faiss_id"]
        self.access_count = data["access_count"]
        self.forget_count = data["forget_count"]

        # 重建 FAISS 索引
        self.index = faiss.read_index(filepath + ".faiss")
        self.faiss_id_to_mid = {i: m.memory_id for i, m in enumerate(self.memories)}
        print(f"[MemoryBank] 已加载 {len(self.memories)} 条记忆")

    def update_user_portrait(self, portrait: str):
        self.user_portrait = portrait

    def update_global_summary(self, summary: str):
        self.global_summary = summary

    # ── 遗忘机制 ──

    def update_forgetting(
        self, current_time: str | float | int | None = None
    ):
        """执行艾宾浩斯遗忘更新"""
        if not self.memories:
            return []

        now = self._parse_time(current_time) if current_time is not None else time.time()
        forgotten = []

        for mem in self.memories:
            t = (now - mem.last_accessed) / 3600.0
            S = mem.strength
            retention = math.exp(-t / S) if S > 0 else 0

            if retention < self.forgetting_threshold:
                mem.strength *= 0.5
                forgotten.append({
                    "memory_id": mem.memory_id,
                    "content": mem.content[:50] + "...",
                    "retention": round(retention, 4),
                })
                self.forget_count += 1

        return forgotten

    def build_augmented_prompt(self, query: str, current_time: Optional[str] = None,
                               top_k: int = 5, include_portrait: bool = True) -> Dict[str, str]:
        """构建增强 prompt"""
        memories = self.retrieve(query, top_k=top_k, current_time=current_time)
        memory_text = "\n".join([
            f"[{m['memory_type']}] {m['content']}"
            for m in memories
        ]) if memories else "（无相关历史记忆）"

        context_parts = []
        if include_portrait and self.user_portrait:
            context_parts.append(f"【用户画像】{self.user_portrait}")
        if self.global_summary:
            context_parts.append(f"【事件摘要】{self.global_summary}")

        system_context = "\n".join(context_parts)

        prompt = f"""你是一个有长期记忆的AI助手。请参考以下信息来回答用户的问题。

{system_context}

【相关历史记忆】
{memory_text}

【用户当前问题】
{query}

请基于历史记忆和当前问题，给出个性化、连贯的回答。"""

        return {
            "system_context": system_context,
            "relevant_memories": memory_text,
            "prompt_template": prompt,
            "retrieved_count": len(memories),
        }


# ═══════════════════════════════════════════════════════════════
# TF-IDF 基线实现
# ═══════════════════════════════════════════════════════════════

class TFIDFMemoryBank(MemorySystem):
    """
    TF-IDF 基线记忆系统

    不使用神经网络 Embedding，用传统 TF-IDF 做关键词匹配。
    作为 MemoryBank 的对照组，验证向量检索的价值。
    """

    def __init__(self, forgetting_threshold: float = 0.3):
        self.forgetting_threshold = forgetting_threshold
        self.memories: List[MemoryPiece] = []
        self.memories_by_id: Dict[str, MemoryPiece] = {}

        # TF-IDF 相关
        self.vectorizer = None
        self.tfidf_matrix = None
        self._needs_rebuild = True

        # 统计
        self.access_count = 0

    def _build_tfidf(self):
        """重新构建 TF-IDF 矩阵"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self.memories:
            return

        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 2), max_features=5000
        )
        contents = [m.content for m in self.memories]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
        self._needs_rebuild = False

    def add(self, messages: List[Tuple[str, str, str]], **kwargs) -> List[str]:
        """添加对话记录"""
        pieces = messages_to_memory_pieces(messages)
        mids = []
        for p in pieces:
            if not p.memory_id:
                p.memory_id = hashlib.md5(f"{p.content}_{p.timestamp}".encode()).hexdigest()[:16]
            self.memories.append(p)
            self.memories_by_id[p.memory_id] = p
            mids.append(p.memory_id)

        self._needs_rebuild = True
        return mids

    def retrieve(self, query: str, top_k: int = 5,
                 filters: Optional[Dict] = None) -> List[Dict]:
        """TF-IDF 检索"""
        if not self.memories:
            return []

        if self._needs_rebuild:
            self._build_tfidf()

        if self.tfidf_matrix is None:
            return []

        # 编码查询
        query_vec = self.vectorizer.transform([query])

        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 获取 top_k 索引
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue
            mem = self.memories[idx]
            results.append({
                "memory_id": mem.memory_id,
                "content": mem.content,
                "memory_type": mem.memory_type.value,
                "semantic_score": round(float(similarities[idx]), 4),
                "composite_score": round(float(similarities[idx]), 4),
                "strength": mem.strength,
                "status": mem.status.value,
                "tags": mem.tags,
                "timestamp": mem.timestamp,
            })

        # 应用 filters
        if filters:
            pieces = [self.memories_by_id[r["memory_id"]] for r in results
                      if r["memory_id"] in self.memories_by_id]
            filtered = filter_memories(pieces, filters)
            filtered_ids = {m.memory_id for m in filtered}
            results = [r for r in results if r["memory_id"] in filtered_ids]

        return results

    def update(self, memory_id: str, operation: UpdateOperation, **kwargs):
        mem = self.memories_by_id.get(memory_id)
        if not mem:
            return
        if operation == UpdateOperation.DELETE:
            mem.status = MemoryStatus.DELETED
        elif operation == UpdateOperation.SUPERSEDE:
            mem.status = MemoryStatus.SUPERSEDED
        elif operation == UpdateOperation.RESTORE:
            mem.status = MemoryStatus.ACTIVE

    def get(self, memory_id: str) -> Optional[MemoryPiece]:
        return self.memories_by_id.get(memory_id)

    def get_all(self, filters: Optional[Dict] = None) -> List[MemoryPiece]:
        if filters:
            return filter_memories(self.memories, filters)
        return list(self.memories)

    def delete(self, memory_id: str, soft: bool = True):
        if soft:
            self.update(memory_id, UpdateOperation.DELETE)
        else:
            mem = self.memories_by_id.pop(memory_id, None)
            if mem and mem in self.memories:
                self.memories.remove(mem)
            self._needs_rebuild = True

    def get_stats(self) -> Dict:
        return {
            "total_memories": len(self.memories),
            "type_distribution": {},
            "status_distribution": {},
            "total_access": self.access_count,
            "index_size": len(self.memories),
        }

    def save(self, filepath: str):
        data = {
            "memories": [m.to_dict() for m in self.memories],
            "access_count": self.access_count,
        }
        with open(filepath + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.memories = [MemoryPiece.from_dict(m) for m in data["memories"]]
        self.memories_by_id = {m.memory_id: m for m in self.memories}
        self.access_count = data.get("access_count", 0)
        self._needs_rebuild = True
