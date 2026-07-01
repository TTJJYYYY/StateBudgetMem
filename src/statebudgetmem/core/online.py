"""
StateBudgetMem — 统一接口规范

本模块定义了记忆系统的抽象接口和数据格式，所有子模块（baselines / versioning / views / routing）
必须遵循这些接口进行实现和对接。

设计参考：
- MemoryBank (AAAI-24): 艾宾浩斯遗忘曲线 + 向量检索
- Mem0 (arXiv 2025): 两阶段流水线 (提取→更新) + Tool Call 机制
- MemGPT: 分层记忆系统
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from statebudgetmem.schemas.records import QueryType


# ═══════════════════════════════════════════════════════════════
# 枚举类型
# ═══════════════════════════════════════════════════════════════

class MemoryType(Enum):
    """记忆类型"""
    DIALOG = "dialog"           # 原始对话
    SUMMARY = "summary"         # 摘要（每日/会话级）
    PORTRAIT = "portrait"       # 用户画像
    FACT = "fact"               # 提取的事实（Mem0 风格）
    EVENT = "event"             # 事件
    PREFERENCE = "preference"   # 偏好


class MemoryStatus(Enum):
    """记忆状态 —— 供 versioning 模块使用"""
    ACTIVE = "active"               # 当前有效
    SUPERSEDED = "superseded"       # 已被替代（新记忆取代旧记忆）
    TEMP_INVALID = "temp_invalid"   # 暂时失效（条件性失效）
    DELETED = "deleted"             # 已删除
    CONFLICTING = "conflicting"     # 存在冲突，待解决


class UpdateOperation(Enum):
    """记忆更新操作类型 —— 参考 Mem0 的 Tool Call 设计"""
    ADD = "add"                     # 新增记忆
    UPDATE = "update"               # 更新现有记忆（补充信息）
    DELETE = "delete"               # 删除记忆
    NOOP = "noop"                   # 无操作
    MERGE = "merge"                 # 合并两条记忆
    SUPERSEDE = "supersede"         # 替代（旧记忆标记为 SUPERSEDED）
    TEMP_INVALIDATE = "temp_invalidate"  # 暂时失效
    RESTORE = "restore"             # 恢复失效记忆




class ViewType(Enum):
    """视图类型 —— 供 views 模块使用。NONE 表示不检索个人记忆。"""
    NONE = "none"
    CURRENT = "current"     # 当前有效状态视图
    HISTORY = "history"     # 完整历史版本视图
    BOTH = "both"           # 双视图


# ═══════════════════════════════════════════════════════════════
# 统一数据格式：MemoryPiece
# ═══════════════════════════════════════════════════════════════

@dataclass
class MemoryPiece:
    """
    统一记忆格式 —— 所有模块共享

    设计原则：
    1. 基础字段（content / timestamp / memory_type）所有模块都用到
    2. 版本管理字段（memory_id / version / parent_id / status）由 versioning 模块维护
    3. 视图字段（tags / confidence）由 views 模块使用
    4. 路由字段（query_types）由 routing 模块标记
    """

    # ── 基础字段（必填）──
    content: str                    # 记忆内容（自然语言文本）
    timestamp: float                # 创建时间戳（unix time）
    memory_type: MemoryType = MemoryType.DIALOG  # 记忆类型

    # ── 向量表示（可选，由 baselines 模块填充）──
    embedding: Optional[Any] = None  # 向量表示

    # ── 检索相关（baselines 维护）──
    strength: float = 1.0           # 记忆强度 S（艾宾浩斯遗忘曲线）
    last_accessed: float = 0.0      # 上次被回忆的时间戳
    access_count: int = 0           # 被回忆次数

    # ── 版本管理字段（versioning 模块维护）──
    memory_id: str = ""             # 唯一标识（UUID）
    version: int = 1                # 版本号
    parent_id: Optional[str] = None # 父记忆ID（形成版本链）
    status: MemoryStatus = MemoryStatus.ACTIVE  # 当前状态
    validity_period: Optional[Tuple[float, Optional[float]]] = None  # (start, end)，end=None 表示至今有效

    # ── 视图字段（views 模块维护）──
    tags: List[str] = field(default_factory=list)  # 标签，如 ["饮食", "偏好", "健康"]
    confidence: float = 1.0         # 记忆可信度 [0, 1]
    source: Optional[str] = None    # 来源（对话ID、摘要ID等）

    # ── 路由标记（routing 模块写入）──
    query_types: List[QueryType] = field(default_factory=list)  # 适用的查询类型

    def to_dict(self) -> dict:
        """序列化为字典（不含 embedding）"""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type.value,
            "strength": self.strength,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "memory_id": self.memory_id,
            "version": self.version,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "validity_period": self.validity_period,
            "tags": self.tags,
            "confidence": self.confidence,
            "source": self.source,
            "query_types": [qt.value for qt in self.query_types],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryPiece":
        """从字典反序列化"""
        return cls(
            content=d.get("content", ""),
            timestamp=d.get("timestamp", 0.0),
            memory_type=MemoryType(d.get("memory_type", "dialog")),
            strength=d.get("strength", 1.0),
            last_accessed=d.get("last_accessed", 0.0),
            access_count=d.get("access_count", 0),
            memory_id=d.get("memory_id", ""),
            version=d.get("version", 1),
            parent_id=d.get("parent_id"),
            status=MemoryStatus(d.get("status", "active")),
            validity_period=tuple(d["validity_period"]) if d.get("validity_period") else None,
            tags=d.get("tags", []),
            confidence=d.get("confidence", 1.0),
            source=d.get("source"),
            query_types=[QueryType(qt) for qt in d.get("query_types", [])],
        )

    def is_active(self) -> bool:
        """当前是否有效"""
        return self.status == MemoryStatus.ACTIVE

    def is_valid_at(self, query_time: float) -> bool:
        """在指定时间是否有效（考虑 validity_period）"""
        if self.validity_period is None:
            return self.is_active()
        start, end = self.validity_period
        if start and query_time < start:
            return False
        if end and query_time > end:
            return False
        return True


# ═══════════════════════════════════════════════════════════════
# 抽象接口：MemorySystem（所有记忆系统的基类）
# ═══════════════════════════════════════════════════════════════

class MemorySystem(ABC):
    """
    记忆系统抽象基类

    所有记忆系统实现（MemoryBank、TFIDFMemory、Mem0-style等）必须遵守此接口。
    这个接口设计参考了 Mem0 的简洁风格，同时保留了 MemoryBank 的遗忘机制。
    """

    @abstractmethod
    def add(self, messages: List[Tuple[str, str, str]], **kwargs) -> List[str]:
        """
        添加对话记录，自动提取和更新记忆

        参考 Mem0 的两阶段流水线：
        1. 提取阶段：从 messages 中提取候选记忆
        2. 更新阶段：判断每条候选记忆的操作类型（ADD/UPDATE/DELETE/NOOP）

        Args:
            messages: [(role, content, timestamp), ...]
            **kwargs: 额外参数（如用户ID、会话ID等）

        Returns:
            新增/更新的 memory_id 列表
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5,
                 filters: Optional[Dict] = None) -> List[Dict]:
        """
        检索记忆

        Args:
            query: 查询文本
            top_k: 返回最相关的 k 条
            filters: 过滤条件，如 {
                "status": "active",
                "tags": ["饮食"],
                "memory_type": "fact",
                "valid_at": 1719360000  # 在指定时间有效的记忆
            }

        Returns:
            记忆列表，每条包含 memory_id, content, score, status, tags 等
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, operation: UpdateOperation, **kwargs):
        """
        更新记忆状态 —— versioning 模块的核心调用点

        Args:
            memory_id: 目标记忆ID
            operation: 操作类型（ADD/UPDATE/DELETE/SUPERSEDE/TEMP_INVALIDATE/RESTORE）
            **kwargs: 额外参数
                - new_content: UPDATE 时的新内容
                - new_memory_id: SUPERSEDE 时的新记忆ID
                - reason: 操作原因
                - validity_end: TEMP_INVALIDATE 时的失效截止时间
        """
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryPiece]:
        """获取单条记忆的完整信息"""
        pass

    @abstractmethod
    def get_all(self, filters: Optional[Dict] = None) -> List[MemoryPiece]:
        """获取所有记忆（可选过滤）"""
        pass

    @abstractmethod
    def delete(self, memory_id: str, soft: bool = True):
        """
        删除记忆

        Args:
            memory_id: 目标记忆ID
            soft: True=软删除（标记为 DELETED），False=硬删除
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """返回记忆库统计信息"""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """持久化记忆库到文件"""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """从文件加载记忆库"""
        pass


# ═══════════════════════════════════════════════════════════════
# 辅助接口：其他模块需要实现的抽象类
# ═══════════════════════════════════════════════════════════════

class VersionManager(ABC):
    """
    版本管理器 —— versioning 模块实现

    负责判断新记忆与旧记忆的关系，执行版本操作。
    """

    @abstractmethod
    def classify_relationship(self, new_memory: MemoryPiece,
                              existing_memories: List[MemoryPiece]) -> UpdateOperation:
        """
        判断新记忆与现有记忆的关系

        Returns:
            ADD / UPDATE / DELETE / SUPERSEDE / MERGE / NOOP
        """
        pass

    @abstractmethod
    def build_version_chain(self, memory_id: str) -> List[MemoryPiece]:
        """构建某条记忆的版本链（从最早到最新）"""
        pass

    @abstractmethod
    def detect_conflicts(self, memory: MemoryPiece) -> List[MemoryPiece]:
        """检测与给定记忆冲突的其他记忆"""
        pass


class ViewManager(ABC):
    """
    视图管理器 —— views 模块实现

    负责维护 Current View 和 History View。
    """

    @abstractmethod
    def get_current_view(self, **filters) -> List[MemoryPiece]:
        """获取当前有效状态的视图（只包含 ACTIVE 的记忆）"""
        pass

    @abstractmethod
    def get_history_view(self, memory_id: Optional[str] = None) -> List[MemoryPiece]:
        """获取完整历史视图（包含 SUPERSEDED / TEMP_INVALID 的记忆）"""
        pass

    @abstractmethod
    def sync_views(self, operation: UpdateOperation, memory: MemoryPiece):
        """当记忆更新时，同步更新两个视图"""
        pass


class QueryRouter(ABC):
    """
    查询路由器 —— routing 模块实现

    负责判断查询类型，路由到合适的视图。
    """

    @abstractmethod
    def classify_query(self, query: str, context: Optional[str] = None) -> QueryType:
        """
        判断查询类型

        Returns:
            CURRENT / HISTORICAL / CHANGE / GENERAL
        """
        pass

    @abstractmethod
    def route(self, query: str, query_type: Optional[QueryType] = None) -> ViewType:
        """
        根据查询类型决定使用哪个视图

        Returns:
            NONE / CURRENT / HISTORY / BOTH
        """
        pass

    @abstractmethod
    def check_relevance(self, memory: MemoryPiece, query: str) -> float:
        """判断某条记忆对当前问题的相关性 [0, 1]"""
        pass


# ═══════════════════════════════════════════════════════════════
# 数据转换工具函数
# ═══════════════════════════════════════════════════════════════

def _parse_timestamp(ts) -> float:
    """解析各种格式的时间戳为 unix float"""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        # 尝试解析常见格式
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                from datetime import datetime
                return datetime.strptime(ts, fmt).timestamp()
            except ValueError:
                continue
        # 如果都失败，尝试直接 float（可能是字符串数字）
        try:
            return float(ts)
        except ValueError:
            pass
    return 0.0


def messages_to_memory_pieces(messages: List[Tuple[str, str, str]],
                                 memory_type: MemoryType = MemoryType.DIALOG) -> List[MemoryPiece]:
    """
    将原始消息列表转换为 MemoryPiece 列表

    Args:
        messages: [(role, content, timestamp), ...]
        memory_type: 默认记忆类型

    Returns:
        MemoryPiece 列表
    """
    pieces = []
    for role, content, timestamp in messages:
        # 生成简单 ID：role_timestamp_hash
        import hashlib
        ts_float = _parse_timestamp(timestamp)
        ts_str = str(timestamp)
        mid = hashlib.md5(f"{role}_{content}_{ts_str}".encode()).hexdigest()[:12]

        pieces.append(MemoryPiece(
            content=f"{role}: {content}",
            timestamp=ts_float,
            memory_type=memory_type,
            memory_id=mid,
            source=f"session_{ts_str}",
        ))
    return pieces


def filter_memories(memories: List[MemoryPiece],
                    filters: Dict) -> List[MemoryPiece]:
    """
    通用记忆过滤函数

    Args:
        memories: 记忆列表
        filters: {
            "status": "active" / ["active", "superseded"],
            "tags": ["饮食"],
            "memory_type": "fact",
            "valid_at": 1719360000,
            "min_confidence": 0.5,
        }
    """
    result = memories

    if "status" in filters:
        target = filters["status"]
        if isinstance(target, str):
            target = [target]
        result = [m for m in result if m.status.value in target]

    if "tags" in filters:
        required_tags = set(filters["tags"])
        result = [m for m in result if required_tags & set(m.tags)]

    if "memory_type" in filters:
        mt = filters["memory_type"]
        if isinstance(mt, str):
            mt = MemoryType(mt)
        result = [m for m in result if m.memory_type == mt]

    if "valid_at" in filters:
        t = filters["valid_at"]
        result = [m for m in result if m.is_valid_at(t)]

    if "min_confidence" in filters:
        thresh = filters["min_confidence"]
        result = [m for m in result if m.confidence >= thresh]

    return result
