"""
routing.models — 查询数据模型 (Pydantic v2)

本模块定义 ``QueryRecord``, 它是 routing 模块的标准输入:
封装用户查询文本 + 参考时间 (reference_time) + 可选上下文。

为什么用 Pydantic 而非 dataclass?
    - Pydantic 提供严格的类型校验与友好的错误信息, 适合作为系统边界的数据契约。
    - 与项目中 ``baselines`` 子包使用的 dataclass ``MemoryPiece`` 互补:
      ``MemoryPiece`` 是内部记忆格式, ``QueryRecord`` 是外部查询格式。

注意：该模型复用 ``schemas.QueryType``，不再维护独立的查询类型枚举。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from statebudgetmem.schemas import QueryType


def _parse_reference_time(value: Any) -> datetime:
    """
    把多种格式的 reference_time 统一成 aware datetime (UTC)。

    支持:
        - datetime        : 直接使用 (naive 视作 UTC)
        - int / float     : Unix 时间戳 (秒)
        - str (ISO 8601)  : "2026-06-20T10:00:00" / "2026-06-20 10:00:00"
        - str (数字)      : "1719360000"
        - str (常见格式)  : "%Y-%m-%d %H:%M" / "%Y-%m-%d %H:%M:%S" / "%Y-%m-%d"
        - None            : 默认为当前 UTC 时间
    """
    if value is None:
        return datetime.now(timezone.utc)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if isinstance(value, str):
        s = value.strip()
        # 1) 纯数字字符串 → Unix 时间戳
        try:
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        except ValueError:
            pass
        # 2) ISO 8601 (容忍末尾 Z)
        iso = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
        # 3) 常见显式格式
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    raise ValueError(
        f"无法解析 reference_time={value!r}; 支持 datetime / unix 时间戳 / ISO 字符串"
    )


class RoutingQueryRecord(BaseModel):
    """
    查询记录 —— routing 模块的标准输入。

    字段说明:
        text           : 用户原始查询文本 (必填)。
        reference_time : 查询的"参考时间", 即用户提问的时刻。
                         用于判断"现在/过去/变化"等时态线索。
                         缺省为当前 UTC 时间。
        context        : 可选的对话上下文 (前几轮摘要), 帮助消歧。
        user_id        : 可选的用户标识, 便于多用户路由。
        metadata       : 任意附加元数据 (如会话 ID), 不参与分类逻辑。

    示例::

        >>> q = QueryRecord(text="我现在还喜欢吃辣吗?", reference_time="2026-09-01 10:00")
        >>> q.text
        '我现在还喜欢吃辣吗?'
        >>> q.reference_time.year
        2026
    """

    text: str = Field(..., description="用户查询文本")
    reference_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="查询参考时间 (用于时态判断)",
    )
    context: Optional[str] = Field(
        default=None, description="可选对话上下文, 用于消歧"
    )
    user_id: Optional[str] = Field(default=None, description="可选用户标识")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="附加元数据"
    )

    # ── 校验器 ───────────────────────────────────────────────────
    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        # 保留原始内容, 仅去除首尾空白; 允许空串 (由 router 决定如何降级)
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="before")
    @classmethod
    def _coerce_reference_time(cls, data: Any) -> Any:
        """在模型构造前把 reference_time 统一成 datetime。"""
        if isinstance(data, dict) and "reference_time" in data:
            data = dict(data)  # shallow copy, 不污染调用方
            data["reference_time"] = _parse_reference_time(data["reference_time"])
        return data

    # ── 便捷方法 ─────────────────────────────────────────────────
    def is_empty(self) -> bool:
        """查询文本是否为空 (含纯空白)。"""
        return not self.text or not self.text.strip()

    def to_prompt_dict(self) -> dict[str, str]:
        """
        生成供 LLM prompt 使用的精简字典。

        时间格式化为 ``%Y-%m-%d %H:%M`` 便于模型理解。
        """
        return {
            "text": self.text,
            "reference_time": self.reference_time.strftime("%Y-%m-%d %H:%M"),
            "context": self.context or "",
        }


QueryRecord = RoutingQueryRecord

__all__ = ["RoutingQueryRecord", "QueryRecord", "QueryType"]
