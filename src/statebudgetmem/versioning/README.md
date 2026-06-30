# 状态版本管理

本目录负责将普通文本记忆组织为带有时间和版本关系的状态记忆。

主要操作：

- ADD
- MERGE
- SUPERSEDE
- TEMP_INVALIDATE
- DELETE
- NOOP

本阶段需要实现：

1. 新旧记忆匹配；
2. 更新操作判断；
3. 有效时间维护；
4. supersedes 和 temporarily_invalidates 关系；
5. 当前状态解析；
6. 版本管理相关测试。
