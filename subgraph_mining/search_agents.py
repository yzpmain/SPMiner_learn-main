"""搜索代理模块 - 兼容性包装。

该模块现在从subgraph_mining.search重新导出，保留以维持向后兼容。
建议新代码使用：
    from subgraph_mining.search import GreedySearchAgent, MCTSSearchAgent
"""

import warnings

warnings.warn(
    "search_agents模块已重构为subgraph_mining.search包。"
    "请使用 'from subgraph_mining.search import ...'。"
    "此兼容性层将在未来版本中移除。",
    DeprecationWarning,
    stacklevel=2
)

# 从新的search包重新导出
from subgraph_mining.search import SearchAgent, GreedySearchAgent, MCTSSearchAgent

__all__ = ['SearchAgent', 'GreedySearchAgent', 'MCTSSearchAgent']
