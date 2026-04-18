"""IO工具模块。

提供图数据加载和保存功能。
"""

import networkx as nx


def load_snap_edgelist(path):
    """从 SNAP 风格的边列表文件中加载无向图。

    支持空格或制表符分隔的边，自动跳过空行和以 '#' 开头的注释行。
    返回最大连通子图，以确保采样操作的一致性。

    Args:
        path: 边列表文件路径（每行格式为 "节点1 节点2"）

    Returns:
        最大连通子图（networkx.Graph）
    """
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                graph.add_edge(int(parts[0]), int(parts[1]))
    # 取最大连通子图，保证子图采样的连通性
    if not nx.is_connected(graph):
        graph = graph.subgraph(
            max(nx.connected_components(graph), key=len)
        ).copy()
    return graph
