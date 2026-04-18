"""图操作工具模块。

提供图采样、WL哈希、子图枚举等图操作功能。
"""

import random
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

cached_masks = None


def sample_neigh(graphs, size):
    """在图集合中按图大小加权采样一个连通邻域。

    采样步骤：
    1. 按 |V| 作为权重选择一张图（大图被选中概率更高）。
    2. 在图中随机选择起点并进行前沿扩展，直到达到给定 size。
    3. 若前沿耗尽导致节点不足，则重新采样。

    Args:
        graphs: networkx 图列表。
        size: 目标邻域节点数。

    Returns:
        (graph, neigh_nodes) 其中 neigh_nodes 为采样到的节点列表。
    """
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


def vec_hash(v):
    """将向量映射为稳定的哈希特征。

    这里通过固定随机掩码与 Python 哈希组合，构造可重复的离散编码，
    用于后续 WL 迭代中的节点表征更新。
    """
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    return v


def wl_hash(g, dim=64, node_anchored=False):
    """计算图的 WL 风格哈希签名。

    该实现使用固定维度的离散向量做迭代聚合，最终把节点向量求和后
    转成 tuple 作为"结构签名"，用于把同构/近同构候选归并计数。

    Args:
        g: networkx.Graph。
        dim: 哈希向量维度。
        node_anchored: 是否使用 anchor 节点信息。

    Returns:
        可哈希的 tuple 签名。
    """
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v].get("anchor", 0) == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                                         axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))


def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    """基于 ESU 思想枚举子图并按 WL 签名聚类。"""
    ps = np.arange(1.0, 0.0, -1.0 / (k + 1)) ** 1.5
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts


def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    """递归扩展当前子图并记录到 motif_counts。"""
    # 基础情形
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
                                  node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # 递归步骤：
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
                     not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
                        node_anchored)
        sg.remove(w)
