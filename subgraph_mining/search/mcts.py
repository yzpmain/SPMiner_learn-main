"""MCTS搜索策略。

提供基于蒙特卡洛树搜索的子图模式搜索实现。
"""

import random
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
from tqdm import tqdm

from core.config.device import get_device
from core.utils.graph import wl_hash
from subgraph_mining.search.base import SearchAgent


class MCTSSearchAgent(SearchAgent):
    """子图模式搜索的 MCTS 实现。

    使用 MCTS 策略搜索最常见的模式。
    """

    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
                 embs, node_anchored=False, analyze=False, model_type="order",
                 out_batch_size=20, c_uct=0.7, frontier_top_k=0):
        """
        参数说明：
            c_uct: UCT 准则中使用的探索常数（参见论文）。
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
                         embs, node_anchored=node_anchored, analyze=analyze,
                         model_type=model_type, out_batch_size=out_batch_size,
                         frontier_top_k=frontier_top_k)
        self.c_uct = c_uct
        assert not analyze

    def init_search(self):
        """初始化 MCTS 运行时缓存。"""
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()
        self.max_size = self.min_pattern_size

    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1

    def has_min_reachable_nodes(self, graph, start_node, n):
        """返回从 start_node 起至少有 n 个可达节点。"""
        for depth_limit in range(n + 1):
            edges = nx.bfs_edges(graph, start_node, depth_limit=depth_limit)
            nodes = set([v for u, v in edges])
            if len(nodes) + 1 >= n:
                return True
        return False

    def step(self):
        """执行一轮 MCTS 扩展与价值回传。"""
        ps = np.array([len(g) for g in self.dataset], dtype=float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        for simulation_n in tqdm(range(self.n_trials //
                                       (self.max_pattern_size + 1 - self.min_pattern_size))):
            # 选择种子节点
            best_graph_idx, best_start_node, best_score = None, None, -float("inf")
            for cand_graph_idx, cand_start_node in self.visited_seed_nodes:
                state = cand_graph_idx, cand_start_node
                my_visit_counts = sum(self.visit_counts[state].values())
                q_score = (sum(self.cum_action_values[state].values()) /
                           (my_visit_counts or 1))
                uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) /
                                                 (my_visit_counts or 1))
                node_score = q_score + uct_score
                if node_score > best_score:
                    best_score = node_score
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node
            # 如果现有种子节点优于选择新种子节点
            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                assert best_start_node in self.dataset[graph_idx].nodes
                graph = self.dataset[graph_idx]
            else:
                found = False
                while not found:
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    # 不选择孤立节点或小的连通分量
                    if self.has_min_reachable_nodes(graph, start_node,
                                                    self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            neigh_g = nx.Graph()
            neigh_g.add_node(start_node, anchor=1)
            cur_state = graph_idx, start_node
            state_list = [cur_state]
            while frontier and len(neigh) < self.max_size:
                frontier = self._prune_frontier(graph, frontier)
                cand_embs = self._get_candidate_embs(graph_idx, graph, neigh,
                                                     frontier)
                best_v_score, best_node_score, best_node = 0, -float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    cand_emb = cand_emb.to(get_device())
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        score += torch.sum(self.model.predict((
                            emb_batch.to(get_device()), cand_emb))).item()
                        n_embs += len(emb_batch)
                    v_score = -np.log(score / n_embs + 1) + 1
                    # 获取下一状态的 WL 哈希值
                    neigh_g = graph.subgraph(neigh + [cand_node]).copy()
                    neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                    for v in neigh_g.nodes:
                        neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                    next_state = wl_hash(neigh_g,
                                         node_anchored=self.node_anchored)
                    # 计算节点分数
                    parent_visit_counts = sum(self.visit_counts[cur_state].values())
                    my_visit_counts = sum(self.visit_counts[next_state].values())
                    q_score = (sum(self.cum_action_values[next_state].values()) /
                               (my_visit_counts or 1))
                    uct_score = self.c_uct * np.sqrt(np.log(parent_visit_counts or
                                                            1) / (my_visit_counts or 1))
                    node_score = q_score + uct_score
                    if node_score > best_node_score:
                        best_node_score = node_score
                        best_v_score = v_score
                        best_node = cand_node
                frontier = list(((set(frontier) |
                                  set(graph.neighbors(best_node))) - visited) -
                                set([best_node]))
                visited.add(best_node)
                neigh.append(best_node)

                # 更新访问次数和 WL 缓存
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                prev_state = cur_state
                cur_state = wl_hash(neigh_g, node_anchored=self.node_anchored)
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)

            # 反向传播价值：将末端评分 best_v_score 沿路径累计。
            for i in range(0, len(state_list) - 1):
                self.cum_action_values[state_list[i]][
                    state_list[i + 1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i + 1]] += 1
        self.max_size += 1

    def finish_search(self):
        """按访问统计选出每个尺寸的高频候选模式。"""
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            for wl_hash_val, count in sorted(counts[pattern_size].items(), key=lambda
                    x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash_val]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq
