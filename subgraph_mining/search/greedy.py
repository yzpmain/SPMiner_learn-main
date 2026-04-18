"""贪心搜索策略。

提供基于贪心策略的子图模式搜索实现。
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
from tqdm import tqdm

from core.config.device import get_device
from core.utils.batch import batch_nx_graphs
from core.utils.graph import wl_hash
from subgraph_mining.search.base import SearchAgent


class GreedySearchAgent(SearchAgent):
    """子图模式搜索的贪心实现。

    算法在每一步贪心地选择下一个节点进行扩展，同时保持模式
    被预测为频繁的。选择下一动作的标准取决于子图匹配模型预测的分数
    （实际分数由 rank_method 参数决定）。
    """

    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
                 embs, node_anchored=False, analyze=False, rank_method="counts",
                 model_type="order", out_batch_size=20, n_beams=1,
                 frontier_top_k=0):
        """
        参数说明：
            rank_method: 贪心搜索启发式需要一个分数来对可能的下一动作排序。
                如果 rank_method=='counts'，使用搜索树中该模式的计数；
                如果 rank_method=='margin'，使用匹配模型预测的该模式的 margin 分数；
                如果 rank_method=='hybrid'，同时考虑计数和 margin 对动作排序。
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
                         embs, node_anchored=node_anchored, analyze=analyze,
                         model_type=model_type, out_batch_size=out_batch_size,
                         frontier_top_k=frontier_top_k)
        self.rank_method = rank_method
        self.n_beams = n_beams
        print("Rank Method:", rank_method)

    def init_search(self):
        """初始化贪心搜索 beam。"""
        ps = np.array([len(g) for g in self.dataset], dtype=float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        for trial in range(self.n_trials):
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx]
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            beams.append([(0, neigh, frontier, visited, graph_idx)])
        self.beam_sets = beams
        self.analyze_embs = []

    def is_search_done(self):
        return len(self.beam_sets) == 0

    def step(self):
        """执行一轮贪心扩展。

        对每个 beam 状态，枚举 frontier 候选节点，
        基于匹配模型分数选择最优扩展。
        """
        new_beam_sets = []
        print("seeds come from", len(set(b[0][-1] for b in self.beam_sets)),
              "distinct graphs")
        analyze_embs_cur = []
        for beam_set in tqdm(self.beam_sets):
            new_beams = []
            for _, neigh, frontier, visited, graph_idx in beam_set:
                graph = self.dataset[graph_idx]
                if len(neigh) >= self.max_pattern_size or not frontier:
                    continue
                frontier = self._prune_frontier(graph, frontier)
                cand_embs = self._get_candidate_embs(graph_idx, graph, neigh,
                                                     frontier)
                best_score, best_node = float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    cand_emb = cand_emb.to(get_device())
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        n_embs += len(emb_batch)
                        if self.model_type == "order":
                            score -= torch.sum(torch.argmax(
                                self.model.clf_model(self.model.predict((
                                    emb_batch.to(get_device()),
                                    cand_emb)).unsqueeze(1)), axis=1)).item()
                        elif self.model_type == "mlp":
                            score += torch.sum(self.model(
                                emb_batch.to(get_device()),
                                cand_emb.unsqueeze(0).expand(len(emb_batch), -1)
                            )[:, 0]).item()
                        else:
                            print("未识别的模型类型")
                    if score < best_score:
                        best_score = score
                        best_node = cand_node
                    new_frontier = list(((set(frontier) |
                                          set(graph.neighbors(cand_node))) - visited) -
                                        set([cand_node]))
                    new_beams.append((
                        score, neigh + [cand_node],
                        new_frontier, visited | set([cand_node]), graph_idx))
            new_beams = list(sorted(new_beams, key=lambda x:
            x[0]))[:self.n_beams]
            for score, neigh, frontier, visited, graph_idx in new_beams[:1]:
                graph = self.dataset[graph_idx]
                # 添加到记录
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                self.cand_patterns[len(neigh_g)].append((score, neigh_g))
                if self.rank_method in ["counts", "hybrid"]:
                    self.counts[len(neigh_g)][wl_hash(neigh_g,
                                                      node_anchored=self.node_anchored)].append(neigh_g)
                if self.analyze and len(neigh) >= 3:
                    emb = self.model.emb_model(batch_nx_graphs(
                        [neigh_g], anchors=[neigh[0]] if self.node_anchored
                        else None)).squeeze(0)
                    analyze_embs_cur.append(emb.detach().cpu().numpy())
            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)
        self.beam_sets = new_beam_sets
        self.analyze_embs.append(analyze_embs_cur)

    def finish_search(self):
        """根据 rank_method 汇总并去重输出模式。"""
        import pickle

        if self.analyze:
            print("Saving analysis info in results/analyze.p")
            with open("results/analyze.p", "wb") as f:
                pickle.dump((self.cand_patterns, self.analyze_embs), f)
            xs, ys = [], []
            for embs_ls in self.analyze_embs:
                for emb in embs_ls:
                    xs.append(emb[0])
                    ys.append(emb[1])
            print("Saving analysis plot in results/analyze.png")
            plt.scatter(xs, ys, color="red", label="motif")
            plt.legend()
            plt.savefig("plots/analyze.png")
            plt.close()

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            if self.rank_method == "hybrid":
                cur_rank_method = "margin" if len(max(
                    self.counts[pattern_size].values(), key=len)) < 3 else "counts"
            else:
                cur_rank_method = self.rank_method

            if cur_rank_method == "margin":
                wl_hashes = set()
                cands = self.cand_patterns[pattern_size]
                cand_patterns_uniq_size = []
                for pattern in sorted(cands, key=lambda x: x[0]):
                    wl_hash_val = wl_hash(pattern[1],
                                          node_anchored=self.node_anchored)
                    if wl_hash_val not in wl_hashes:
                        wl_hashes.add(wl_hash_val)
                        cand_patterns_uniq_size.append(pattern[1])
                        if len(cand_patterns_uniq_size) >= self.out_batch_size:
                            cand_patterns_uniq += cand_patterns_uniq_size
                            break
            elif cur_rank_method == "counts":
                for _, neighs in list(sorted(self.counts[pattern_size].items(),
                                             key=lambda x: len(x[1]), reverse=True))[:self.out_batch_size]:
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("未识别的排名方法")
        return cand_patterns_uniq
