"""搜索代理基类。

提供子图模式搜索的基础框架。
"""

from collections import defaultdict

from core.utils.batch import batch_nx_graphs


class SearchAgent:
    """用于在嵌入空间中识别频繁子图的搜索策略类。

    该问题被建模为搜索过程。第一个动作选择一个种子节点作为生长起点。
    后续动作选择数据集中的一个节点连接到现有子图模式，
    每次将模式大小增加 1。

    详细原理和算法请参阅论文。
    """

    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
                 embs, node_anchored=False, analyze=False, model_type="order",
                 out_batch_size=20, frontier_top_k=0):
        """通过在嵌入空间中游走进行子图模式搜索。

        参数说明：
            min_pattern_size: 待识别频繁子图的最小尺寸。
            max_pattern_size: 待识别频繁子图的最大尺寸。
            model: 已训练的子图匹配模型（PyTorch nn.Module）。
            dataset: 用于挖掘频繁子图模式的 DeepSNAP 数据集。
            embs: 采样节点邻域的嵌入（参见论文）。
            node_anchored: 是否识别节点锚定的子图模式。
                节点锚定搜索过程必须使用节点锚定模型（在子图匹配 config.py 中指定）。
            analyze: 是否启用分析可视化。
            model_type: 子图匹配模型类型（须与 model 参数保持一致）。
            out_batch_size: 挖掘算法为每种尺寸输出的频繁子图数量。
                这些被预测为数据集中出现频率最高的 out_batch_size 个子图。
            frontier_top_k: 每一步保留的 frontier 候选上限。0 表示不剪枝。
        """
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size
        self.frontier_top_k = frontier_top_k
        self.cand_emb_cache = {}

    def run_search(self, n_trials=1000):
        """统一搜索驱动器。

        子类只需实现 init_search / step / finish_search，
        即可复用这套主循环。
        """
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        self.init_search()
        while not self.is_search_done():
            self.step()
        return self.finish_search()

    def init_search(self):
        raise NotImplementedError

    def is_search_done(self):
        raise NotImplementedError

    def step(self):
        """执行一步搜索的抽象方法。
        每一步向子图模式中添加一个新节点。
        run_search 至少调用 min_pattern_size 次 step 以生成至少该尺寸的模式。
        由具体搜索策略实现类继承。
        """
        raise NotImplementedError

    def _candidate_cache_key(self, graph_idx, nodes, anchor_node=None):
        """为候选子图生成稳定缓存键。"""
        return graph_idx, frozenset(nodes), anchor_node if self.node_anchored else None

    def _get_candidate_embs(self, graph_idx, graph, neigh, frontier):
        """批量获取 frontier 对应候选子图的 embedding，并缓存重复状态。"""
        cache_keys = []
        cand_graphs = []
        anchors = []
        cand_nodes = []
        cand_embs = [None] * len(frontier)
        anchor_node = neigh[0] if self.node_anchored else None

        for idx, cand_node in enumerate(frontier):
            nodes = list(neigh) + [cand_node]
            cache_key = self._candidate_cache_key(graph_idx, nodes, anchor_node)
            cache_keys.append(cache_key)
            cand_nodes.append(cand_node)
            if cache_key in self.cand_emb_cache:
                cand_embs[idx] = self.cand_emb_cache[cache_key]
            else:
                cand_graphs.append(graph.subgraph(nodes))
                if self.node_anchored:
                    anchors.append(anchor_node)

        if cand_graphs:
            import torch
            new_embs = self.model.emb_model(batch_nx_graphs(
                cand_graphs, anchors=anchors if self.node_anchored else None))
            for cand_node, cache_key, emb in zip(
                    [n for i, n in enumerate(cand_nodes) if cand_embs[i] is None],
                    [k for i, k in enumerate(cache_keys) if cand_embs[i] is None],
                    new_embs,
            ):
                emb = emb.detach().cpu()
                self.cand_emb_cache[cache_key] = emb

        for idx, cache_key in enumerate(cache_keys):
            if cand_embs[idx] is None:
                cand_embs[idx] = self.cand_emb_cache[cache_key]
        return cand_embs

    def _prune_frontier(self, graph, frontier):
        """按候选节点度数保留前 K 个 frontier 节点。"""
        if self.frontier_top_k and len(frontier) > self.frontier_top_k:
            frontier = sorted(frontier,
                              key=lambda node: (graph.degree(node), -node),
                              reverse=True)[:self.frontier_top_k]
        return frontier

    def finish_search(self):
        raise NotImplementedError
