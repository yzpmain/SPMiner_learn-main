"""合成图数据生成模块。

提供多种合成图生成器（Erdos-Renyi、Watts-Strogatz、Barabasi-Albert等）。
"""

import logging
import networkx as nx
import numpy as np
import deepsnap.dataset as dataset


class ERGenerator(dataset.Generator):
    """Erdos-Renyi随机图生成器。"""

    def __init__(self, sizes, p_alpha=1.3, **kwargs):
        super(ERGenerator, self).__init__(sizes, **kwargs)
        self.p_alpha = p_alpha

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        alpha = self.p_alpha
        mean = np.log2(num_nodes) / num_nodes
        beta = alpha / mean - alpha
        p = np.random.beta(alpha, beta)
        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            p = np.random.beta(alpha, beta)
            graph = nx.gnp_random_graph(num_nodes, p)
        logging.debug('Generated {}-node E-R graphs with average p: {}'.format(
            num_nodes, mean))
        return graph


class WSGenerator(dataset.Generator):
    """Watts-Strogatz小世界图生成器。"""

    def __init__(self, sizes, density_alpha=1.3,
                 rewire_alpha=2, rewire_beta=2, **kwargs):
        super(WSGenerator, self).__init__(sizes, **kwargs)
        self.density_alpha = density_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        curr_num_graphs = 0

        density_alpha = self.density_alpha
        density_mean = np.log2(num_nodes) / num_nodes
        density_beta = density_alpha / density_mean - density_alpha

        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta
        while curr_num_graphs < 1:
            k = int(np.random.beta(density_alpha, density_beta) * num_nodes)
            k = max(k, 2)
            p = np.random.beta(rewire_alpha, rewire_beta)
            try:
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
                curr_num_graphs += 1
            except:
                pass
        logging.debug('Generated {}-node W-S graph with average density: {}'.format(
            num_nodes, density_mean))
        return graph


class BAGenerator(dataset.Generator):
    """Barabasi-Albert优先连接图生成器。"""

    def __init__(self, sizes, max_p=0.2, max_q=0.2, **kwargs):
        super(BAGenerator, self).__init__(sizes, **kwargs)
        self.max_p = 0.2
        self.max_q = 0.2

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        found = False
        m = np.random.choice(max_m) + 1
        p = np.min([np.random.exponential(20), self.max_p])
        q = np.min([np.random.exponential(20), self.max_q])
        while not found:
            graph = nx.extended_barabasi_albert_graph(num_nodes, m, p, q)
            if nx.is_connected(graph):
                found = True
        logging.debug('Generated {}-node extended B-A graph with max m: {}'.format(
            num_nodes, max_m))
        return graph


class PowerLawClusterGenerator(dataset.Generator):
    """幂律聚类图生成器。"""

    def __init__(self, sizes, max_triangle_prob=0.5, **kwargs):
        super(PowerLawClusterGenerator, self).__init__(sizes, **kwargs)
        self.max_triangle_prob = max_triangle_prob

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        m = np.random.choice(max_m) + 1
        p = np.random.uniform(high=self.max_triangle_prob)
        found = False
        while not found:
            graph = nx.powerlaw_cluster_graph(num_nodes, m, p)
            if nx.is_connected(graph):
                found = True
        logging.debug('Generated {}-node powerlaw cluster graph with max m: {}'.format(
            num_nodes, max_m))
        return graph


def get_generator(sizes, size_prob=None, dataset_len=None):
    """获取组合图生成器。

    Args:
        sizes: 图大小范围
        size_prob: 大小概率分布
        dataset_len: 数据集长度

    Returns:
        EnsembleGenerator实例
    """
    generator = dataset.EnsembleGenerator(
        [ERGenerator(sizes, size_prob=size_prob),
         WSGenerator(sizes, size_prob=size_prob),
         BAGenerator(sizes, size_prob=size_prob),
         PowerLawClusterGenerator(sizes, size_prob=size_prob)],
        dataset_len=dataset_len)
    return generator


def get_dataset(task, dataset_len, sizes, size_prob=None, **kwargs):
    """获取合成图数据集。

    Args:
        task: 任务类型
        dataset_len: 数据集长度
        sizes: 图大小范围
        size_prob: 大小概率分布
        **kwargs: 额外参数

    Returns:
        GraphDataset实例
    """
    generator = get_generator(sizes, size_prob=size_prob,
                              dataset_len=dataset_len)
    return dataset.GraphDataset(
        None, task=task, generator=generator, **kwargs)
