"""统一数据集加载模块。

提供标准化的数据集加载功能，支持子图匹配训练和子图挖掘两种使用场景。
"""

from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from core.utils.io import load_snap_edgelist
from core.data.synthetic import get_generator


def load_dataset(name):
    """加载 PyTorch Geometric 中提供的真实世界数据集。

    用于 DiskDataSource 的辅助函数，返回 (train, test, task) 三元组。
    支持的数据集包括：
    - TUDataset: enzymes, proteins, cox2, aids, reddit-binary, imdb-binary, firstmm_db, dblp
    - PPI: ppi
    - QM9: qm9
    - NetworkX: atlas
    - SNAP: facebook, as-733, as20000102

    Args:
        name: 数据集名称

    Returns:
        tuple: (train_list, test_list, task_string)
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name='ENZYMES')
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name='PROTEINS')
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name='COX2')
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name='AIDS')
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name='REDDIT-BINARY')
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name='IMDB-BINARY')
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name='FIRSTMM_DB')
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v1", name='DBLP_v1')
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    elif name == "facebook":
        graph = load_snap_edgelist("data/facebook_combined.txt")
        return [graph], [graph], "graph"
    elif name in ("as-733", "as20000102"):
        graph = load_snap_edgelist("data/as20000102.txt")
        return [graph], [graph], "graph"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name:
                    del graph.name
                graph = pyg_utils.to_networkx(graph).to_undirected()
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task


def make_plant_dataset(size):
    """构造带植入模式的合成图数据集。

    用于验证挖掘算法是否能从噪声图中恢复出高频结构模式：
    - 先生成一个固定模式 pattern；
    - 再把 pattern 并到随机图中并随机连边；
    - 返回由 1000 张图组成的数据集。

    Args:
        size: 合成图的大小参数

    Returns:
        list: 包含 1000 个 networkx.Graph 的列表
    """
    import matplotlib.pyplot as plt
    
    generator = get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs


def load_mining_dataset(name):
    """加载用于子图挖掘的数据集。

    支持子图挖掘场景的所有数据集，包括 TUDataset、PPI、QM9、
    自定义边列表格式（roadnet-*、diseasome、usroads、mn-roads、infect）、
    植入模式数据集（plant-*）以及 SNAP 数据集（facebook、as-733、as20000102）。

    Args:
        name: 数据集名称

    Returns:
        tuple: (dataset_list, task_string)
    """
    if name == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif name == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
        task = 'graph'
    elif name == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
        task = 'graph'
    elif name == 'dblp':
        dataset = TUDataset(root='/tmp/dblp', name='DBLP_v1')
        task = 'graph-truncate'
    elif name == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
        task = 'graph'
    elif name.startswith('roadnet-'):
        graph = nx.Graph()
        with open("data/{}.txt".format(name), "r") as f:
            for row in f:
                if not row.startswith("#"):
                    a, b = row.split("\t")
                    graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
        task = 'graph'
    elif name in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {
            "diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"
        }
        graph = nx.Graph()
        with open("data/{}".format(fn[name]), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif name.startswith('plant-'):
        size = int(name.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'
    elif name == 'facebook':
        graph = load_snap_edgelist("data/facebook_combined.txt")
        dataset = [graph]
        task = 'graph'
    elif name in ['as-733', 'as20000102']:
        graph = load_snap_edgelist("data/as20000102.txt")
        dataset = [graph]
        task = 'graph'
    else:
        raise ValueError(f"Unknown dataset for mining: {name}")

    return dataset, task
