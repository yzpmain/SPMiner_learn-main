import argparse
import time
import os

import numpy as np
import torch
from tqdm import tqdm

import torch_geometric.utils as pyg_utils

from core import utils
from core.config import get_device
from core.data.datasets import load_mining_dataset
from core.models import build_model
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search import GreedySearchAgent, MCTSSearchAgent

import matplotlib.pyplot as plt

import random
from collections import defaultdict
import networkx as nx
import pickle

def pattern_growth(dataset, task, args):
    """SPMiner 主流程：采样 -> 嵌入 -> 搜索 -> 输出。

    该函数是挖掘入口中的核心逻辑：
    1. 加载匹配模型（作为频繁性评分器）；
    2. 构建候选邻域集合；
    3. 批量编码邻域嵌入；
    4. 调用 Greedy/MCTS 搜索代理；
    5. 保存可视化与序列化结果。
    """
    # 初始化模型
    model = build_model(args.method_type, 1, args.hidden_dim, args,
                        model_path=args.model_path, eval_mode=True)

    if task == "graph-labeled":
        dataset, labels = dataset

    # 将不同来源数据统一为 networkx.Graph 列表，
    # 便于后续采样和搜索器逻辑复用。
    neighs_pyg, neighs = [], []
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    if task == "graph-labeled": print("using label 0")
    graphs = []
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
        graphs.append(graph)
    if args.use_whole_graphs:
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                print(i)
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0: print(i, j)
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        neigh = list(nx.single_source_shortest_path_length(graph,
                            node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    if len(neigh) > 1:
                        neigh = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            neigh = neigh.subgraph(max(
                                nx.connected_components(neigh), key=len))
                        neigh = nx.convert_node_labels_to_integers(neigh)
                        neigh.add_edge(0, 0)
                        neighs.append(neigh)
        elif args.sample_method == "tree":
            # tree 采样：每次从数据集中随机抽图，再扩展一个连通邻域。
            start_time = time.time()
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                neigh.add_edge(0, 0)
                neighs.append(neigh)
                if args.node_anchored:
                    anchors.append(0)   # after converting labels, 0 will be anchor

    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        #top = min(len(neighs), (i+1)*args.batch_size)
        top = (i+1)*args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                anchors=anchors if args.node_anchored else None)
            emb = model.emb_model(batch)
            emb = emb.to(torch.device("cpu"))

        embs.append(emb)

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    # 搜索阶段：把候选邻域嵌入交给策略代理，输出频繁模式。
    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, out_batch_size=args.out_batch_size,
            frontier_top_k=args.frontier_top_k)
    elif args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size,
            frontier_top_k=args.frontier_top_k)
    out_graphs = agent.run_search(args.n_trials)
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    # 可视化输出模式：每种大小按出现顺序保存图像。
    count_by_size = defaultdict(int)
    for pattern in out_graphs:
        if args.node_anchored:
            colors = ["red"] + ["blue"]*(len(pattern)-1)
            nx.draw(pattern, node_color=colors, with_labels=True)
        else:
            nx.draw(pattern)
        print("Saving plots/cluster/{}-{}.png".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.savefig("plots/cluster/{}-{}.png".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.savefig("plots/cluster/{}-{}.pdf".format(len(pattern),
            count_by_size[len(pattern)]))
        plt.close()
        count_by_size[len(pattern)] += 1

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

def main():
    """解码器 CLI 入口。

    负责：
    - 组合 encoder/decoder 参数；
    - 读取数据集；
    - 调用 pattern_growth 执行完整挖掘。
    """
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='解码器参数')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    print("Using dataset {}".format(args.dataset))
    dataset, task = load_mining_dataset(args.dataset)

    pattern_growth(dataset, task, args) 

if __name__ == '__main__':
    main()

