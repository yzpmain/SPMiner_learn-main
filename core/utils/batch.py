"""批处理工具模块。

提供图批处理功能。
"""

import torch
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph

from core.config.device import get_device
from core.features.augment import FeatureAugment


def batch_nx_graphs(graphs, anchors=None):
    """将NetworkX图列表批处理为DeepSNAP Batch。

    Args:
        graphs: NetworkX图列表
        anchors: 锚点节点列表（可选）

    Returns:
        DeepSNAP Batch对象
    """
    augmenter = FeatureAugment()

    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    batch = batch.to(get_device())
    return batch
