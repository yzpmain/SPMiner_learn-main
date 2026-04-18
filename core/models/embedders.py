"""图嵌入模型实现。

提供序嵌入模型和基线MLP模型，用于子图匹配任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.encoders import SkipLastGNN
from core.config.device import get_device


class BaselineMLP(nn.Module):
    """最简单的双图拼接分类基线。

    输入是两个图的向量表示，直接拼接后送入 MLP 做二分类。
    该模型常用于对比 order embedding 的效果。
    """

    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)


class OrderEmbedder(nn.Module):
    """序嵌入模型。

    通过约束"子图嵌入应小于或等于超图嵌入"的方式，
    学习一个能够表达子图包含关系的空间。
    """

    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        # forward 不直接做分类，只返回嵌入对供 predict / criterion 使用。
        return emb_as, emb_bs

    def predict(self, pred):
        """批量预测 b 是否是 a 的子图，其中 emb_as, emb_bs = pred。

        Args:
            pred: 图对嵌入列表 (emb_as, emb_bs)

        Returns:
            违反量 e（越小表示越可能是子图关系）
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                 device=emb_as.device), emb_bs - emb_as) ** 2, dim=1)
        return e

    def criterion(self, pred, intersect_embs, labels):
        """序嵌入的损失函数。

        e 项表示违反量（当 b 是 a 的子图时）。
        对于正例，e 项被最小化（接近 0）；
        对于负例，e 项被训练为至少大于 self.margin。

        Args:
            pred: forward 输出的嵌入列表
            intersect_embs: 未使用
            labels: pred 中每对图的子图标签
        """
        emb_as, emb_bs = pred
        # e 表示违反序关系的程度：
        # 若 emb_bs <= emb_as 则该项越接近 0 越好。
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                 device=get_device()), emb_bs - emb_as) ** 2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
                                                device=get_device()), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss
