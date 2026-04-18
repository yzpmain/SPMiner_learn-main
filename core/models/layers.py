"""自定义图卷积层实现。

提供项目特定的图神经网络层：
- SAGEConv: 自定义GraphSAGE风格卷积层
- GINConv: 带边权支持的GIN卷积层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class SAGEConv(pyg_nn.MessagePassing):
    """自定义 GraphSAGE 风格卷积层。

    与标准实现的差异主要在于：
    - 显式去除自环；
    - 通过线性层处理邻居消息；
    - 在 update 阶段把聚合结果与中心节点特征拼接再变换。
    """

    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        参数说明：
            res_n_id (Tensor, 可选): 来自 :obj:`NeighborSampler` 生成的
                :obj:`DataFlow` 的残差节点索引，用于在 :obj:`x` 中选择中心节点特征。
                在二部图操作且 :obj:`concat` 为 :obj:`True` 时必须提供。
                （默认值：:obj:`None`）
        """
        # 为避免自环对消息传递产生额外干扰，这里先移除自环边。
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        # 这里保留了边权接口，但默认仅对邻居特征做线性变换。
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GINConv(pyg_nn.MessagePassing):
    """带边权支持的 GIN 卷积实现。"""

    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        # 输入特征允许是一维张量，因此先统一成二维。
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
                                                              edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
                                                          edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
