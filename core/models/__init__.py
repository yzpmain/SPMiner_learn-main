"""图神经网络模型模块。

提供子图匹配和挖掘所需的各类模型：
- OrderEmbedder: 序嵌入模型
- BaselineMLP: MLP基线模型
- SkipLastGNN: 带跳跃连接的GNN编码器
- SAGEConv: 自定义GraphSAGE卷积层
- GINConv: 带边权支持的GIN卷积层
"""

from core.models.embedders import OrderEmbedder, BaselineMLP
from core.models.encoders import SkipLastGNN
from core.models.layers import SAGEConv, GINConv
from core.models.factory import build_model

__all__ = [
    'OrderEmbedder',
    'BaselineMLP',
    'SkipLastGNN',
    'SAGEConv',
    'GINConv',
    'build_model',
]
