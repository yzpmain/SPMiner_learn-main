"""SPMiner核心模块。

该模块包含项目的基础组件，包括：
- config: 配置管理
- models: 图神经网络模型定义
- data: 数据处理和加载
- features: 特征增强和预处理
- utils: 通用工具函数

使用示例:
    from core import models, data, utils
    from core.config import get_device, build_optimizer
"""

__version__ = "1.0.0"

# 从各个子模块导出常用功能
from core.config.device import get_device
from core.config.optimizer import parse_optimizer, build_optimizer
from core.models.embedders import OrderEmbedder, BaselineMLP
from core.models.encoders import SkipLastGNN
from core.models.layers import SAGEConv, GINConv
from core.models.factory import build_model
from core.data.sources import (
    DataSource,
    OTFSynDataSource,
    OTFSynImbalancedDataSource,
    DiskDataSource,
    DiskImbalancedDataSource,
    load_dataset,
    make_data_source
)
from core.data.datasets import load_mining_dataset, make_plant_dataset
from core.data.synthetic import get_generator, get_dataset
from core.utils.graph import sample_neigh, wl_hash, enumerate_subgraph
from core.utils.batch import batch_nx_graphs
from core.utils.io import load_snap_edgelist

__all__ = [
    # Config
    'get_device',
    'parse_optimizer',
    'build_optimizer',
    # Models
    'OrderEmbedder',
    'BaselineMLP',
    'SkipLastGNN',
    'SAGEConv',
    'GINConv',
    'build_model',
    # Data
    'DataSource',
    'OTFSynDataSource',
    'OTFSynImbalancedDataSource',
    'DiskDataSource',
    'DiskImbalancedDataSource',
    'load_dataset',
    'make_data_source',
    'load_mining_dataset',
    'make_plant_dataset',
    'get_generator',
    'get_dataset',
    # Utils
    'sample_neigh',
    'wl_hash',
    'enumerate_subgraph',
    'batch_nx_graphs',
    'load_snap_edgelist',
]
