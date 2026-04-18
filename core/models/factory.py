"""模型构建工厂。

提供统一的模型实例化、设备分配和权重加载接口。
"""

import torch
from core.config.device import get_device
from core.models.embedders import OrderEmbedder, BaselineMLP


def build_model(method_type, input_dim, hidden_dim, args,
                model_path=None, eval_mode=False):
    """根据方法类型构建模型。

    Args:
        method_type: "order" | "mlp"
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        args: 模型超参数（需包含 conv_type, n_layers, skip, dropout, margin 等）
        model_path: 若提供则加载预训练权重
        eval_mode: 是否设为评估模式

    Returns:
        加载到正确设备上的模型实例

    Raises:
        ValueError: 当 method_type 不被支持时
    """
    if method_type == "order":
        model = OrderEmbedder(input_dim, hidden_dim, args)
    elif method_type == "mlp":
        model = BaselineMLP(input_dim, hidden_dim, args)
    else:
        raise ValueError(f"Unsupported method_type: {method_type}. "
                         f"Supported types: 'order', 'mlp'")

    model.to(get_device())

    if model_path:
        model.load_state_dict(torch.load(model_path,
            map_location=get_device()))

    if eval_mode:
        model.eval()

    return model
