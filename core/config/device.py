"""设备管理模块。

提供CUDA/CPU设备的统一管理。
"""

import torch

device_cache = None


def get_device():
    """懒加载运行设备（优先 CUDA）。
    
    Returns:
        torch.device: 计算设备
    """
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    return device_cache
