"""配置管理模块。

提供设备管理、优化器配置等基础配置功能。
"""

from core.config.device import get_device
from core.config.optimizer import parse_optimizer, build_optimizer

__all__ = ['get_device', 'parse_optimizer', 'build_optimizer']
