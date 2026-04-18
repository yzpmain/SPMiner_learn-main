"""Common模块 - 兼容性层。

该模块现在作为core模块的兼容性包装，所有功能已从core重新导出。
建议新代码直接从core导入：
    from core import models, data, utils
    from core.config import get_device

此模块保留以维持向后兼容。
"""

import warnings

warnings.warn(
    "common模块已重构为core模块。"
    "请使用 'from core import ...' 替代 'from common import ...'。"
    "此兼容性层将在未来版本中移除。",
    DeprecationWarning,
    stacklevel=2
)

# 从core重新导出所有符号
from core import *
from core.models import *
from core.data import *
from core.utils import *
from core.features import *
from core.config import *
