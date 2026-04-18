"""特征预处理模块。

提供图特征预处理和变换功能。
"""

import torch
import torch.nn as nn

from core.features.augment import FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS, AUGMENT_METHOD


class Preprocess(nn.Module):
    """特征预处理器，用于在输入GNN前处理节点特征。"""

    def __init__(self, dim_in):
        super(Preprocess, self).__init__()
        self.dim_in = dim_in
        if AUGMENT_METHOD == 'add':
            self.module_dict = {
                key: nn.Linear(aug_dim, dim_in)
                for key, aug_dim in zip(FEATURE_AUGMENT,
                                        FEATURE_AUGMENT_DIMS)
            }

    @property
    def dim_out(self):
        """计算输出维度。"""
        if AUGMENT_METHOD == 'concat':
            return self.dim_in + sum(
                [aug_dim for aug_dim in FEATURE_AUGMENT_DIMS])
        elif AUGMENT_METHOD == 'add':
            return self.dim_in
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                AUGMENT_METHOD))

    def forward(self, batch):
        """前向处理批次数据。"""
        if AUGMENT_METHOD == 'concat':
            feature_list = [batch.node_feature]
            for key in FEATURE_AUGMENT:
                feature_list.append(batch[key])
            batch.node_feature = torch.cat(feature_list, dim=-1)
        elif AUGMENT_METHOD == 'add':
            for key in FEATURE_AUGMENT:
                batch.node_feature = batch.node_feature + self.module_dict[key](
                    batch[key])
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                AUGMENT_METHOD))
        return batch
