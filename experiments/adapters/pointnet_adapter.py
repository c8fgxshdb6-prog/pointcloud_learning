"""
PointNet++ 特征与自适应模块之间的格式转换函数。
PointNet++ 输出特征形状: (B, C, N)  (批大小, 通道数, 点数)
自适应模块输入形状:    (B, N, C)  (批大小, 点数, 通道数)
"""

import torch

def pointnet_feature_to_adaptive(feature):
    """
    将 PointNet++ 特征从 (B, C, N) 转换为 (B, N, C)
    """
    return feature.transpose(1, 2)

def adaptive_feature_to_pointnet(feature):
    """
    将自适应模块输出的特征从 (B, N, C) 转回 (B, C, N)
    """
    return feature.transpose(1, 2)