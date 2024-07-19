#!/usr/bin/env python
# coding=utf-8

"""
@author: zth
@contact: zhitonghui1@jd.com
@file: gnn_data.py
@date: 2024/6/26 17:43
@desc: 
"""
import torch
from torch_geometric.data import HeteroData

# 创建异质图数据集
data = HeteroData()

# 添加节点特征
data['user'].x = torch.randn(4, 10)  # 4个用户，每个用户10维特征
data['item'].x = torch.randn(3, 10)  # 3个物品，每个物品10维特征

# 添加边索引和边特征
data['user', 'interacts', 'item'].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
data['user', 'interacts', 'item'].edge_attr = torch.randn(4, 5)  # 4条边，每条边5维特征

print(data)
