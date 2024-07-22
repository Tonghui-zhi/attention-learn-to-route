#!/usr/bin/env python
# coding=utf-8

"""
@author: zth
@contact: zhitonghui1@jd.com
@file: gnn_model_test.py
@date: 2024/6/26 15:25
@desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv
from gnn_data import data

# 参数设置
in_channels = 16
out_channels = 8
hidden_channels = 64
heads = 4
num_nodes1 = 10
num_nodes2 = 10
edge_index = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)  # 示例边索引

# 假设我们有两种节点类型和一种边类型
data = HeteroData()
data['node_type1'].x = torch.randn((num_nodes1, in_channels))
data['node_type2'].x = torch.randn((num_nodes2, in_channels))
data['node_type1', 'edge_type', 'node_type2'].edge_index = edge_index

# # 定义一个异质图的多头图注意力机制模型
# class HeteroGAT(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, heads):
#         super(HeteroGAT, self).__init__()
#         self.conv1 = HeteroConv({
#             ('node_type1', 'edge_type', 'node_type2'): GATConv(in_channels, out_channels, heads=heads, add_self_loops=False),
#             ('node_type2', 'edge_type', 'node_type1'): GATConv(in_channels, out_channels, heads=heads, add_self_loops=False)
#         }, aggr='mean')
#
#     def forward(self, data):
#         x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict = {key: x.relu() for key, x in x_dict.items()}
#
#         return x_dict
#
#
# # 创建模型并执行前向传播
# model = HeteroGAT(in_channels, out_channels, heads)
# out = model(data)
#
# print(out)

from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(HeteroGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # 对不同类型的边使用不同的GCN
            conv = HeteroConv({
            ('node_type1', 'edge_type', 'node_type2'): GATConv(-1, hidden_channels, heads=heads, add_self_loops=False),
            ('node_type2', 'edge_type', 'node_type1'): GATConv(-1, hidden_channels, heads=heads, add_self_loops=False)
            }, aggr='sum')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for conv in self.convs:
          x_dict = conv(x_dict, edge_index_dict)
          x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['node_type1'])

model = HeteroGNN(in_channels, hidden_channels, out_channels,
                  num_layers=2)
out = model(data)
print(out)