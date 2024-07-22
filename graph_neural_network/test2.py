import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv
from torch_geometric.data import HeteroData


# 示例输入
x_dict = {
    'paper': torch.randn(100, 16),
    'author': torch.randn(50, 16)
}

edge_index_dict = {
    ('paper', 'cites', 'paper'): torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long),
    ('author', 'writes', 'paper'): torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long),
    ('paper', 'rev_writes', 'author'): torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
}

# 边特征示例
edge_attr_dict = {
    ('paper', 'cites', 'paper'): torch.randn(3, 8),  # 假设每条边有8维特征
    ('author', 'writes', 'paper'): torch.randn(3, 8),
    ('paper', 'rev_writes', 'author'): torch.randn(3, 8)
}

data = HeteroData()

# 添加节点特征
for node_type, x in x_dict.items():
    data[node_type].x = x

# 添加边索引和边特征
for edge_type, edge_index in edge_index_dict.items():
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_attr = edge_attr_dict[edge_type]

print(data)