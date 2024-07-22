import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 假设我们有4个节点，每个节点有3维特征
node_features = torch.tensor([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [10, 11, 12]], dtype=torch.float)

# 边列表，假设是全连接图
edge_index = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                           [1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0]], dtype=torch.long)

# 定义GAT层
gat_conv = GATConv(in_channels=3, out_channels=2, heads=4, concat=False)

# 进行前向传播
node_embeddings = gat_conv(node_features, edge_index)

print("Node Embeddings after GATConv:")
print(node_embeddings)
