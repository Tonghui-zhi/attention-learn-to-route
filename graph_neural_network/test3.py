#!/usr/bin/env python
# coding=utf-8

"""
@author: zth
@contact: zhitonghui1@jd.com
@file: test3.py
@date: 2024/6/26 22:20
@desc: 
"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv
from torch_geometric.data import HeteroData
from data_process import kefu_data, test_hetero_Dataset


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, tar, heads):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # ('paper', 'cites', 'paper'): EdgeGCNConv(-1, hidden_channels),
                # ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
                edge_type: GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels * heads, out_channels)
        self.tar = tar

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict[self.tar])

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask = data['paper'].train_mask
    loss = F.cross_entropy(out[mask], data['paper'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(data).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['paper'][split]
        acc = (pred[mask] == data['paper'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

def train_batch(batch):
    model.train()
    optimizer.zero_grad()
    out = model(batch)
    mask = batch[model.tar].train_mask
    loss = F.cross_entropy(out[mask], batch[model.tar].y[mask])

    # for name, parms in model.named_parameters():
    #     print('-->name:', name)
    #     # print('-->para:', parms)
    #     print('-->grad_requirs:', parms.requires_grad)
    #     print('-->grad_value:', parms.grad.norm(2) if parms.grad is not None else parms.grad)
    #     print("===更新前")

    loss.backward()
    optimizer.step()

    # print("=============更新之后===========")
    # for name, parms in model.named_parameters():
    #     print('-->name:', name)
    #     # print('-->para:', parms)
    #     print('-->grad_requirs:', parms.requires_grad)
    #     print('-->grad_value:', parms.grad.norm(2) if parms.grad is not None else parms.grad)
    #     print("===更新后")

    return float(loss), out

@torch.no_grad
def test_batch(batch):
    model.eval()
    pred = model(batch).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = batch[model.tar][split]
        acc = (pred[mask] == batch[model.tar].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


if __name__ == '__main__':
    data = pd.read_pickle('../data/kf/kf20_kf_seed1234.pkl')
    dataset, dataloader = kefu_data(data, batch_size=100)
    # dataset, dataloader = test_hetero_Dataset(batch_size=1)

    model = HeteroGNN(dataset[0].metadata(), hidden_channels=128, out_channels=10, num_layers=2, tar='item', heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    for epoch in range(100):
        for batch in dataloader:
            loss, out = train_batch(batch)
            train_acc, val_acc, test_acc = test_batch(batch)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

