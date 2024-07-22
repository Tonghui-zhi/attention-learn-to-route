#!/usr/bin/env python
# coding=utf-8

"""
@author: zth
@contact: zhitonghui1@jd.com
@file: data_process.py
@date: 2024/6/27 11:16
@desc: 
"""
import pandas as pd
import torch
from torch_geometric.data import Dataset, DataLoader, HeteroData
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, MovieLens100K

class MyHeteroDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def test_hetero_Dataset(batch_size):
    # 创建多个HeteroData对象
    num_graphs = 10
    data_list = []

    for _ in range(num_graphs):
        data = HeteroData()
        data['paper'].x = torch.randn(20, 16)
        data['author'].x = torch.randn(20, 16)
        data[('paper', 'cites', 'paper')].edge_index = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
        data[('author', 'writes', 'paper')].edge_index = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
        data[('paper', 'rev_writes', 'author')].edge_index = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)

        # 边特征示例
        edge_attr_dict = {
            ('paper', 'cites', 'paper'): torch.randn(3, 8),  # 假设每条边有8维特征
            ('author', 'writes', 'paper'): torch.randn(3, 8),
            ('paper', 'rev_writes', 'author'): torch.randn(3, 8)
        }
        # 添加边索引和边特征
        for edge_type, edge_index in data.edge_index_dict.items():
            # data[edge_type].edge_index = edge_index
            data[edge_type].edge_attr = edge_attr_dict[edge_type]

        data['paper'].year = torch.randint(1970, 2021, (20,))
        data['paper'].y = torch.randint(0, 10, (20,))
        data['paper'].train_mask = torch.rand(20) < 0.8
        data['paper'].val_mask = (torch.rand(20) < 0.1) & ~data['paper'].train_mask
        data['paper'].test_mask = ~data['paper'].train_mask & ~data['paper'].val_mask
        data_list.append(data)

    # 创建自定义数据集
    dataset = MyHeteroDataset(data_list)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # 使用DataLoader
    # for batch in dataloader:
    #     # 处理batch
    #     print(batch)
    return dataset, dataloader


def kefu_data(instance_list, batch_size):
    num_graphs = len(instance_list)
    data_list = []
    for i in range(num_graphs):
        data_list.append(constract_graph_element(instance_list[i]))
    dataset = MyHeteroDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader


def constract_graph_element(instance):
    """
    根据matrix造图
    """
    matrix = instance[0]
    capacity = instance[1]

    num_item = len(matrix)
    num_worker = len(matrix[0])

    data = HeteroData()
    data['worker'].x = torch.tensor(capacity).view(-1, 1)
    data['item'].x = torch.randn(len(matrix), 2)

    item_list, worker_list, time_list = zip(*[
        (i, j, matrix[i][j])
        for i in range(num_item)
        for j in range(num_worker)
        if matrix[i][j] != 0
    ])

    assert len(item_list) == len(time_list), "边索引数量和边特征的数量必须相同！"

    data[('worker', 'deal', 'item')].edge_index = torch.tensor([worker_list, item_list], dtype=torch.long)
    data[('item', 'to', 'worker')].edge_index = torch.tensor([item_list, worker_list], dtype=torch.long)

    edge_attr_dict = {
        ('worker', 'deal', 'item'): torch.tensor(time_list).view(-1, 1),
        ('item', 'to', 'worker'): torch.tensor(time_list).view(-1, 1)
    }

    # 添加边特征
    for edge_type, edge_index in data.edge_index_dict.items():
        data[edge_type].edge_attr = edge_attr_dict[edge_type]

    data['item'].y = torch.randint(0, 10, (num_item,))
    data['item'].train_mask = torch.randn(num_item) < 0.8
    data['item'].val_mask = (torch.rand(num_item) < 0.1) & ~data['item'].train_mask  # 10% 验证数据
    data['item'].test_mask = ~data['item'].train_mask & ~data['item'].val_mask  # 10% 测试数据

    return data

def kefu_data_(instance_list, batch_size):
    num_graphs = instance_list.shape[0]
    data_list = []
    for i in range(num_graphs):
        data_list.append(construct_graph_element_(instance_list[i]))
    dataset = MyHeteroDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

def construct_graph_element_(instance):
    """
    根据matrix造图
    """
    matrix_batch = instance['deal_time']
    capacity_batch = instance['vehicle_capacity']

    num_graph = matrix_batch.shape[0]
    num_item = matrix_batch.shape[1]
    num_worker = matrix_batch.shape[2]
    data_list = []

    for id in range(num_graph):
        data = HeteroData()
        matrix = matrix_batch[id, :, :]
        capacity = capacity_batch[id, :]

        data['worker'].x = capacity.view(-1, 1)
        data['item'].x = torch.randn(matrix.shape[0], 2)

        item_list, worker_list, time_list = zip(*[
            (i, j, matrix[i][j])
            for i in range(num_item)
            for j in range(num_worker)
            if matrix[i][j] != 0
        ])

        assert len(item_list) == len(time_list), "边索引数量和边特征的数量必须相同！"

        data[('worker', 'deal', 'item')].edge_index = torch.tensor([worker_list, item_list], dtype=torch.long)
        data[('item', 'to', 'worker')].edge_index = torch.tensor([item_list, worker_list], dtype=torch.long)

        edge_attr_dict = {
            ('worker', 'deal', 'item'): torch.tensor(time_list).view(-1, 1),
            ('item', 'to', 'worker'): torch.tensor(time_list).view(-1, 1)
        }

        # 添加边特征
        for edge_type, edge_index in data.edge_index_dict.items():
            data[edge_type].edge_attr = edge_attr_dict[edge_type]

        data_list.append(data)

    dataset = MyHeteroDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=num_graph, shuffle=True)

    return dataset, dataloader

def construct_graph_element_2(instance):
    """
    根据matrix造图
    """
    matrix_batch = instance['deal_time']
    capacity_batch = instance['vehicle_capacity']

    num_graph = matrix_batch.shape[0]
    num_item = matrix_batch.shape[1]
    num_worker = matrix_batch.shape[2]
    data_list = []

    for id in range(num_graph):
        data = HeteroData()
        matrix = matrix_batch[id, :, :]
        capacity = capacity_batch[id, :]

        data['worker'].x = capacity.view(-1, 1)
        data['item'].x = torch.randn(matrix.shape[0], 2)

        item_list, worker_list, time_list = zip(*[
            (i, j, matrix[i][j])
            for i in range(num_item)
            for j in range(num_worker)
            if matrix[i][j] != 0
        ])

        assert len(item_list) == len(time_list), "边索引数量和边特征的数量必须相同！"

        data[('worker', 'deal', 'item')].edge_index = torch.tensor([worker_list, item_list], dtype=torch.long)
        data[('item', 'to', 'worker')].edge_index = torch.tensor([item_list, worker_list], dtype=torch.long)

        edge_attr_dict = {
            ('worker', 'deal', 'item'): torch.tensor(time_list).view(-1, 1),
            ('item', 'to', 'worker'): torch.tensor(time_list).view(-1, 1)
        }

        # 添加边特征
        for edge_type, edge_index in data.edge_index_dict.items():
            data[edge_type].edge_attr = edge_attr_dict[edge_type]

        data_list.append(data)

    from torch_geometric.data import Batch
    batch = Batch.from_data_list(data_list)

    return batch, len(data_list)

def make_instance(args):
    deal_time, vehicle_capacity, *args = args
    grid_size = 1
    return {
        'deal_time': torch.tensor(deal_time, dtype=torch.float) / grid_size,
        'vehicle_capacity': torch.tensor(vehicle_capacity, dtype=torch.float)
    }


if __name__ == "__main__":
    data = pd.read_pickle('../data/kf/kf20_kf_seed1234.pkl')[0: 300]
    num_sample = 50
    data_dict = [make_instance(args) for args in data[0: num_sample]]
    deal_time_tensor = torch.concat([x['deal_time'].view(1, 20 ,5) for x in data_dict], dim=0)
    capacity_tensor = torch.concat([x['vehicle_capacity'].view(1, -1) for x in data_dict], dim=0)
    data = {'deal_time': deal_time_tensor, 'vehicle_capacity': capacity_tensor}
    dataset, dataloader = construct_graph_element_(data)
    print(dataset)

    # for batch in dataloader:
    #     model = HeteroGNN(batch.metadata(), hidden_channels=128, out_channels=10, num_layers=2, tar='item', heads=4)
    #     out = model(batch)
    #     print(out.shape)

