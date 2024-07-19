#!/usr/bin/env python
# coding=utf-8

"""
@author: zth
@contact: zhitonghui1@jd.com
@file: problem_kf.py
@date: 2024/6/17 13:14
@desc: 考虑车辆容量不同的cvrp
"""
import os.path
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset
from problems.kf.state_kf import StateKF


class KF(object):

    NAME = 'kf'
    VEHICLE_NUM = 10
    VEHICLE_CAPACITY = 1

    @staticmethod
    def get_costs(dataset, pi):
        """

        Args:
            dataset: demand, dim = batch_size, (vehicle_size, item_size)
            pi:  path, dim = batch_size, item_size

        Returns: cost per batch

        """

        batch_size, demand_num, vehicle_num = dataset['deal_time'].size()
        # 计算每条路径，每个车的实际承载量
        routes_dict = {}
        over_loaded_cost = {}
        finish_time_cost = {}
        final_cost = {}
        overall_deal_time = {}
        for idx in range(batch_size):
            routes_dict[idx] = [r[r != 0] for r in np.split(pi[idx], np.where(pi[idx] == 0)[0]) if (r != 0).any()]
            if len(routes_dict[idx]) == 0:
                print('找到的线路为空！！！！！！！！！')
                over_loaded_cost[idx] = torch.tensor(100, dtype=torch.float64)
                finish_time_cost[idx] = torch.tensor(100, dtype=torch.float64)
                final_cost[idx] = torch.tensor(100, dtype=torch.float64)
            else:
                # 以最晚完成时间为loss，不考虑员工负荷
                # over_loaded_cost[idx] = torch.sum(torch.stack([torch.clamp(torch.tensor(len(route) * (1 / dataset['deal_time'].size(1)) - dataset['vehicle_capacity'][idx, vehicle_id]), min=0)
                #        for vehicle_id, route in enumerate(routes_dict[idx])]))
                deal_time = torch.cat((torch.zeros(batch_size, 1, vehicle_num), dataset['deal_time']), dim=1)
                overall_deal_time[idx] = torch.sum(torch.stack([deal_time[idx, route, vehicle_id].sum() for vehicle_id, route in enumerate(routes_dict[idx])]))
                finish_time_cost[idx] = torch.max(torch.stack([deal_time[idx, route, vehicle_id].sum() for vehicle_id, route in enumerate(routes_dict[idx])]))
                final_cost[idx] = (overall_deal_time[idx] + 3 * finish_time_cost[idx])
        # 选取最长处理时长作为每批的loss, cost.shape = [batch_size, 1]
        cost = torch.stack(list(final_cost.values()), dim=0)
        # 选取总处理时长作为loss
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return KFDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateKF.initialize(*args, **kwargs)


def make_instance(args):
    deal_time, vehicle_capacity, *args = args
    grid_size = 1
    return {
        'deal_time': torch.tensor(deal_time, dtype=torch.float) / grid_size,
        'vehicle_capacity': torch.tensor(vehicle_capacity, dtype=torch.float)
    }


class KFDataset(Dataset):

    def __init__(self, filename=None, size=20, num_samples=50, offset=0, distribution=None):

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            CAPACITIES = np.random.randint(low=8, high=12, size=KF.VEHICLE_NUM)

            self.data = [
                {
                    'deal_time': torch.FloatTensor(size, KF.VEHICLE_NUM).uniform_(0, 1),
                    'vehicle_capacity': torch.FloatTensor(KF.VEHICLE_NUM).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    problem = KF()
    val_dataset = problem.make_dataset(
        size=50, num_samples=100, filename=None)
    print(val_dataset)



