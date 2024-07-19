#!/usr/bin/env python
# coding=utf-8

"""
@author: zth
@contact: zhitonghui1@jd.com
@file: state_kf.py
@date: 2024/6/17 13:14
@desc: 
"""
import torch
import numpy as np
from typing import NamedTuple

from utils.boolmask import mask_long2bool, mask_long_scatter


class StateKF(NamedTuple):
    # 固定输入

    ids: torch.Tensor
    deal_time: torch.Tensor

    # 状态
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # 追踪已经游览过的节点
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor # 追踪步数

    vehicle_capacity: torch.Tensor
    vehicle_id: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.deal_time.size(0))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        deal_time = input['deal_time']
        vehicle_capacity = input['vehicle_capacity']

        batch_size, n_loc, _ = deal_time.size()

        return StateKF(
            deal_time=deal_time,
            ids=torch.arange(batch_size, dtype=torch.int64, device=deal_time.device)[:, None], # 新增步数维度
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=deal_time.device),
            used_capacity=deal_time.new_zeros(batch_size, 1),
            visited_=(
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=deal_time.device
                )
            ),
            lengths=torch.zeros(batch_size, 1, device=deal_time.device),
            cur_coord=torch.tensor(0), # 默认初始点是-1
            i=torch.zeros(1, dtype=torch.int64, device=deal_time.device),
            vehicle_id=torch.zeros(vehicle_capacity.size(0)),
            vehicle_capacity=vehicle_capacity
        )

    def get_final_cost(self):

        assert self.all_finished()

        return max(self.lengths)

    def update(self, selected):
        vehicle_num = self.vehicle_capacity.size(-1)

        # 若选到depot，说明该车结束任务，换下辆车
        selected_is_depot = (selected == 0)  # (batch_size)
        vehicle_id = self.vehicle_id + selected_is_depot.int()

        # 更新状态
        selected = selected[:, None]
        prev_a = selected * (vehicle_id < vehicle_num)[:, None].type(torch.int64)
        batch_size = self.deal_time.size(0) # 批次数
        n_loc = self.deal_time.size(1)

        # 当前所在的需求点
        cur_coord = selected

        # 计算每辆车的当前负载
        deal_time = torch.cat((torch.zeros(batch_size, 1, vehicle_num), self.deal_time), dim=1)
        # print(self.deal_time.size(), selected.size(), vehicle_id.size(), self.lengths.size(), self.deal_time.size())
        update_mask = (self.vehicle_id < vehicle_num)[:, None].float()
        lengths_update = deal_time[
            torch.arange(batch_size).unsqueeze(1),
            selected.type(torch.int64),
            self.vehicle_id.unsqueeze(1).type(torch.int64)
        ] * update_mask

        lengths = self.lengths + lengths_update

        # 增加每辆车的负载，如果是depot则把车辆负载清零
        used_capacity = (self.used_capacity + 1/n_loc) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, vehicle_id=vehicle_id
        )

    def all_finished(self):
        return self.i >= self.deal_time.size(1) and self.visited.all()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """

        Returns: 得到一个 (batch_size, n_loc + 1) 的mask根据可行的actions (0 = depot)，
        取决于已经处理的点和剩余的车辆容量，0 = feasible, 1 = infeasible
        禁止连续两次处理depot，除非所有点都已经被处理
        如果车辆数已经用完，则只能选择虚拟点0
        """
        vehicle_num = self.deal_time.size(-1)
        batch_size = self.deal_time.size(0)
        n_loc = self.deal_time.size(1)

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.deal_time.size(1))
        ones_demands = torch.ones_like(visited_loc)/visited_loc.size(-1)

        # 对于需求，通过索引 id 插入 steps_dim，对于已用容量，插入节点维度以进行广播
        # update_mask = (self.vehicle_id < vehicle_num)[:, None, None]
        # exceeds_cap = (ones_demands + self.used_capacity[:, :, None] > self.vehicle_capacity[torch.arange(self.vehicle_capacity.size(0)).unsqueeze(1), self.vehicle_id.type(torch.int64)].unsqueeze(1)) * update_mask

        # 无法访问的节点是那些已经被访问过(可以运需求过大的点)，或者车辆用完后不能再访问点
        mask_loc = visited_loc.to(torch.bool)
        # 判断是否可以访问depot，如果刚刚访问了仓库且仍有未服务的节点，则不能再次访问仓库; 如果用到最后一辆，也不能再访问
        mask_depot = ((self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)) | (self.vehicle_id >= vehicle_num - 1)[:, None]
        mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)
        return mask


    def construct_solutions(self, actions):
        return actions

