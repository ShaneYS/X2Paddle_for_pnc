# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

class Round(object):
    def __init__(self):
        # do not need any parameters
        pass

    def __call__(self, X):
        out = paddle.round(X)
        return out


class RotRPEAttentionWeightWithIndexComputation(object):
    # /data/project/baidu_for_rpc/baidu/adt-pnc/pnc-learning/scripts/prediction_models/models/lordformer_base_model_cuda_atten_torch/wayformer/ops/attention_rpe/attention_rpe.py
    def __init__(self):
        pass

    def __call__(self, query_features, key_features, key_attn_index, qk_relative_pos):
        # custom op
        bs_nhead = query_features.shape[1]
        total_query_num = query_features.shape[0]
        key_index_size = key_attn_index.shape[2]

        out = paddle.zeros([bs_nhead, total_query_num, key_index_size], dtype="float32")
        return out


class RotRPEProjectValueWithIndexComputation(object):
    def __init__(self):
        pass

    def __call__(self, attn_weight, value_features, key_attn_index, qk_relative_pos):
        # custom op
        total_query_num = attn_weight.shape[1]
        bs_nhead = attn_weight.shape[0]
        hdim = value_features.shape[2]

        out = paddle.zeros([total_query_num, bs_nhead, hdim], dtype="float32")
        return out


class LayerNormalization(object):

    def __init__(self, axis, epsilon):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, X, Scale, B):
        if self.axis == -1:
            self.axis = X.shape[-1]
        out = paddle.nn.functional.layer_norm(
            X,
            normalized_shape=self.axis,
            weight=Scale,
            bias=B,
            epsilon=self.epsilon,
            name=None,
        )
        return out


class Atan(object):
    def __init__(self):
        # do not need any parameters
        pass

    def __call__(self, input):
        out = paddle.atanh(input)
        return out


class GridSample(object):
    def __init__(self, align_corners, mode, padding_mode):
        self.align_corners = align_corners
        self.mode = mode
        self.padding_mode = padding_mode

    def __call__(self, X, grid):
        self.align_corners = bool(self.align_corners)
        out = paddle.nn.functional.grid_sample(
            X,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return out

import paddle
import numpy as np

# def paddle_scatter_elements(data, indices, updates, axis=0, reduction="none"):
#     """
#     Paddle实现ONNX ScatterElements算子
#     Args:
#         data (paddle.Tensor): 待更新的基础张量，形状为(D0, D1, ..., Dn)
#         indices (paddle.Tensor): 索引张量，形状与updates一致，值为axis维度上的索引
#         updates (paddle.Tensor): 更新张量，形状与indices一致
#         axis (int): 指定更新的轴，默认0
#         reduction (str): 更新方式，"none"（替换）、"add"（累加）、"mul"（相乘），默认"none"
#     Returns:
#         paddle.Tensor: 更新后的张量，形状与data一致
#     """
#     # 1. 校验参数合法性
#     assert reduction in ["none", "add", "mul"], f"不支持的reduction方式：{reduction}"
#     assert data.ndim == indices.ndim == updates.ndim, "data、indices、updates的维度必须一致"
#     assert indices.shape == updates.shape, "indices与updates的形状必须一致"
#     assert 0 <= axis < data.ndim, f"axis {axis} 超出data维度范围（0~{data.ndim-1}）"

#     # 2. 生成完整坐标（除axis外的所有维度的索引 + indices的axis维度索引）
#     # 2.1 获取各维度的形状
#     data_shape = data.shape
#     indices_shape = indices.shape  # 与updates一致

#     # 2.2 生成除axis外的所有维度的索引网格
#     # 例如：若shape=(2,3,4)，axis=1，则生成0维度和2维度的网格索引
#     mesh_dims = []
#     for dim in range(data.ndim):
#         if dim != axis:
#             # 生成当前维度的索引（0到data_shape[dim]-1）
#             mesh_dims.append(paddle.arange(data_shape[dim], dtype=paddle.int64))
#     # 生成网格索引（形状为(..., 非axis维度数)）
#     mesh = paddle.meshgrid(*mesh_dims, indexing="ij")  # 用"ij"模式匹配ONNX索引逻辑

#     # 2.3 将网格索引与indices组合为完整坐标（形状为(indices.numel(), data.ndim)）
#     # 调整indices维度顺序（与网格索引对齐）
#     # 例如：axis=1时，网格索引是(dim0, dim2)，indices是(dim0, dim1, dim2)，需将indices展平后与网格拼接
#     indices_flat = paddle.reshape(indices, [-1])  # 展平为1D
#     # 网格索引展平后拼接（每个维度的索引都展平为1D）
#     coord_list = []
#     dim_idx = 0  # 网格维度的索引
#     for dim in range(data.ndim):
#         if dim == axis:
#             # 当前维度是axis，用indices的值作为索引
#             coord_list.append(indices_flat)
#         else:
#             # 当前维度非axis，用网格索引的值
#             mesh_flat = paddle.reshape(mesh[dim_idx], [-1])  # 展平为1D
#             coord_list.append(mesh_flat)
#             dim_idx += 1
#     # 组合为坐标张量（形状：(N, data.ndim)，N=indices.numel()）
#     coordinates = paddle.stack(coord_list, axis=-1)

#     # 3. 展平updates（与坐标数量匹配）
#     updates_flat = paddle.reshape(updates, [-1])

#     # 4. 用scatter_nd执行更新（Paddle的scatter_nd支持reduce参数）
#     if reduction == "none":
#         # 直接替换（默认行为）
#         result = paddle.scatter_nd(
#             indices=coordinates,
#             updates=updates_flat,
#             shape=data_shape,
#             input=data  # 以data为初始值
#         )
#     else:
#         # 累加或相乘（需指定reduce参数）
#         result = paddle.scatter_nd(
#             indices=coordinates,
#             updates=updates_flat,
#             shape=data_shape,
#             input=data,
#             reduce=reduction  # "add"或"mul"
#         )

#     return result
class ScatterElements(object):
    def __init__(self, axis, reduction):
        self.axis = axis
        self.reduction = reduction

    def __call__(self, data, indices, updates):
        # out = paddle_scatter_elements(data, indices, updates, axis=self.axis, reduction=self.reduction)
        
        # 模拟：直接生成一个与data形状一致的0-1随机张量作为输出
        out = paddle.rand((*data.shape, ), dtype='float32')
        return out
