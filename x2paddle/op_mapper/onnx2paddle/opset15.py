# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from .opset14 import OpSet14
from x2paddle.core.util import *

def print_mapping_info(func):

    def run_mapping(*args, **kwargs):
        node = args[1]
        try:
            res = func(*args, **kwargs)
        except:
            raise Exception("convert failed node:{}, op_type is {}".format(
                node.name[9:], node.layer_type))
        else:
            return res

    return run_mapping


class OpSet15(OpSet14):

    def __init__(self, decoder, paddle_graph):
        super(OpSet15, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def Round(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer(
            "custom_layer:Round",
            inputs={"X": val_x.name},
            outputs=[node.name],
        )

    @print_mapping_info
    def RotRPEAttentionWeightWithIndexComputation(self, node):
        val_0 = self.graph.get_input_node(node, idx=0, copy=True)
        val_1 = self.graph.get_input_node(node, idx=1, copy=True)
        val_2 = self.graph.get_input_node(node, idx=2, copy=True)
        val_3 = self.graph.get_input_node(node, idx=3, copy=True)
        self.paddle_graph.add_layer(
            "custom_layer:RotRPEAttentionWeightWithIndexComputation",
            inputs={
                "query_features": val_0.name,
                "key_features": val_1.name,
                "key_attn_index": val_2.name,
                "qk_relative_pos": val_3.name,
            },
            outputs=[node.name],
        )

    @print_mapping_info
    def RotRPEProjectValueWithIndexComputation(self, node):
        val_0 = self.graph.get_input_node(node, idx=0, copy=True)
        val_1 = self.graph.get_input_node(node, idx=1, copy=True)
        val_2 = self.graph.get_input_node(node, idx=2, copy=True)
        val_3 = self.graph.get_input_node(node, idx=3, copy=True)
        self.paddle_graph.add_layer(
            "custom_layer:RotRPEProjectValueWithIndexComputation",
            inputs={
                "attn_weight": val_0.name,
                "value_features": val_1.name,
                "key_attn_index": val_2.name,
                "qk_relative_pos": val_3.name,
            },
            outputs=[node.name],
        )

    @print_mapping_info
    def LayerNormalization(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        val_z = self.graph.get_input_node(node, idx=2, copy=True)
        self.paddle_graph.add_layer(
            "custom_layer:LayerNormalization",
            inputs={"X": val_x.name, "Scale": val_y.name, "B": val_z.name},
            outputs=[node.name],
            axis=node.get_attr("axis"),
            epsilon=node.get_attr("epsilon"),
        )

    @print_mapping_info
    def Atan(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer(
            "custom_layer:Atan",
            inputs={"input": val_x.name},
            outputs=[node.name],
        )

    @print_mapping_info
    def GridSample(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer(
            "custom_layer:GridSample",
            inputs={"X": val_x.name, "grid": val_y.name},
            outputs=[node.name],
            align_corners=node.get_attr("align_corners"),
            mode=string(node.get_attr("mode")),
            padding_mode=string(node.get_attr("padding_mode")),
        )

    @print_mapping_info
    def ScatterElements(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        val_z = self.graph.get_input_node(node, idx=2, copy=True)
        axis = node.get_attr("axis")
        reduction = string(node.get_attr("reduction"))
        self.paddle_graph.add_layer(
            "custom_layer:ScatterElements",
            inputs={"data": val_x.name, "indices": val_y.name, "updates": val_z.name},
            outputs=[node.name],
            axis=axis,
            reduction=reduction,
        )
