# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import thumt.utils as utils
import thumt.modules as modules

from thumt.utils import prune_linear_layer, prune_vector
from thumt.utils import reinit_linear_layer, reinit_vector_


class PickyAttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(PickyAttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.has_residual_transform = params.residual_transform
        self.thin_ffn = params.thin_ffn
        self.pruned_heads = 0

        with utils.scope(name):
            self.attention = modules.PickyMultiHeadAttention(params.hidden_size,
                                                             params.head_size,
                                                             params.num_heads, 
                                                             params.attention_dropout,
                                                             params.weight_function,
                                                             params.fake_weight)
            self.layer_norm = modules.FitLayerNorm(params.hidden_size)

            if self.has_residual_transform:
                self.residual_transform = modules.Affine(params.hidden_size, params.hidden_size,
                                                         name="residual_transform")
                self.reset_parameters()
            else:
                self.residual_transform = lambda x: x

    def _prune_heads(self, heads):
        if len(heads) == 0:
            return
        index = utils.find_pruneable_heads_indices(
            heads, 
            self.attention.num_heads, 
            self.attention.head_size, 
        )

        # Prune linear layers
        self.attention.q_transform = prune_linear_layer(self.attention.q_transform, index, scale=False)
        self.attention.k_transform = prune_linear_layer(self.attention.k_transform, index, scale=False)
        self.attention.v_transform = prune_linear_layer(self.attention.v_transform, index, scale=False)
        self.attention.o_transform = prune_linear_layer(self.attention.o_transform, index, dim=1, scale=False)

        # Update hyper params
        self.attention.num_heads = self.attention.num_heads - len(heads)
        self.attention.attention_hidden_size = self.attention.head_size * self.attention.num_heads
        self.pruned_heads += len(heads)

    def prune_dim(self, index):
        if not self.thin_ffn:
            self.attention.prune_dim(index)

        if self.has_residual_transform:
            output_dim_to_reserve = utils.reverse_select(index["input"], self.residual_transform.weight.size(0))
            self.residual_transform = prune_linear_layer(self.residual_transform, output_dim_to_reserve, scale=False)
            if not self.normalization == "before":
                self.layer_norm.prune_dim(output_dim_to_reserve)

    def reinit_heads(self, heads):
        if len(heads) == 0:
            return
        index = utils.find_pruneable_heads_indices(
            heads, 
            self.attention.num_heads, 
            self.attention.head_size, 
        )
        # Reinitialize linear layers
        self.attention.q_transform = reinit_linear_layer(self.attention.q_transform, index, dim=0)
        self.attention.k_transform = reinit_linear_layer(self.attention.k_transform, index, dim=0)
        self.attention.v_transform = reinit_linear_layer(self.attention.v_transform, index, dim=0)
        self.attention.o_transform = reinit_linear_layer(self.attention.o_transform, index, dim=1)

    def reinit_dim(self, index):
        # no need to do this
        pass

    def forward(self, x, bias, memory=None, state=None):
        if self.attention.num_heads == 0:
            return x

        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y = self.attention(y, bias, memory, None)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, memory, kv)
            state["k"], state["v"] = k, v

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return self.residual_transform(x) + y
        else:
            return self.layer_norm(self.residual_transform(x) + y)

    def reset_parameters(self):
        nn.init.eye_(self.residual_transform.weight)
        nn.init.constant_(self.residual_transform.bias, 0.0)


class PickyFFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(PickyFFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.input_hidden_size = params.hidden_size
        self.output_hidden_size = params.hidden_size
        self.thin_output = params.ffn_thin_output
        self.has_exit_transform = params.exit_transform
        self.thin_ffn = params.thin_ffn

        with utils.scope(name):
            if self.thin_ffn:
                self.entry_transform = modules.Affine(params.hidden_size, params.hidden_size, name="entry_transform")
            self.ffn_layer = modules.PickyFeedForward(params.hidden_size,
                                                      params.filter_size,
                                                      dropout=params.relu_dropout,
                                                      weight_function=params.weight_function,
                                                      fake_weight=params.fake_weight)
            self.layer_norm = modules.FitLayerNorm(params.hidden_size)

            if self.thin_output:
                if self.has_exit_transform:
                    self.exit_transform = modules.Affine(params.hidden_size, params.hidden_size,
                                                          name="exit_transform")
                else:
                    self.exit_transform = lambda x: x

        self.reset_parameters()

    def prune_dim(self, index):
        self.ffn_layer.prune_dim(index, prune_output=self.thin_output)
        self.input_hidden_size = self.input_hidden_size - len(index["input"])

        if self.thin_ffn:
            input_dim_to_reserve = utils.reverse_select(index["input"], self.entry_transform.weight.size(0))
            self.entry_transform = prune_linear_layer(self.entry_transform, input_dim_to_reserve, dim=0)
            self.input_hidden_size = len(input_dim_to_reserve)

        if self.thin_output and self.has_exit_transform:
            output_dim_to_reserve = utils.reverse_select(index["output"], self.exit_transform.weight.size(1))
            self.exit_transform = prune_linear_layer(self.exit_transform, output_dim_to_reserve, dim=1)
            if not self.thin_ffn:
                self.layer_norm.prune_dim(output_dim_to_reserve)
            self.output_hidden_size = len(output_dim_to_reserve)

    def reinit_dim(self, index):
        self.ffn_layer.reinit_dim(index)

        if self.thin_ffn:
            input_dim_to_reserve = utils.reverse_select(index["input"], self.entry_transform.weight.size(0))
            self.entry_transform = reinit_linear_layer(self.entry_transform, input_dim_to_reserve, dim=0)

        if self.thin_output and self.has_exit_transform:
            output_dim_to_reserve = utils.reverse_select(index["output"], self.exit_transform.weight.size(1))
            self.exit_transform = reinit_linear_layer(self.exit_transform, output_dim_to_reserve, dim=1)

    def forward(self, x):
        if self.input_hidden_size == 0:
            return x

        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.thin_ffn:
            y = self.entry_transform(y)
        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            if self.thin_output:
                return self.exit_transform(x + y)
            else:
                return y
        else:
            if self.thin_output:
                if self.thin_ffn:
                    return self.layer_norm(x + self.exit_transform(y))
                else:
                    return self.exit_transform(self.layer_norm(x + y))
            else:
                return self.layer_norm(y)

    def reset_parameters(self):
        if self.thin_output and self.has_exit_transform:
            nn.init.eye_(self.exit_transform.weight)
            nn.init.constant_(self.exit_transform.bias, 0.0)
        if self.thin_ffn:
            nn.init.eye_(self.entry_transform.weight)
            nn.init.constant_(self.entry_transform.bias, 0.0)


class PickyTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(PickyTransformerEncoderLayer, self).__init__(name=name)

        self.additional_params = []
        self.skip_residual = params.skip_residual
        self.ffn_weights = params.ffn_weights

        with utils.scope(name):
            self.self_attention = PickyAttentionSubLayer(params)
            self.feed_forward = PickyFFNSubLayer(params)

            if params.skip_residual:
                self.layer_norm = modules.LayerNorm(params.hidden_size)

            # kappa for self attn
            self.kappa = nn.Parameter(torch.empty(params.num_heads))
            self.add_name(self.kappa, "kappa")
            self.additional_params.append(self.kappa)

            # weight for ffn hidden
            if self.ffn_weights:
                self.ffn_input_weight = nn.Parameter(torch.empty(params.hidden_size))
                self.ffn_inter_weight = nn.Parameter(torch.empty(params.filter_size))
                self.add_name(self.ffn_input_weight, "ffn_input_weight")
                self.add_name(self.ffn_inter_weight, "ffn_inter_weight")
                self.additional_params.append(self.ffn_input_weight)
                self.additional_params.append(self.ffn_inter_weight)
                if params.ffn_thin_output:
                    self.ffn_output_weight = nn.Parameter(torch.empty(params.hidden_size))
                    self.add_name(self.ffn_output_weight, "ffn_output_weight")
                    self.additional_params.append(self.ffn_output_weight)
                if params.weight_function.lower() == "relu":
                    nn.init.constant_(self.ffn_input_weight, 1.0)
                    nn.init.constant_(self.ffn_inter_weight, 1.0)
                else:
                    nn.init.constant_(self.ffn_input_weight, 0.0)
                    nn.init.constant_(self.ffn_inter_weight, 0.0)
                if params.ffn_thin_output:
                    if params.weight_function.lower() == "relu":
                        nn.init.constant_(self.ffn_output_weight, 1.0)
                    else:
                        nn.init.constant_(self.ffn_output_weight, 0.0)

        if params.weight_function.lower() == "relu":
            nn.init.constant_(self.kappa, 1.0)
        else:
            nn.init.constant_(self.kappa, 0.0)
    
    def load_additional_params(self):
        additional_params_dict = dict()
        additional_params_dict["kappa"] = self.kappa
        if self.ffn_weights:
            additional_params_dict["ffn_input_weight"] = self.ffn_input_weight
            additional_params_dict["ffn_inter_weight"] = self.ffn_inter_weight
            if self.feed_forward.thin_output:
                additional_params_dict["ffn_output_weight"] = self.ffn_output_weight
        self.self_attention.attention.additional_params = additional_params_dict
        self.feed_forward.ffn_layer.additional_params = additional_params_dict

    def prune_heads(self, heads):
        if len(heads) > 0:
            self.self_attention._prune_heads(heads)
            
            remain_heads = utils.reverse_select(heads, self.kappa.size(0))
            self.kappa = prune_vector(self.kappa, remain_heads, scale=False)
            self.add_name(self.kappa, "kappa")

    def prune_dim(self, index):
        if len(index["input"]) == 0 and len(index["inter"]) == 0 and len(index["output"]) == 0:
            return
        self.self_attention.prune_dim(index)
        self.feed_forward.prune_dim(index)
        
        if self.ffn_weights:
            input_index = utils.reverse_select(index["input"], self.ffn_input_weight.size(0))
            inter_index = utils.reverse_select(index["inter"], self.ffn_inter_weight.size(0))

            self.ffn_input_weight = prune_vector(self.ffn_input_weight, input_index, scale=False)
            self.ffn_inter_weight = prune_vector(self.ffn_inter_weight, inter_index, scale=False)
            if self.feed_forward.thin_output:
                output_index = utils.reverse_select(index["output"], self.ffn_output_weight.size(0))
                self.ffn_output_weight = prune_vector(self.ffn_output_weight, output_index, scale=False)
                self.add_name(self.ffn_output_weight, "ffn_output_weight")

            self.add_name(self.ffn_input_weight, "ffn_input_weight")
            self.add_name(self.ffn_inter_weight, "ffn_inter_weight")

    def reinit_heads(self, heads, recover_weights=False):
        if len(heads) > 0:
            self.self_attention.reinit_heads(heads)
        if recover_weights:
            with torch.no_grad():
                self.kappa.zero_()
        else:
            self.reinit_kappa(heads)

    def reinit_dim(self, index, recover_weights=False):
        if len(index["input"]) > 0 or len(index["inter"]) > 0 or len(index["output"]) > 0:
            self.feed_forward.reinit_dim(index)
        if self.ffn_weights:
            if recover_weights:
                with torch.no_grad():
                    self.ffn_input_weight.zero_()
                    self.ffn_inter_weight.zero_()
                    self.ffn_output_weight.zero_()
            else:
                self.reinit_ffn_weights(index)

    def reinit_kappa(self, heads=None):
        value = 1.0
        if heads is not None:
            remain_heads = utils.reverse_select(heads, self.kappa.size(0))
        else:
            remain_heads = []
        reinit_vector_(self.kappa, remain_heads, value)

    def reinit_ffn_weights(self, index=None):
        value = 1.0
        if self.ffn_weights:
            if index is not None:
                input_index = utils.reverse_select(index["input"], self.ffn_input_weight.size(0))
                inter_index = utils.reverse_select(index["inter"], self.ffn_inter_weight.size(0))
            else:
                input_index = []
                inter_index = []

            reinit_vector_(self.ffn_input_weight, input_index, value)
            reinit_vector_(self.ffn_inter_weight, inter_index, value)
            if self.feed_forward.thin_output:
                if index is not None:
                    output_index = utils.reverse_select(index["output"], self.ffn_output_weight.size(0))
                else:
                    output_index = []
                reinit_vector_(self.ffn_output_weight, output_index, value)

    def forward(self, x, bias):
        self.load_additional_params()
        y = self.self_attention(x, bias)
        y = self.feed_forward(y)

        if self.skip_residual:
            return self.layer_norm(x + y)
        else:
            return y

class PickyTransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(PickyTransformerDecoderLayer, self).__init__(name=name)

        self.additional_params = []
        self.skip_residual = params.skip_residual
        self.ffn_weights = params.ffn_weights

        with utils.scope(name):
            self.self_attention = PickyAttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = PickyAttentionSubLayer(params,
                                                    name="encdec_attention")
            self.feed_forward = PickyFFNSubLayer(params)

            if params.skip_residual:
                self.layer_norm = modules.LayerNorm(params.hidden_size)

            # kappa for self attn and encdec attn
            self.self_kappa = nn.Parameter(torch.empty(params.num_heads))
            self.encdec_kappa = nn.Parameter(torch.empty(params.num_heads))
            self.add_name(self.self_kappa, "self_kappa")
            self.add_name(self.encdec_kappa, "encdec_kappa")
            self.additional_params.append(self.self_kappa)
            self.additional_params.append(self.encdec_kappa)

            # weight for ffn hidden
            if self.ffn_weights:
                self.ffn_input_weight = nn.Parameter(torch.empty(params.hidden_size))
                self.ffn_inter_weight = nn.Parameter(torch.empty(params.filter_size))
                self.add_name(self.ffn_input_weight, "ffn_input_weight")
                self.add_name(self.ffn_inter_weight, "ffn_inter_weight")
                self.additional_params.append(self.ffn_input_weight)
                self.additional_params.append(self.ffn_inter_weight)
                if params.ffn_thin_output:
                    self.ffn_output_weight = nn.Parameter(torch.empty(params.hidden_size))
                    self.add_name(self.ffn_output_weight, "ffn_output_weight")
                    self.additional_params.append(self.ffn_output_weight)

                if params.weight_function.lower() == "relu":
                    nn.init.constant_(self.ffn_input_weight, 1.0)
                    nn.init.constant_(self.ffn_inter_weight, 1.0)
                else:
                    nn.init.constant_(self.ffn_input_weight, 0.0)
                    nn.init.constant_(self.ffn_inter_weight, 0.0)
                if params.ffn_thin_output:
                    if params.weight_function.lower() == "relu":
                        nn.init.constant_(self.ffn_output_weight, 1.0)
                    else:
                        nn.init.constant_(self.ffn_output_weight, 0.0)

        if params.weight_function.lower() == "relu":
            nn.init.constant_(self.self_kappa, 1.0)
            nn.init.constant_(self.encdec_kappa, 1.0)
        else:
            nn.init.constant_(self.self_kappa, 0.0)
            nn.init.constant_(self.encdec_kappa, 0.0)

    def load_additional_params(self):
        self_additional_params_dict = dict()
        encdec_additional_params_dict = dict()
        self_additional_params_dict["kappa"] = self.self_kappa
        encdec_additional_params_dict["kappa"] = self.encdec_kappa
        if self.ffn_weights:
            encdec_additional_params_dict["ffn_input_weight"] = self.ffn_input_weight
            encdec_additional_params_dict["ffn_inter_weight"] = self.ffn_inter_weight
            if self.feed_forward.thin_output:
                encdec_additional_params_dict["ffn_output_weight"] = self.ffn_output_weight
        self.self_attention.attention.additional_params = self_additional_params_dict
        self.encdec_attention.attention.additional_params = encdec_additional_params_dict
        self.feed_forward.ffn_layer.additional_params = encdec_additional_params_dict

    def self_prune_heads(self, heads):
        if len(heads) > 0:
            self.self_attention._prune_heads(heads)
            
            remain_heads = utils.reverse_select(heads, self.self_kappa.size(0))
            self.self_kappa = prune_vector(self.self_kappa, remain_heads, scale=False)
            self.add_name(self.self_kappa, "self_kappa")

    def encdec_prune_heads(self, heads):
        if len(heads) > 0:
            self.encdec_attention._prune_heads(heads)
            
            remain_heads = utils.reverse_select(heads, self.encdec_kappa.size(0))
            self.encdec_kappa = prune_vector(self.encdec_kappa, remain_heads, scale=False)
            self.add_name(self.encdec_kappa, "encdec_kappa")

    def prune_dim(self, index):
        if len(index["input"]) == 0 and len(index["inter"]) == 0 and len(index["output"]) == 0:
            return
        self.encdec_attention.prune_dim(index)
        self.feed_forward.prune_dim(index)

        if self.ffn_weights:
            input_index = utils.reverse_select(index["input"], self.ffn_input_weight.size(0))
            inter_index = utils.reverse_select(index["inter"], self.ffn_inter_weight.size(0))

            self.ffn_input_weight = prune_vector(self.ffn_input_weight, input_index, scale=False)
            self.ffn_inter_weight = prune_vector(self.ffn_inter_weight, inter_index, scale=False)
            if self.feed_forward.thin_output:
                output_index = utils.reverse_select(index["output"], self.ffn_output_weight.size(0))
                self.ffn_output_weight = prune_vector(self.ffn_output_weight, output_index, scale=False)
                self.add_name(self.ffn_output_weight, "ffn_output_weight")

            self.add_name(self.ffn_input_weight, "ffn_input_weight")
            self.add_name(self.ffn_inter_weight, "ffn_inter_weight")

    def self_reinit_heads(self, heads, recover_weights=False):
        if len(heads) > 0:
            self.self_attention.reinit_heads(heads)
        if recover_weights:
            with torch.no_grad():
                self.self_kappa.zero_()
        else:
            self.reinit_self_kappa(heads)

    def encdec_reinit_heads(self, heads, recover_weights=False):
        if len(heads) > 0:
            self.encdec_attention.reinit_heads(heads)
        if recover_weights:
            with torch.no_grad():
                self.encdec_kappa.zero_()
        else:
            self.reinit_encdec_kappa(heads)

    def reinit_dim(self, index, recover_weights=False):
        if len(index["input"]) > 0 or len(index["inter"]) > 0 or len(index["output"]) > 0:
            self.feed_forward.reinit_dim(index)
        if self.ffn_weights:
            if recover_weights:
                with torch.no_grad():
                    self.ffn_input_weight.zero_()
                    self.ffn_inter_weight.zero_()
                    self.ffn_output_weight.zero_()
            else:
                self.reinit_ffn_weights(index)

    def reinit_self_kappa(self, heads=None):
        value = 1.0
        if heads is not None:
            remain_heads = utils.reverse_select(heads, self.self_kappa.size(0))
        else:
            remain_heads = []
        reinit_vector_(self.self_kappa, remain_heads, value)

    def reinit_encdec_kappa(self, heads=None):
        value = 1.0
        if heads is not None:
            remain_heads = utils.reverse_select(heads, self.encdec_kappa.size(0))
        else:
            remain_heads = []
        reinit_vector_(self.encdec_kappa, remain_heads, value)

    def reinit_ffn_weights(self, index=None):
        value = 1.0
        if self.ffn_weights:
            if index is not None:
                input_index = utils.reverse_select(index["input"], self.ffn_input_weight.size(0))
                inter_index = utils.reverse_select(index["inter"], self.ffn_inter_weight.size(0))
            else:
                input_index = []
                inter_index = []

            reinit_vector_(self.ffn_input_weight, input_index, value)
            reinit_vector_(self.ffn_inter_weight, inter_index, value)
            if self.feed_forward.thin_output:
                if index is not None:
                    output_index = utils.reverse_select(index["output"], self.ffn_output_weight.size(0))
                else:
                    output_index = []
                reinit_vector_(self.ffn_output_weight, output_index, value)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        self.load_additional_params()
        y = self.self_attention(x, attn_bias, state=state)
        y = self.encdec_attention(y, encdec_bias, memory)
        y = self.feed_forward(y)
        if self.skip_residual:
            return self.layer_norm(x + y)
        else:
            return y


class PickyTransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(PickyTransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                PickyTransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.FitLayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

        self.additional_params = []
        for layer in self.layers:
            self.additional_params += layer.additional_params

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            layer = int(layer)
            self.layers[layer].prune_heads(heads)

    def prune_dim(self, indexes):
        for (_, index), layer in list(zip(indexes.items(), self.layers)):
            layer.prune_dim(index)
        #if self.normalization == "before":
        #    self.layer_norm.prune_dim(utils.reverse_select(index["output"], self.layer_norm.weight.size(0)))

    def _reinit_heads(self, heads_to_reinit, recover_weights=False):
        for layer, heads in heads_to_reinit.items():
            layer = int(layer)
            self.layers[layer].reinit_heads(heads, recover_weights)

    def reinit_dim(self, indexes, recover_weights=False):
        for (_, index), layer in list(zip(indexes.items(), self.layers)):
            layer.reinit_dim(index, recover_weights)

    def reinit_kappa_and_ffn_weights(self):
        for layer in self.layers:
            layer.reinit_kappa()
            layer.reinit_ffn_weights()


    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class PickyTransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(PickyTransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                PickyTransformerDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.FitLayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

        self.additional_params = []
        for layer in self.layers:
            self.additional_params += layer.additional_params

    def _prune_heads(self, self_heads_to_prune, encdec_heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in self_heads_to_prune.items():
            layer = int(layer)
            self.layers[layer].self_prune_heads(heads)

        for layer, heads in encdec_heads_to_prune.items():
            layer = int(layer)
            self.layers[layer].encdec_prune_heads(heads)

    def prune_dim(self, indexes):
        for (_, index), layer in list(zip(indexes.items(), self.layers)):
            layer.prune_dim(index)
        #if self.normalization == "before":
        #    self.layer_norm.prune_dim(utils.reverse_select(index["output"], self.layer_norm.weight.size(0)))

    def _reinit_heads(self, self_heads_to_reinit, encdec_heads_to_reinit, recover_weights=False):
        for layer, heads in self_heads_to_reinit.items():
            layer = int(layer)
            self.layers[layer].self_reinit_heads(heads, recover_weights)

        for layer, heads in encdec_heads_to_reinit.items():
            layer = int(layer)
            self.layers[layer].encdec_reinit_heads(heads, recover_weights)

    def reinit_dim(self, indexes, recover_weights=False):
        for (_, index), layer in list(zip(indexes.items(), self.layers)):
            layer.reinit_dim(index, recover_weights)

    def reinit_kappa_and_ffn_weights(self):
        for layer in self.layers:
            layer.reinit_self_kappa()
            layer.reinit_encdec_kappa()
            layer.reinit_ffn_weights()

    def forward(self, x, attn_bias, encdec_bias, memory, state=None):
        for i, layer in enumerate(self.layers):
            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory,
                          state["decoder"]["layer_%d" % i])
            else:
                x = layer(x, attn_bias, encdec_bias, memory, None)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class PickyTransformer(modules.Module):

    def __init__(self, params, name="picky_transformer"):
        super(PickyTransformer, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = PickyTransformerEncoder(params)
            self.decoder = PickyTransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.attention_hidden_size = params.num_heads * params.head_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.ffn_thin_output = params.ffn_thin_output
        self.ffn_weights = params.ffn_weights
        self.head_weight_loss = params.head_weight_loss
        self.head_weight_loss_weight = params.head_weight_loss_weight

        self.additional_params = self.encoder.additional_params + self.decoder.additional_params
        self.reset_parameters()

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.bias, "bias")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.normal_(self.softmax_weights, mean=0.0,
                            std=self.params.hidden_size ** -0.5)

    def find_pruned_heads(self):
        heads_to_prune = {
                "encoder": {i: list(range(layer.self_attention.pruned_heads)) 
                            for i, layer in enumerate(self.encoder.layers)},
                "decoder": {i: list(range(layer.self_attention.pruned_heads)) 
                            for i, layer in enumerate(self.decoder.layers)},
                "encdec": {i: list(range(layer.encdec_attention.pruned_heads)) 
                            for i, layer in enumerate(self.decoder.layers)}
                         }
        return heads_to_prune

    def get_binary_head_mask(self, heads_to_prune):
        binary_mask = []
        for num_layer, layer in enumerate(self.encoder.layers):
            for num_head, k in enumerate(layer.kappa):
                if num_head in heads_to_prune["encoder"][num_layer]:
                    binary_mask.append(1)
                else:
                    binary_mask.append(0)

        for num_layer, layer in enumerate(self.decoder.layers):
            for num_head, k in enumerate(layer.self_kappa):
                if num_head in heads_to_prune["decoder"][num_layer]:
                    binary_mask.append(1)
                else:
                    binary_mask.append(0)
            for num_head, k in enumerate(layer.encdec_kappa):
                if num_head in heads_to_prune["encdec"][num_layer]:
                    binary_mask.append(1)
                else:
                    binary_mask.append(0)
        
        return torch.Tensor(binary_mask).long()

    def get_heads_from_mask(self, binary_mask):
            heads_to_prune = {
                    "encoder": {layer: [] for layer, _ in enumerate(self.encoder.layers)},
                    "decoder": {layer: [] for layer, _ in enumerate(self.decoder.layers)},
                    "encdec": {layer: [] for layer, _ in enumerate(self.decoder.layers)}
                             }
            ind = 0
            for num_layer, layer in enumerate(self.encoder.layers):
                for num_head, k in enumerate(layer.kappa):
                    if binary_mask[ind]:
                        heads_to_prune["encoder"][num_layer].append(num_head)
                    ind += 1

            for num_layer, layer in enumerate(self.decoder.layers):
                for num_head, k in enumerate(layer.self_kappa):
                    if binary_mask[ind]:
                        heads_to_prune["decoder"][num_layer].append(num_head)
                    ind += 1
                for num_head, k in enumerate(layer.encdec_kappa):
                    if binary_mask[ind]:
                        heads_to_prune["encdec"][num_layer].append(num_head)
                    ind += 1

            return heads_to_prune


    def find_pruneable_heads(self, p, random=False, layerwise=False):
        with torch.no_grad():
            heads_to_prune = {
                    "encoder": {layer: [] for layer, _ in enumerate(self.encoder.layers)},
                    "decoder": {layer: [] for layer, _ in enumerate(self.decoder.layers)},
                    "encdec": {layer: [] for layer, _ in enumerate(self.decoder.layers)}
                             }
            encoder_kappa = torch.cat([layer.kappa for layer in self.encoder.layers])
            decoder_kappa = torch.cat([layer.self_kappa for layer in self.decoder.layers])
            encdec_kappa = torch.cat([layer.encdec_kappa for layer in self.decoder.layers])
            all_kappa = torch.cat([encoder_kappa, decoder_kappa, encdec_kappa])
            if random:
                fake_kappas = []
                for num_layer, layer in enumerate(self.encoder.layers):
                    layer.fake_kappa = torch.zeros_like(layer.kappa).normal_()
                    fake_kappas.append(layer.fake_kappa)
                for num_layer, layer in enumerate(self.decoder.layers):
                    layer.fake_self_kappa = torch.zeros_like(layer.self_kappa).normal_()
                    layer.fake_encdec_kappa = torch.zeros_like(layer.encdec_kappa).normal_()
                    fake_kappas.append(layer.fake_self_kappa)
                    fake_kappas.append(layer.fake_encdec_kappa)

                for kappa in fake_kappas:
                    dist.broadcast(kappa, 0)

                all_kappa = torch.cat(fake_kappas)
                num_heads_to_prune = math.floor(p * all_kappa.size(0))
                threshold = all_kappa.sort()[0][num_heads_to_prune].item()

                for num_layer, layer in enumerate(self.encoder.layers):
                    for num_head, k in enumerate(layer.fake_kappa):
                        if k.item() < threshold:
                            heads_to_prune["encoder"][num_layer].append(num_head)
                    del layer.fake_kappa

                for num_layer, layer in enumerate(self.decoder.layers):
                    for num_head, k in enumerate(layer.fake_self_kappa):
                        if k.item() < threshold:
                            heads_to_prune["decoder"][num_layer].append(num_head)
                    del layer.fake_self_kappa
                    for num_head, k in enumerate(layer.fake_encdec_kappa):
                        if k.item() < threshold:
                            heads_to_prune["encdec"][num_layer].append(num_head)
                    del layer.fake_encdec_kappa

                return heads_to_prune

            if layerwise:
                for num_layer, layer in enumerate(self.encoder.layers):
                    num_heads_to_prune = math.floor(p * layer.kappa.size(0))
                    threshold = layer.kappa.sort()[0][num_heads_to_prune].item()
                    for num_head, k in enumerate(layer.kappa):
                        if k.item() < threshold:
                            heads_to_prune["encoder"][num_layer].append(num_head)

                for num_layer, layer in enumerate(self.decoder.layers):
                    num_heads_to_prune = math.floor(p * layer.self_kappa.size(0))
                    threshold = layer.self_kappa.sort()[0][num_heads_to_prune].item()
                    for num_head, k in enumerate(layer.self_kappa):
                        if k.item() < threshold:
                            heads_to_prune["decoder"][num_layer].append(num_head)
                    num_heads_to_prune = math.floor(p * layer.encdec_kappa.size(0))
                    threshold = layer.encdec_kappa.sort()[0][num_heads_to_prune].item()
                    for num_head, k in enumerate(layer.encdec_kappa):
                        if k.item() < threshold:
                            heads_to_prune["encdec"][num_layer].append(num_head)

                return heads_to_prune

            num_heads_to_prune = math.floor(p * all_kappa.size(0))
            threshold = all_kappa.sort()[0][num_heads_to_prune].item()

            for num_layer, layer in enumerate(self.encoder.layers):
                for num_head, k in enumerate(layer.kappa):
                    if k.item() < threshold:
                        heads_to_prune["encoder"][num_layer].append(num_head)

            for num_layer, layer in enumerate(self.decoder.layers):
                for num_head, k in enumerate(layer.self_kappa):
                    if k.item() < threshold:
                        heads_to_prune["decoder"][num_layer].append(num_head)
                for num_head, k in enumerate(layer.encdec_kappa):
                    if k.item() < threshold:
                        heads_to_prune["encdec"][num_layer].append(num_head)
                
        return heads_to_prune

    def find_pruneable_dim(self, heads_to_prune):
        """
            return dims to prune
        """
        with torch.no_grad():
            indexes_to_prune = {
                    "encoder": {layer: {"input": [], "inter": [], "output": []} 
                                for layer, _ in enumerate(self.encoder.layers)},
                    "decoder": {layer: {"input": [], "inter": [], "output": []} 
                                for layer, _ in enumerate(self.decoder.layers)}
                               }

            for layer_num, heads in heads_to_prune["encoder"].items():
                if len(heads) > 0:
                    layer_num = int(layer_num)
                    layer = self.encoder.layers[layer_num]
                    if self.ffn_weights:
                        ffn_input_weight = layer.ffn_input_weight
                        ffn_inter_weight = layer.ffn_inter_weight
                    else:
                        ffn_input_size = layer.feed_forward.ffn_layer.input_transform.weight.size(1)
                        ffn_inter_size = layer.feed_forward.ffn_layer.input_transform.weight.size(0)
                        ffn_input_weight = torch.randn(ffn_input_size)
                        ffn_inter_weight = torch.randn(ffn_inter_size)
                        ffn_input_weight = ffn_input_weight.to(layer.feed_forward.ffn_layer.input_transform.weight)
                        ffn_inter_weight = ffn_inter_weight.to(layer.feed_forward.ffn_layer.input_transform.weight)

                    prune_ratio = len(heads) / layer.self_attention.attention.num_heads
                    
                    input_dim_to_prune = math.floor(prune_ratio * ffn_input_weight.size(0))
                    input_indexes_to_prune = ffn_input_weight.topk(input_dim_to_prune, 
                                                                   largest=False, 
                                                                   sorted=False).indices.tolist()
                    indexes_to_prune["encoder"][layer_num]["input"] = input_indexes_to_prune

                    inter_dim_to_prune = math.floor(prune_ratio * ffn_inter_weight.size(0))
                    inter_indexes_to_prune = ffn_inter_weight.topk(inter_dim_to_prune, 
                                                                   largest=False, 
                                                                   sorted=False).indices.tolist()
                    indexes_to_prune["encoder"][layer_num]["inter"] = inter_indexes_to_prune

                    if self.ffn_thin_output:
                        if self.ffn_weights:
                            ffn_output_weight = layer.ffn_output_weight
                        else:
                            output_transform = layer.feed_forward.ffn_layer.output_transform
                            ffn_output_size = output_transform.weight.size(0)
                            ffn_output_weight = torch.randn(ffn_output_size)
                            ffn_output_weight = ffn_output_weight.to(output_transform.weight)

                        output_dim_to_prune = math.floor(prune_ratio * ffn_output_weight.size(0))
                        output_indexes_to_prune = ffn_output_weight.topk(output_dim_to_prune, 
                                                                       largest=False, 
                                                                       sorted=False).indices.tolist()
                        indexes_to_prune["encoder"][layer_num]["output"] = output_indexes_to_prune

            for layer_num, heads in heads_to_prune["encdec"].items():
                if len(heads) > 0:
                    layer_num = int(layer_num)
                    layer = self.decoder.layers[layer_num]
                    if self.ffn_weights:
                        ffn_input_weight = layer.ffn_input_weight
                        ffn_inter_weight = layer.ffn_inter_weight
                    else:
                        ffn_input_size = layer.feed_forward.ffn_layer.input_transform.weight.size(1)
                        ffn_inter_size = layer.feed_forward.ffn_layer.input_transform.weight.size(0)
                        ffn_input_weight = torch.randn(ffn_input_size)
                        ffn_inter_weight = torch.randn(ffn_inter_size)
                        ffn_input_weight = ffn_input_weight.to(layer.feed_forward.ffn_layer.input_transform.weight)
                        ffn_inter_weight = ffn_inter_weight.to(layer.feed_forward.ffn_layer.input_transform.weight)

                    prune_ratio = len(heads) / layer.encdec_attention.attention.num_heads
                    
                    input_dim_to_prune = math.floor(prune_ratio * ffn_input_weight.size(0))
                    input_indexes_to_prune = ffn_input_weight.topk(input_dim_to_prune, 
                                                                   largest=False, 
                                                                   sorted=False).indices.tolist()
                    indexes_to_prune["decoder"][layer_num]["input"] = input_indexes_to_prune

                    inter_dim_to_prune = math.floor(prune_ratio * ffn_inter_weight.size(0))
                    inter_indexes_to_prune = ffn_inter_weight.topk(inter_dim_to_prune, 
                                                                   largest=False, 
                                                                   sorted=False).indices.tolist()
                    indexes_to_prune["decoder"][layer_num]["inter"] = inter_indexes_to_prune

                    if self.ffn_thin_output:
                        if self.ffn_weights:
                            ffn_output_weight = layer.ffn_output_weight
                        else:
                            output_transform = layer.feed_forward.ffn_layer.output_transform
                            ffn_output_size = output_transform.weight.size(0)
                            ffn_output_weight = torch.randn(ffn_output_size)
                            ffn_output_weight = ffn_output_weight.to(output_transform.weight)

                        output_dim_to_prune = math.floor(prune_ratio * ffn_output_weight.size(0))
                        output_indexes_to_prune = ffn_output_weight.topk(output_dim_to_prune, 
                                                                       largest=False, 
                                                                       sorted=False).indices.tolist()
                        indexes_to_prune["decoder"][layer_num]["output"] = output_indexes_to_prune

        return indexes_to_prune

    def prune_heads(self, heads_to_prune):
        encoder_heads_to_prune = heads_to_prune["encoder"]
        decoder_heads_to_prune = heads_to_prune["decoder"]
        encdec_heads_to_prune = heads_to_prune["encdec"]

        self.encoder._prune_heads(encoder_heads_to_prune)
        self.decoder._prune_heads(decoder_heads_to_prune, encdec_heads_to_prune)

    def prune_dim(self, indexes):
        """
            index: indexes of dimension to prune
        """
        encoder_indexes = indexes["encoder"]
        decoder_indexes = indexes["decoder"]

        self.encoder.prune_dim(indexes=encoder_indexes)
        self.decoder.prune_dim(indexes=decoder_indexes)

    def reinitialize_heads_and_dims(self, heads_to_reinit, indexes_to_reinit, recover_weights=False):
        encoder_heads_to_reinit = heads_to_reinit["encoder"]
        decoder_heads_to_reinit = heads_to_reinit["decoder"]
        encdec_heads_to_reinit = heads_to_reinit["encdec"]
        self.encoder._reinit_heads(encoder_heads_to_reinit, recover_weights)
        self.decoder._reinit_heads(decoder_heads_to_reinit, encdec_heads_to_reinit, recover_weights)
        
        encoder_indexes = indexes_to_reinit["encoder"]
        decoder_indexes = indexes_to_reinit["decoder"]
        self.encoder.reinit_dim(encoder_indexes, recover_weights)
        self.decoder.reinit_dim(decoder_indexes, recover_weights)

    def reinit_kappa_and_ffn_weights(self):
        self.encoder.reinit_kappa_and_ffn_weights()
        self.decoder.reinit_kappa_and_ffn_weights()

    def get_trainable_masks(self, trainable_parameters, heads_to_reinit, indexes_to_reinit):
        trainable_masks = []
        head_size = self.params.head_size
        for name, var in trainable_parameters:
            parse_name = name.split(".")
            if parse_name[0] in ["encoder", "decoder"]:
                layer_num = int(parse_name[2])
                module = parse_name[3]
                if module == "self_attention":
                    reinit_heads = heads_to_reinit[parse_name[0]][layer_num]
                    if len(reinit_heads) > 0:
                        if parse_name[-2] == "layer_norm" or parse_name[-1] == "bias":
                            trainable_masks.append(torch.zeros_like(var).bool())
                        else:
                            if parse_name[-2] == "o_transform":
                                trainable_masks.append(utils.headwise_mask(var, head_size, 0, reinit_heads))
                            else:
                                trainable_masks.append(utils.headwise_mask(var, head_size, 1, reinit_heads))
                    else:
                        trainable_masks.append(torch.zeros_like(var).bool())
                elif module == "encdec_attention":
                    reinit_heads = heads_to_reinit["encdec"][layer_num]
                    if len(reinit_heads) > 0:
                        if parse_name[-2] == "layer_norm" or parse_name[-1] == "bias":
                            trainable_masks.append(torch.zeros_like(var).bool())
                        else:
                            if parse_name[-2] == "o_transform":
                                trainable_masks.append(utils.headwise_mask(var, head_size, 1, reinit_heads))
                            else:
                                trainable_masks.append(utils.headwise_mask(var, head_size, 0, reinit_heads))
                    else:
                        trainable_masks.append(torch.zeros_like(var).bool())
                elif module == "feed_forward":
                    reinit_dims = indexes_to_reinit[parse_name[0]][layer_num]
                    ffn_module = parse_name[4]
                    if len(reinit_dims["input"]) > 0 or len(reinit_dims["inter"]) > 0 or len(reinit_dims["output"]) > 0:
                        if ffn_module == "entry_transform":
                            reinit_dim = reinit_dims["input"]
                            trainable_masks.append(utils.headwise_mask(var, 1, 0, reinit_dim))
                        elif ffn_module == "ffn_layer":
                            reinit_input_dim = reinit_dims["input"]
                            reinit_inter_dim = reinit_dims["inter"]
                            reinit_output_dim = reinit_dims["output"]
                            if parse_name[-2] == "input_transform":
                                t_mask = utils.headwise_mask(var, 1, 0, reinit_inter_dim)
                                if parse_name[-1] == "weight":
                                    t_mask = utils.headwise_mask(t_mask, 1, 1, reinit_input_dim)
                                trainable_masks.append(t_mask)
                            elif parse_name[-2] == "output_transform":
                                t_mask = utils.headwise_mask(var, 1, 0, reinit_output_dim)
                                if parse_name[-1] == "weight":
                                    t_mask = utils.headwise_mask(t_mask, 1, 1, reinit_inter_dim)
                                trainable_masks.append(t_mask)
                        elif ffn_module == "exit_transform":
                            reinit_dim = reinit_dims["output"]
                            if parse_name[-1] == "weight":
                                trainable_masks.append(utils.headwise_mask(var, 1, 1, reinit_dim))
                            else:
                                trainable_masks.append(torch.zeros_like(var).bool())
                        elif ffn_module == "layer_norm":
                            trainable_masks.append(torch.zeros_like(var).bool())
                    else:
                        trainable_masks.append(torch.zeros_like(var).bool())
            else:
                trainable_masks.append(torch.zeros_like(var).bool())

        return trainable_masks

    def summary_weights(self, summary, step, accumulate_steps=100):
        # summary mean in step interval
        if not hasattr(self, "weights"):
            self.weights = {
                    "encoder": {"layer_{}".format(i): [0. for j in range(self.params.num_heads)] 
                                for i in range(len(self.encoder.layers))},
                    "decoder": {"layer_{}".format(i): [0. for j in range(self.params.num_heads)] 
                                for i in range(len(self.decoder.layers))},
                    "encdec": {"layer_{}".format(i): [0. for j in range(self.params.num_heads)] 
                               for i in range(len(self.decoder.layers))},
                           }

        for layer_i, layer in enumerate(self.encoder.layers):
            if hasattr(layer.self_attention.attention, "weights"):
                weights = layer.self_attention.attention.weights.tolist()
                for head_i, w in enumerate(weights):
                    summary.scalar("encoder layer_{}/head_{}".format(layer_i, head_i), 
                                   self.weights["encoder"]["layer_{}".format(layer_i)][head_i] / accumulate_steps,
                                   step)
                    if step % accumulate_steps == 0:
                        self.weights["encoder"]["layer_{}".format(layer_i)][head_i] = 0.
                    else:
                        self.weights["encoder"]["layer_{}".format(layer_i)][head_i] += w

        for layer_i, layer in enumerate(self.decoder.layers):
            if hasattr(layer.self_attention.attention, "weights"):
                self_weights = layer.self_attention.attention.weights.tolist()
                for head_i, w in enumerate(self_weights):
                    summary.scalar("decoder layer_{}/head_{}".format(layer_i, head_i), 
                                   self.weights["decoder"]["layer_{}".format(layer_i)][head_i] / accumulate_steps,
                                   step)
                    if step % accumulate_steps == 0:
                        self.weights["decoder"]["layer_{}".format(layer_i)][head_i] = 0.
                    else:
                        self.weights["decoder"]["layer_{}".format(layer_i)][head_i] += w
            if hasattr(layer.encdec_attention.attention, "weights"):
                encdec_weights = layer.encdec_attention.attention.weights.tolist()
                for head_i, w in enumerate(encdec_weights):
                    summary.scalar("encdec layer_{}/head_{}".format(layer_i, head_i), 
                                   self.weights["encdec"]["layer_{}".format(layer_i)][head_i] / accumulate_steps,
                                   step)
                    if step % accumulate_steps == 0:
                        self.weights["encdec"]["layer_{}".format(layer_i)][head_i] = 0.
                    else:
                        self.weights["encdec"]["layer_{}".format(layer_i)][head_i] += w

    def compute_head_weight_loss(self):
        weight_func = self.encoder.layers[0].self_attention.attention.compute_weight
        encoder_kappa = torch.cat([weight_func(layer.kappa) for layer in self.encoder.layers])
        decoder_kappa = torch.cat([weight_func(layer.self_kappa) for layer in self.decoder.layers])
        encdec_kappa = torch.cat([weight_func(layer.encdec_kappa) for layer in self.decoder.layers])
        all_kappa = torch.cat([encoder_kappa, decoder_kappa, encdec_kappa])
        if self.head_weight_loss == "half":
            crit = torch.ones_like(all_kappa) * 0.5
        elif self.head_weight_loss == "zero":
            crit = torch.zeros_like(all_kappa)
        loss = (all_kappa - crit).norm(p=1) * self.head_weight_loss_weight
        return loss

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = nn.functional.dropout(self.encoding(decoder_input),
                                              self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, mode=mode)
        loss = self.criterion(logits, labels)
        mask = mask.to(torch.float32)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        if mode == "eval":
            if level == "sentence":
                loss = -torch.sum(loss * mask, 1)
            else:
                loss = torch.exp(-loss) * mask - (1 - mask)
        else:
            loss = (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

        if not self.head_weight_loss == "none":
            weights_loss = self.compute_head_weight_loss()
            loss += weights_loss

        return loss

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, layer.self_attention.attention.attention_hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, layer.self_attention.attention.attention_hidden_size],
                                     device=device)
                } for i, layer in enumerate(self.decoder.layers)
            }
        }

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            head_size=64,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            weight_function="sigmoid",
            ffn_thin_output=True, # will disable residual when set to False
            skip_residual=False,
            residual_transform=False,
            exit_transform=True,
            ffn_weights=False,
            thin_ffn=True,
            head_weight_loss="none",
            head_weight_loss_weight=0.1,
            fake_weight=False,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def base_params_v2():
        params = PickyTransformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = PickyTransformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = PickyTransformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def test_params():
        params = PickyTransformer.base_params()
        params.hidden_size = 64
        params.head_size = 16
        params.filter_size = 256
        params.num_heads = 4
        params.residual_dropout = 0.0
        params.learning_rate = 5e-4
        params.train_steps = 100000
        params.num_encoder_layers = 3
        params.num_decoder_layers = 3

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return PickyTransformer.base_params()
        elif name == "base_v2":
            return PickyTransformer.base_params_v2()
        elif name == "big":
            return PickyTransformer.big_params()
        elif name == "big_v2":
            return PickyTransformer.big_params_v2()
        elif name == "test":
            return PickyTransformer.test_params()
        else:
            return PickyTransformer.base_params()
