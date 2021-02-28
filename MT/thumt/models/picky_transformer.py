# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import thumt.utils as utils
import thumt.modules as modules

from thumt.utils import prune_linear_layer, prune_vector


class PickyAttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(PickyAttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.has_residual_transform = params.residual_transform
        self.pruned_heads = 0

        with utils.scope(name):
            self.attention = modules.PickyMultiHeadAttention(params.hidden_size,
                                                             params.head_size,
                                                             params.num_heads, 
                                                             params.attention_dropout,
                                                             params.weight_function)
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
        self.attention.q_transform = prune_linear_layer(self.attention.q_transform, index)
        self.attention.k_transform = prune_linear_layer(self.attention.k_transform, index)
        self.attention.v_transform = prune_linear_layer(self.attention.v_transform, index)
        self.attention.o_transform = prune_linear_layer(self.attention.o_transform, index, dim=1)

        # Update hyper params
        self.attention.num_heads = self.attention.num_heads - len(heads)
        self.attention.attention_hidden_size = self.attention.head_size * self.attention.num_heads
        self.pruned_heads += len(heads)

    def prune_dim(self, index):
        self.attention.prune_dim(index)

        if self.has_residual_transform:
            output_dim_to_reserve = utils.reverse_select(index["input"], self.residual_transform.weight.size(0))
            self.residual_transform = prune_linear_layer(self.residual_transform, output_dim_to_reserve)
            if not self.normalization == "before":
                self.layer_norm.prune_dim(output_dim_to_reserve)

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
        self.has_outer_transform = params.outer_transform

        with utils.scope(name):
            self.ffn_layer = modules.PickyFeedForward(params.hidden_size,
                                                      params.filter_size,
                                                      dropout=params.relu_dropout,
                                                      weight_function=params.weight_function)
            self.layer_norm = modules.FitLayerNorm(params.hidden_size)

            if self.thin_output:
                if self.has_outer_transform:
                    self.outer_transform = modules.Affine(params.hidden_size, params.hidden_size,
                                                          name="outer_transform")
                    self.reset_parameters()
                else:
                    self.outer_transform = lambda x: x

    def prune_dim(self, index):
        self.ffn_layer.prune_dim(index, prune_output=self.thin_output)
        self.input_hidden_size = self.input_hidden_size - len(index["input"])

        if self.thin_output:
            if self.has_outer_transform:
                output_dim_to_reserve = utils.reverse_select(index["output"], self.outer_transform.weight.size(1))
                self.outer_transform = prune_linear_layer(self.outer_transform, output_dim_to_reserve, dim=1)
                self.layer_norm.prune_dim(output_dim_to_reserve)
                self.output_hidden_size = len(output_dim_to_reserve)

    def forward(self, x):
        if self.input_hidden_size == 0:
            return x

        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            if self.thin_output:
                return self.outer_transform(x + y)
            else:
                return y
        else:
            if self.thin_output:
                return self.outer_transform(self.layer_norm(x + y))
            else:
                return self.layer_norm(y)

    def reset_parameters(self):
        nn.init.eye_(self.outer_transform.weight)
        nn.init.constant_(self.outer_transform.bias, 0.0)


class PickyTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(PickyTransformerEncoderLayer, self).__init__(name=name)

        self.additional_params = []
        self.skip_residual = params.skip_residual

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

        nn.init.constant_(self.kappa, 0.0)
        nn.init.constant_(self.ffn_input_weight, 0.0)
        nn.init.constant_(self.ffn_inter_weight, 0.0)
        if params.ffn_thin_output:
            nn.init.constant_(self.ffn_output_weight, 0.0)
    
    def load_additional_params(self):
        additional_params_dict = dict()
        additional_params_dict["kappa"] = self.kappa
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

        nn.init.constant_(self.self_kappa, 0.0)
        nn.init.constant_(self.encdec_kappa, 0.0)
        nn.init.constant_(self.ffn_input_weight, 0.0)
        nn.init.constant_(self.ffn_inter_weight, 0.0)
        if params.ffn_thin_output:
            nn.init.constant_(self.ffn_output_weight, 0.0)

    def load_additional_params(self):
        self_additional_params_dict = dict()
        encdec_additional_params_dict = dict()
        self_additional_params_dict["kappa"] = self.self_kappa
        encdec_additional_params_dict["kappa"] = self.encdec_kappa
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
        if self.normalization == "before":
            self.layer_norm.prune_dim(utils.reverse_select(index["output"], self.layer_norm.weight.size(0)))

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
        if self.normalization == "before":
            self.layer_norm.prune_dim(utils.reverse_select(index["output"], self.layer_norm.weight.size(0)))

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

    def find_pruneable_heads(self, p):
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
                    layer = self.encoder.layers[layer_num]
                    ffn_input_weight = layer.ffn_input_weight
                    ffn_inter_weight = layer.ffn_inter_weight

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
                        ffn_output_weight = layer.ffn_output_weight
                        output_dim_to_prune = math.floor(prune_ratio * ffn_output_weight.size(0))
                        output_indexes_to_prune = ffn_output_weight.topk(output_dim_to_prune, 
                                                                       largest=False, 
                                                                       sorted=False).indices.tolist()
                        indexes_to_prune["encoder"][layer_num]["output"] = output_indexes_to_prune

            for layer_num, heads in heads_to_prune["encdec"].items():
                if len(heads) > 0:
                    layer = self.decoder.layers[layer_num]
                    ffn_input_weight = layer.ffn_input_weight
                    ffn_inter_weight = layer.ffn_inter_weight

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
                        ffn_output_weight = layer.ffn_output_weight
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
                return -torch.sum(loss * mask, 1)
            else:
                return  torch.exp(-loss) * mask - (1 - mask)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

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
            ffn_thin_output=False, # will disable residual when set to False
            skip_residual=True,
            residual_transform=True,
            outer_transform=True,
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
