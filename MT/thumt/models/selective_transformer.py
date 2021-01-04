# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import thumt.utils as utils
import thumt.modules as modules
from thumt.utils import prune_linear_layer, prune_vector
from thumt.modules.affine import Affine


class AttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.pruned_heads = set()

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(params.hidden_size, 
                                                        params.num_heads, 
                                                        params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def _prune_heads(self, heads, pruned_heads):
        if len(heads) == 0:
            return
        heads, index = utils.find_pruneable_heads_and_indices(
            heads, 
            self.attention.num_heads, 
            self.attention.head_size, 
            pruned_heads
        )

        # Prune linear layers
        self.attention.q_transform = prune_linear_layer(self.attention.q_transform, index)
        self.attention.k_transform = prune_linear_layer(self.attention.k_transform, index)
        self.attention.v_transform = prune_linear_layer(self.attention.v_transform, index)
        self.attention.o_transform = prune_linear_layer(self.attention.o_transform, index, dim=1)

        # Update hyper params
        self.attention.num_heads = self.attention.num_heads - len(heads)
        self.attention.hidden_size = self.attention.head_size * self.attention.num_heads


    def forward(self, x, bias, memory=None, state=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.attention.num_heads == 0:
            if self.normalization == "before":
                return x
            else:
                return self.layer_norm(x)

        if self.training or state is None:
            y = self.attention(y, bias, memory, None)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, memory, kv)
            state["k"], state["v"] = k, v

        y = F.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)

    def forward_with_head_analysis(self, x, bias, memory=None, state=None, mode=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y, head_feature = self.attention.forward_with_head_analysis(y, bias, memory, None, mode=mode)
        else:
            kv = [state["k"], state["v"]]
            y, k, v, head_feature = self.attention.forward_with_head_analysis(y, bias, memory, kv, mode=mode)
            state["k"], state["v"] = k, v

        y = F.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y, head_feature
        else:
            return self.layer_norm(x + y), head_feature


class SelectiveAttentionSubLayer(modules.Module):

    def __init__(self, params, name="selective_attention"):
        super(SelectiveAttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.pruned_heads = set()

        with utils.scope(name):
            self.attention = modules.SelectiveMultiHeadAttention(
                    params.hidden_size,
                    params.head_size, 
                    params.num_heads, 
                    params.attention_dropout,
                    input_aware_select=params.input_aware_select,
                    select_weight_function=params.select_weight_function,
                    select_method=params.select_method,
                    select_number=params.select_number
                )
            self.layer_norm = modules.LayerNorm(params.hidden_size)

        self.additional_params = self.attention.additional_params

    def _prune_heads(self, heads, pruned_heads):
        if len(heads) == 0:
            return
        heads, index = utils.find_pruneable_heads_and_indices(
            heads, 
            self.attention.num_heads, 
            self.attention.head_size, 
            pruned_heads
        )

        # Prune linear layers
        self.attention.q_transform = prune_linear_layer(self.attention.q_transform, index)
        self.attention.k_transform = prune_linear_layer(self.attention.k_transform, index)
        self.attention.v_transform = prune_linear_layer(self.attention.v_transform, index)
        self.attention.o_transform = prune_linear_layer(self.attention.o_transform, index, dim=1)

        # Update hyper params
        self.attention.num_heads = self.attention.num_heads - len(heads)
        self.attention.hidden_size = self.attention.head_size * self.attention.num_heads

    def forward(self, x, bias, memory=None, state=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.attention.num_heads == 0:
            if self.normalization == "before":
                return x
            else:
                return self.layer_norm(x)

        if self.training or state is None:
            y = self.attention(y, bias, memory, None)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, memory, kv)
            state["k"], state["v"] = k, v

        y = F.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)

    def forward_with_head_analysis(self, x, bias, memory=None, state=None, mode=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y, head_feature = self.attention.forward_with_head_analysis(y, bias, memory, None, mode=mode)
        else:
            kv = [state["k"], state["v"]]
            y, k, v, head_feature = self.attention.forward_with_head_analysis(y, bias, memory, kv, mode=mode)
            state["k"], state["v"] = k, v

        y = F.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y, head_feature
        else:
            return self.layer_norm(x + y), head_feature



class FFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="selective_ffn_layer"):
        super(FFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 params.filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = F.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class SelectiveTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="selective_encoder_layer", shared_layer=None):
        super(SelectiveTransformerEncoderLayer, self).__init__(name=name)

        self.params = params

        with utils.scope(name):
            if shared_layer is not None and params.shared_layer_params:
                self.self_attention = shared_layer.self_attention
                self.feed_forward = shared_layer.feed_forward
            else:
                self.self_attention = SelectiveAttentionSubLayer(params) 
                self.feed_forward = FFNSubLayer(params)

        self.pruned_heads = set()
        self.additional_params = []
        self.build_additional_params(params)

    def build_additional_params(self, params):
        with utils.scope(self.name):
            if params.input_aware_select:
                self.select_transform = Affine(params.hidden_size, params.num_heads,
                                               name="select_transform")
                self.additional_params += list(self.select_transform.parameters())
            else:
                self.kappa = nn.Parameter(torch.empty(params.num_heads))
                self.add_name(self.kappa, "kappa")
                self.additional_params.append(self.kappa)

        if params.input_aware_select:
            nn.init.xavier_uniform_(self.select_transform.weight, 2 ** -0.5)
            nn.init.constant_(self.select_transform.bias, 0.0)
        else:
            nn.init.constant_(self.kappa, 0.0)

    def load_additional_params(self):
        if self.params.input_aware_select:
            self.self_attention.attention.select_transform = self.select_transform
        else:
            self.self_attention.attention.kappa = self.kappa

    def prune_heads(self, heads):
        self.self_attention._prune_heads(heads, self.pruned_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, x, bias):
        self.load_additional_params()
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x

    def forward_with_head_analysis(self, x, bias, mode):
        # head_feature is 1-dim tensor
        x, head_feature = self.self_attention.forward_with_head_analysis(x, bias, mode=mode)
        x = self.feed_forward(x)
        return x, head_feature

class SelectiveTransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="selective_decoder_layer", shared_layer=None):
        super(SelectiveTransformerDecoderLayer, self).__init__(name=name)

        self.params = params

        with utils.scope(name):
            if shared_layer is not None and params.shared_layer_params:
                self.self_attention = shared_layer.self_attention
                self.encdec_attention = shared_layer.encdec_attention
                self.feed_forward = shared_layer.feed_forward
            else:
                self.self_attention = SelectiveAttentionSubLayer(params,
                                                                 name="self_attention")
                self.encdec_attention = SelectiveAttentionSubLayer(params,
                                                                   name="encdec_attention")
                self.feed_forward = FFNSubLayer(params)

        self.self_pruned_heads = set()
        self.encdec_pruned_heads = set()
        self.additional_params = []
        self.build_additional_params(params)

    def build_additional_params(self, params):
        with utils.scope(self.name):
            if params.input_aware_select:
                self.self_select_transform = Affine(params.hidden_size, params.num_heads,
                                               name="self_select_transform")
                self.encdec_select_transform = Affine(params.hidden_size, params.num_heads,
                                               name="encdec_select_transform")
                self.additional_params += list(self.self_select_transform.parameters())
                self.additional_params += list(self.encdec_select_transform.parameters())
            else:
                self.self_kappa = nn.Parameter(torch.empty(params.num_heads))
                self.add_name(self.self_kappa, "self_kappa")
                self.encdec_kappa = nn.Parameter(torch.empty(params.num_heads))
                self.add_name(self.encdec_kappa, "encdec_kappa")
                self.additional_params.append(self.self_kappa)
                self.additional_params.append(self.encdec_kappa)

        if params.input_aware_select:
            nn.init.xavier_uniform_(self.self_select_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.encdec_select_transform.weight, 2 ** -0.5)
            nn.init.constant_(self.self_select_transform.bias, 0.0)
            nn.init.constant_(self.encdec_select_transform.bias, 0.0)
        else:
            nn.init.constant_(self.self_kappa, 0.0)
            nn.init.constant_(self.encdec_kappa, 0.0)
 
    def load_additional_params(self):
        if self.params.input_aware_select:
            self.self_attention.attention.select_transform = self.self_select_transform
            self.encdec_attention.attention.select_transform = self.encdec_select_transform
        else:
            self.self_attention.attention.kappa = self.self_kappa
            self.encdec_attention.attention.kappa = self.encdec_kappa

    def self_prune_heads(self, heads):
        self.self_attention._prune_heads(heads, self.self_pruned_heads)
        self.self_pruned_heads = self.self_pruned_heads.union(heads)

    def encdec_prune_heads(self, heads):
        self.encdec_attention._prune_heads(heads, self.encdec_pruned_heads)
        self.encdec_pruned_heads = self.encdec_pruned_heads.union(heads)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        self.load_additional_params()
        x = self.self_attention(x, attn_bias, state=state)
        x = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x

    def forward_with_head_analysis(self, x, attn_bias, encdec_bias, memory, state=None, mode=None):
        x, decoder_feature = self.self_attention.forward_with_head_analysis(x, attn_bias, state=state, mode=mode)
        x, encdec_feature = self.encdec_attention.forward_with_head_analysis(x, encdec_bias, memory, mode=mode)
        x = self.feed_forward(x)
        return x, decoder_feature, encdec_feature


class SelectiveTransformerEncoder(modules.Module):

    def __init__(self, params, name="selective_encoder"):
        super(SelectiveTransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            if params.shared_layer_params:
                shared_layer = SelectiveTransformerEncoderLayer(params)
                self.layers = nn.ModuleList([
                    SelectiveTransformerEncoderLayer(params, name="layer_%d" % i,
                                                     shared_layer=shared_layer)
                    for i in range(params.num_encoder_layers)])
            else:
                self.layers = nn.ModuleList([
                    SelectiveTransformerEncoderLayer(params, name="layer_%d" % i)
                    for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
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

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x

    def forward_with_head_analysis(self, x, bias, mode):
        head_feature = []
        for layer in self.layers:
            x, layer_head_feature = layer.forward_with_head_analysis(x, bias, mode)
            # layer_head_feature is list for confidence and intermediate variable for grad_sensitivity
            if mode == "confidence":
                head_feature.append(layer_head_feature.tolist())
            elif mode == "grad_sensitivity":
                head_feature.append(layer_head_feature)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x, head_feature


class SelectiveTransformerDecoder(modules.Module):

    def __init__(self, params, name="selective_decoder"):
        super(SelectiveTransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            if params.shared_layer_params:
                shared_layer = SelectiveTransformerDecoderLayer(params)
                self.layers = nn.ModuleList([
                    SelectiveTransformerDecoderLayer(params, name="layer_%d" % i,
                                                     shared_layer=shared_layer)
                    for i in range(params.num_encoder_layers)])
            else:
                self.layers = nn.ModuleList([
                    SelectiveTransformerDecoderLayer(params, name="layer_%d" % i)
                    for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
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

    @property
    def decoder_additional_params(self):
        decoder_additional_params = []
        for layer in self.layers:
            decoder_additional_params += layer.self_attention.additional_params
        return decoder_additional_params

    @property
    def encdec_additional_params(self):
        encdec_additional_params = []
        for layer in self.layers:
            encdec_additional_params += layer.encdec_attention.additional_params
        return encdec_additional_params

    def forward_with_head_analysis(self, x, attn_bias, encdec_bias, memory, state=None, mode=None):
        decoder_head_feature = []
        encdec_head_feature = []
        for i, layer in enumerate(self.layers):
            if state is not None:
                x, layer_decoder_feature, layer_encdec_feature = layer.forward_with_head_analysis(x, attn_bias, encdec_bias, memory, state["decoder"]["layer_%d" % i], mode=mode)
            else:
                x, layer_decoder_feature, layer_encdec_feature = layer.forward_with_head_analysis(x, attn_bias, encdec_bias, memory, None, mode=mode)

            # layer_head_feature is list for confidence and intermediate variable for grad_sensitivity
            if mode == "confidence":
                decoder_head_feature.append(layer_decoder_feature.tolist())
                encdec_head_feature.append(layer_encdec_feature.tolist())
            elif mode == "grad_sensitivity":
                decoder_head_feature.append(layer_decoder_feature)
                encdec_head_feature.append(layer_encdec_feature)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x, decoder_head_feature, encdec_head_feature


class SelectiveTransformer(modules.Module):

    def __init__(self, params, name="selective_transformer"):
        super().__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = SelectiveTransformerEncoder(params)
            self.decoder = SelectiveTransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers

        self.env_name = params.env_name

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

    def prune_heads(self, heads_to_prune):
        encoder_heads_to_prune = heads_to_prune["encoder"]
        decoder_heads_to_prune = heads_to_prune["decoder"]
        encdec_heads_to_prune = heads_to_prune["encdec"]

        self.encoder._prune_heads(encoder_heads_to_prune)
        self.decoder._prune_heads(decoder_heads_to_prune, encdec_heads_to_prune)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = F.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = F.dropout(self.encoding(inputs), self.dropout, self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = F.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = F.dropout(self.encoding(decoder_input), self.dropout, self.training)

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

    def encode_with_head_analysis(self, features, state, mode):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = F.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = F.dropout(self.encoding(inputs), self.dropout, self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output, head_feature = self.encoder.forward_with_head_analysis(inputs, enc_attn_bias, mode)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state, head_feature

    def decode_with_head_analysis(self, features, state, mode):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = F.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = F.dropout(self.encoding(decoder_input), self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        decoder_output, decoder_head_feature, encdec_head_feature = self.decoder.forward_with_head_analysis(
                        decoder_input, 
                        dec_attn_bias,
                        enc_attn_bias, 
                        encoder_output, 
                        state,
                        mode)

        state["dec_attn_bias"] = dec_attn_bias

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)


        return logits, state, decoder_head_feature, encdec_head_feature


    def compute_confidence(self, features, labels):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state, encoder_confidence = self.encode_with_head_analysis(features, state, mode="confidence")
        _, _, decoder_confidence, encdec_confidence = self.decode_with_head_analysis(features, state, mode="confidence")

        return encoder_confidence, decoder_confidence, encdec_confidence

    def compute_grad_sensitivity(self, features, labels):
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0], labels.device)
        state, encoder_variable = self.encode_with_head_analysis(features, state, mode="grad_sensitivity")
        logits, _, decoder_variable, encdec_variable = self.decode_with_head_analysis(features, state, 
                                                                                      mode="grad_sensitivity")
        loss = self.criterion(logits, labels)
        mask = mask.to(torch.float32)
        loss = (torch.sum(loss * mask) / torch.sum(mask)).to(logits)
        loss.backward()
        encoder_var_grad = [v.grad for v in encoder_variable]
        decoder_var_grad = [v.grad for v in decoder_variable]
        encdec_var_grad = [v.grad for v in encdec_variable]

        encoder_scores = []
        for i, (var, grad) in enumerate(list(zip(encoder_variable, encoder_var_grad))):
            if var.dim() == 3:
                num_heads= self.encoder.layers[i].self_attention.attention.num_heads
                head_size= self.encoder.layers[i].self_attention.attention.head_size
                var = var.reshape(var.size(0), var.size(1), num_heads, head_size)
                grad = grad.reshape(grad.size(0), grad.size(1), num_heads, head_size)
                score = torch.einsum("blhd,blhd->bhl", var, grad)
            else:
                score = torch.einsum("bhld,bhld->bhl", var, grad)
            score *= features["source_mask"].unsqueeze(1).to(score)
            score = score.abs().sum(dim=(0, 2)).tolist()
            encoder_scores.append(score)

        decoder_scores = []
        for i, (var, grad) in enumerate(list(zip(decoder_variable, decoder_var_grad))):
            if var.dim() == 3:
                num_heads= self.decoder.layers[i].self_attention.attention.num_heads
                head_size= self.decoder.layers[i].self_attention.attention.head_size
                var = var.reshape(var.size(0), var.size(1), num_heads, head_size)
                grad = grad.reshape(grad.size(0), grad.size(1), num_heads, head_size)
                score = torch.einsum("blhd,blhd->bhl", var, grad)
            else:
                score = torch.einsum("bhld,bhld->bhl", var, grad)
            score *= features["target_mask"].unsqueeze(1).to(score)
            score = score.abs().sum(dim=(0, 2)).tolist()
            decoder_scores.append(score)

        encdec_scores = []
        for i, (var, grad) in enumerate(list(zip(encdec_variable, encdec_var_grad))):
            if var.dim() == 3:
                num_heads= self.decoder.layers[i].encdec_attention.attention.num_heads
                head_size= self.decoder.layers[i].encdec_attention.attention.head_size
                var = var.reshape(var.size(0), var.size(1), num_heads, head_size)
                grad = grad.reshape(grad.size(0), grad.size(1), num_heads, head_size)
                score = torch.einsum("blhd,blhd->bhl", var, grad)
            else:
                score = torch.einsum("bhld,bhld->bhl", var, grad)
            score *= features["target_mask"].unsqueeze(1).to(score)
            score = score.abs().sum(dim=(0, 2)).tolist()
            encdec_scores.append(score)

        return encdec_scores, decoder_scores, encdec_scores

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
                    "k": torch.zeros([batch_size, 0, layer.self_attention.attention.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, layer.self_attention.attention.hidden_size],
                                     device=device)
                } for i, layer in enumerate(self.decoder.layers)
            }
        }

        return state

    def compute_head_selection_weight(self, var, name):
        return torch.sigmoid(var)


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
            shared_layer_params=False,
            # select settings
            input_aware_select=False,
            select_weight_function="sigmoid",
            select_method="hard",
            select_number=0,
            env_name="train",
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
        params = SelectiveTransformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = SelectiveTransformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = SelectiveTransformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def test_params():
        params = SelectiveTransformer.base_params()
        params.hidden_size = 64
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
            return SelectiveTransformer.base_params()
        elif name == "base_v2":
            return SelectiveTransformer.base_params_v2()
        elif name == "big":
            return SelectiveTransformer.big_params()
        elif name == "big_v2":
            return SelectiveTransformer.big_params_v2()
        elif name == "test":
            return SelectiveTransformer.test_params()
        else:
            return SelectiveTransformer.base_params()
