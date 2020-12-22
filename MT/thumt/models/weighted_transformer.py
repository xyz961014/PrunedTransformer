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

    def prune_heads(self, heads, pruned_heads):
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


class WeightedAttentionSubLayer(modules.Module):

    def __init__(self, params, name="weighted_attention"):
        super(WeightedAttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.enable_kappa = params.enable_kappa
        self.enable_alpha = params.enable_alpha
        self.pruned_heads = set()

        with utils.scope(name):
            self.attention = modules.WeightedMultiHeadAttention(params.hidden_size, 
                                                                params.num_heads, 
                                                                params.attention_dropout,
                                                                enable_kappa=params.enable_kappa,
                                                                enable_alpha=params.enable_alpha,
                                                                expand_kappa_norm=params.expand_kappa_norm)
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
        if self.enable_kappa:
            self.attention.kappa = prune_vector(self.attention.kappa, heads, 
                                                self.attention.num_heads, pruned_heads)

        # Update hyper params
        self.attention.num_heads = self.attention.num_heads - len(heads)
        self.attention.hidden_size = self.attention.head_size * self.attention.num_heads


    def forward(self, x, bias, memory=None, state=None):
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

        y = F.dropout(y, self.dropout, self.training)

        if self.enable_alpha:
            if self.normalization == "before":
                return x.unsqueeze(1) + y
            else:
                return self.layer_norm(x.unsqueeze(1) + y)
        else:
            if self.normalization == "before":
                return x + y
            else:
                return self.layer_norm(x + y)


class WeightedFFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="weighted_ffn_layer"):
        super(WeightedFFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.enable_alpha = params.enable_alpha

        with utils.scope(name):
            self.ffn_layer = modules.WeightedFeedForward(params.hidden_size,
                                                         params.filter_size,
                                                         params.num_heads,
                                                         dropout=params.relu_dropout,
                                                         enable_alpha=params.enable_alpha,
                                                         expand_alpha_norm=params.expand_alpha_norm)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

        self.additional_params = self.ffn_layer.additional_params

    def _prune_heads(self, heads, pruned_heads):
        if len(heads) == 0:
            return
        if self.enable_alpha:
            self.ffn_layer.alpha = prune_vector(self.ffn_layer.alpha, heads, 
                                                self.ffn_layer.num_heads, pruned_heads)
        # Update hyper params
        self.ffn_layer.num_heads = self.ffn_layer.num_heads - len(heads)


    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = F.dropout(y, self.dropout, self.training)

        if self.enable_alpha:
            if self.normalization == "before":
                return x.sum(dim=1) + y
            else:
                return self.layer_norm(x.sum(dim=1) + y)
        else:
            if self.normalization == "before":
                return x + y
            else:
                return self.layer_norm(x + y)


class WeightedTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="weighted_encoder_layer"):
        super(WeightedTransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = WeightedAttentionSubLayer(params)
            self.feed_forward = WeightedFFNSubLayer(params)

        self.pruned_heads = set()
        self.additional_params = self.self_attention.additional_params + self.feed_forward.additional_params

    def prune_heads(self, heads):
        self.self_attention._prune_heads(heads, self.pruned_heads)
        self.feed_forward._prune_heads(heads, self.pruned_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, x, bias):
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class WeightedTransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="weighted_decoder_layer"):
        super(WeightedTransformerDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = WeightedAttentionSubLayer(params,
                                                              name="encdec_attention")
            self.feed_forward = WeightedFFNSubLayer(params)

        self.additional_params = self.encdec_attention.additional_params + self.feed_forward.additional_params

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x = self.self_attention(x, attn_bias, state=state)
        x = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x


class WeightedTransformerEncoder(modules.Module):

    def __init__(self, params, name="weighted_encoder"):
        super(WeightedTransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                WeightedTransformerEncoderLayer(params, name="layer_%d" % i)
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
            self.layers[layer].prune_heads(heads)

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class WeightedTransformerDecoder(modules.Module):

    def __init__(self, params, name="weighted_decoder"):
        super(WeightedTransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                WeightedTransformerDecoderLayer(params, name="layer_%d" % i)
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
            self.layers[layer].prune_heads(heads)

        for layer, heads in encdec_heads_to_prune.items():
            self.layers[layer].prune_heads(heads)

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


class WeightedTransformer(modules.Module):

    def __init__(self, params, name="weighted_transformer"):
        super().__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = WeightedTransformerEncoder(params)
            self.decoder = WeightedTransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
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
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device)
                } for i in range(self.num_decoder_layers)
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
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            enable_alpha=True,
            enable_kappa=True,
            expand_alpha_norm=False,
            expand_kappa_norm=False,
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
        params = WeightedTransformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = WeightedTransformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = WeightedTransformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def test_params():
        params = WeightedTransformer.base_params()
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
            return WeightedTransformer.base_params()
        elif name == "base_v2":
            return WeightedTransformer.base_params_v2()
        elif name == "big":
            return WeightedTransformer.big_params()
        elif name == "big_v2":
            return WeightedTransformer.big_params_v2()
        elif name == "test":
            return WeightedTransformer.test_params()
        else:
            return WeightedTransformer.base_params()
