# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules


class FitAttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(FitAttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.FitMultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.FitLayerNorm(params.hidden_size)

    def prune_dim(self, index):
        self.attention.prune_dim(index)
        self.layer_norm.prune_dim(index)

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

        y = utils.dim_dropout(y, self.dropout, self.training, dim=-1)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class FitFFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(FitFFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FitFeedForward(params.hidden_size,
                                                 params.filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.FitLayerNorm(params.hidden_size)

    def prune_dim(self, index):
        self.ffn_layer.prune_dim(index)
        self.layer_norm.prune_dim(index)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = utils.dim_dropout(y, self.dropout, self.training, dim=-1)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class FitTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(FitTransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = FitAttentionSubLayer(params)
            self.feed_forward = FitFFNSubLayer(params)

    def prune_dim(self, index):
        self.self_attention.prune_dim(index)
        self.feed_forward.prune_dim(index)

    def forward(self, x, bias):
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class FitTransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(FitTransformerDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = FitAttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = FitAttentionSubLayer(params,
                                                    name="encdec_attention")
            self.feed_forward = FitFFNSubLayer(params)

    def prune_dim(self, index):
        self.self_attention.prune_dim(index)
        self.encdec_attention.prune_dim(index)
        self.feed_forward.prune_dim(index)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x = self.self_attention(x, attn_bias, state=state)
        x = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x


class FitTransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(FitTransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                FitTransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.FitLayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def prune_dim(self, index):
        for layer in self.layers:
            layer.prune_dim(index)
        if self.normalization == "before":
            self.layer_norm.prune_dim(index)

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class FitTransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(FitTransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                FitTransformerDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.FitLayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def prune_dim(self, index):
        for layer in self.layers:
            layer.prune_dim(index)
        if self.normalization == "before":
            self.layer_norm.prune_dim(index)

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


class FitTransformer(modules.Module):

    def __init__(self, params, name="fit_transformer"):
        super(FitTransformer, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = FitTransformerEncoder(params)
            self.decoder = FitTransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.attn_hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
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

    def prune_embedding_and_weights(self, index):
        index_len = index.size(0)
        index = index.to(self.bias.device)
        svoc_size = len(self.params.vocabulary["source"])
        tvoc_size = len(self.params.vocabulary["target"])

        if not self.params.shared_embedding_and_softmax_weights:
            w = self.softmax_weights.index_select(1, index).clone().detach()
            with utils.scope(self.name):
                self.softmax_weights = torch.nn.Parameter(
                    torch.empty([tvoc_size, index_len]))
                self.add_name(self.softmax_weights, "softmax_weights")
            self.softmax_weights.requires_grad = False
            self.softmax_weights.copy_(w.contiguous())
            self.softmax_weights.requires_grad = True

        if not self.params.shared_source_target_embedding:
            src_e = self.source_embedding.index_select(1, index).clone().detach()
            trg_e = self.target_embedding.index_select(1, index).clone().detach()
            with utils.scope(self.name):
                self.source_embedding = torch.nn.Parameter(
                    torch.empty([svoc_size, index_len]))
                self.target_embedding = torch.nn.Parameter(
                    torch.empty([tvoc_size, index_len]))
                self.add_name(self.source_embedding, "source_embedding")
                self.add_name(self.target_embedding, "target_embedding")
            self.source_embedding.requires_grad = False
            self.target_embedding.requires_grad = False
            self.source_embedding.copy_(src_e.contiguous())
            self.target_embedding.copy_(trg_e.contiguous())
            self.source_embedding.requires_grad = True
            self.target_embedding.requires_grad = True
        else:
            e = self.weights.index_select(1, index).clone().detach()
            with utils.scope(self.name):
                self.weights = torch.nn.Parameter(
                    torch.empty([svoc_size, index_len]))
                self.add_name(self.weights, "weights")
            self.weights.requires_grad = False
            self.weights.copy_(e.contiguous())
            self.weights.requires_grad = True

        b = self.bias.index_select(0, index).clone().detach()
        with utils.scope(self.name):
            self.bias = torch.nn.Parameter(torch.zeros([index_len]))
            self.add_name(self.bias, "bias")
        self.bias.requires_grad = False
        self.bias.copy_(b.contiguous())
        self.bias.requires_grad = True


    def prune_dim(self, index=None, p=0.1):
        """
            index: index of dimension to keep
            p: if index is not given, sample it based on p
        """
        if index is None:
            # if index is not given, prune random dims based on probablity p
            index_len = round((1 - p) * self.params.hidden_size)
            if index_len:
                index = torch.ones(self.params.hidden_size).multinomial(index_len)
                index = index.sort()[0]
        else:
            index_len = index.size(0)

        if index_len:
            self.encoder.prune_dim(index=index)
            self.decoder.prune_dim(index=index)
            self.prune_embedding_and_weights(index=index)

        self.params.hidden_size = index_len
        self.hidden_size = index_len

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = utils.dim_dropout(self.encoding(inputs), self.dropout, self.training, dim=-1)

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
        decoder_input = utils.dim_dropout(self.encoding(decoder_input), self.dropout, self.training, dim=-1)

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
                    "k": torch.zeros([batch_size, 0, self.attn_hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.attn_hidden_size],
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
        params = FitTransformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = FitTransformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = FitTransformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def test_params():
        params = FitTransformer.base_params()
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
            return FitTransformer.base_params()
        elif name == "base_v2":
            return FitTransformer.base_params_v2()
        elif name == "big":
            return FitTransformer.big_params()
        elif name == "big_v2":
            return FitTransformer.big_params_v2()
        elif name == "test":
            return FitTransformer.test_params()
        else:
            return FitTransformer.base_params()
