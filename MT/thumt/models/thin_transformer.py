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


class ThinAttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(ThinAttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.residual = params.attention_residual

        attention_hidden_size = params.num_heads * params.head_size

        with utils.scope(name):
            self.attention = modules.ThinMultiHeadAttention(params.hidden_size, 
                                                            params.head_size,
                                                            params.num_heads, 
                                                            params.attention_dropout)
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = modules.LayerNorm(attention_hidden_size)
            if params.residual_transform:
                self.residual_transform = modules.Affine(params.hidden_size, attention_hidden_size,
                                                         name="residual_transform")
                self.reset_parameters()
            else:
                if params.hidden_size % attention_hidden_size == 0:
                    self.residual_transform = lambda x: x.reshape(*x.size()[:2], attention_hidden_size, -1).sum(-1)
                else:
                    raise ValueError("size not match for unparameterized residual")

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

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.residual:
            if self.normalization == "before":
                return self.residual_transform(x) + y
            else:
                return self.layer_norm(self.residual_transform(x) + y)
        else:
            if self.normalization == "before":
                return y
            else:
                return self.layer_norm(y)

    def reset_parameters(self):
        nn.init.eye_(self.residual_transform.weight)
        nn.init.constant_(self.residual_transform.bias, 0.0)


class ThinFFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(ThinFFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization
        self.thin_output = params.thin_ffn_output

        ffn_hidden_size = params.num_heads * params.head_size
        ffn_filter_size = ffn_hidden_size * 4
        if self.thin_output:
            ffn_output_size = ffn_hidden_size
        else:
            ffn_output_size = params.hidden_size

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(ffn_hidden_size,
                                                 ffn_filter_size,
                                                 ffn_output_size,
                                                 dropout=params.relu_dropout)
            if self.thin_output or self.normalization == "before":
                self.layer_norm = modules.LayerNorm(ffn_hidden_size)
            else:
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            if params.outer_transform:
                self.outer_transform = modules.Affine(ffn_hidden_size, params.hidden_size,
                                                       name="outer_transform")
                self.reset_parameters()
            else:
                if params.hidden_size % ffn_hidden_size == 0:
                    fold_num = params.hidden_size // ffn_hidden_size
                    self.outer_transform = lambda x: \
                        x.unsqueeze(-1).expand(-1, -1, -1, fold_num).reshape(*x.size()[:2], params.hidden_size)
                else:
                    raise ValueError("size not match for unparameterized residual")

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.thin_output:
            if self.normalization == "before":
                return self.outer_transform(x + y)
            else:
                return self.outer_transform(self.layer_norm(x + y))
        else:
            if self.normalization == "before":
                return y
            else:
                return self.layer_norm(y)

    def reset_parameters(self):
        nn.init.eye_(self.outer_transform.weight)
        nn.init.constant_(self.outer_transform.bias, 0.0)


class ThinTransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(ThinTransformerEncoderLayer, self).__init__(name=name)

        self.skip_residual = params.skip_residual

        with utils.scope(name):
            self.self_attention = ThinAttentionSubLayer(params)
            self.feed_forward = ThinFFNSubLayer(params)

            if params.skip_residual:
                self.layer_norm = modules.LayerNorm(params.hidden_size)


    def forward(self, x, bias):
        y = self.self_attention(x, bias)
        y = self.feed_forward(y)

        if self.skip_residual:
            return self.layer_norm(x + y)
        else:
            return y


class ThinTransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(ThinTransformerDecoderLayer, self).__init__(name=name)

        attention_hidden_size = params.num_heads * params.head_size
        self.skip_residual = params.skip_residual

        with utils.scope(name):
            self.self_attention = ThinAttentionSubLayer(params,
                                                        name="self_attention")
            if params.output_transform:
                self.output_transform = modules.Affine(attention_hidden_size, params.hidden_size,
                                                       name="output_transform")
                self.reset_parameters()
            else:
                self.output_transform = lambda x: x
            self.encdec_attention = ThinAttentionSubLayer(params,
                                                          name="encdec_attention")
            self.feed_forward = ThinFFNSubLayer(params)

            if params.skip_residual:
                self.layer_norm = modules.LayerNorm(params.hidden_size)


    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        y = self.self_attention(x, attn_bias, state=state)
        y = self.output_transform(y)
        y = self.encdec_attention(y, encdec_bias, memory)
        y = self.feed_forward(y)

        if self.skip_residual:
            return self.layer_norm(x + y)
        else:
            return y

    def reset_parameters(self):
        nn.init.eye_(self.output_transform.weight)
        nn.init.constant_(self.output_transform.bias, 0.0)


class ThinTransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(ThinTransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                ThinTransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class ThinTransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(ThinTransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                ThinTransformerDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

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


class ThinTransformer(modules.Module):

    def __init__(self, params, name="thin_transformer"):
        super(ThinTransformer, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = ThinTransformerEncoder(params)
            self.decoder = ThinTransformerDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
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
            residual_transform=True,
            outer_transform=True,
            output_transform=True,
            skip_residual=False,
            attention_residual=True,
            thin_ffn_output=True, # will disable ffn residual
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
        params = ThinTransformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = ThinTransformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = ThinTransformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def test_params():
        params = ThinTransformer.base_params()
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
            return ThinTransformer.base_params()
        elif name == "base_v2":
            return ThinTransformer.base_params_v2()
        elif name == "big":
            return ThinTransformer.big_params()
        elif name == "big_v2":
            return ThinTransformer.big_params_v2()
        elif name == "test":
            return ThinTransformer.test_params()
        else:
            return ThinTransformer.base_params()
