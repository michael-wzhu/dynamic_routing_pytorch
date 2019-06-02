# -*- coding: utf-8 -*-
"""
@File: cnn_encoder.py
@Copyright: 2019 Michael Zhu
@License：the Apache License, Version 2.0
@Author：Michael Zhu
@version：
@Date：
@Desc: 
"""

from typing import Optional, Tuple

import torch
from torch import nn
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn import Activation
from overrides import overrides
from torch.nn import Conv1d, Linear


@Seq2SeqEncoder.register("rnn_encoder")
class RnnEncoder(Seq2SeqEncoder):
    """
    A ``RnnEncoder`` is a rnn layer.  As a
    :class:`Seq2SeqEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, num_tokens, output_dim)``.

    Parameters
    ----------
    input_dim : ``int``
        input dimension
    output_dim: ``int``
        output dimension, which should be divided by 2 if bidirectional == true
    rnn_name" ``str``
        name of the rnn networks
    bidirectional: ``bool``, default=``True``
        whether the rnn is bidirectional
    dropout: ``float``, default=``None``
        dropout rate
    normalizer: ``str``, default = ``None``
        name of the normalization we use
    affine_for_normalizer: bool = False
        whether affine is used in the normalization
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 rnn_name: str = "lstm",
                 bidirectional: bool = True,
                 dropout: float = None,
                 normalizer: str = None,
                 affine_for_normalizer: bool = False) -> None:
        super(RnnEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_name = rnn_name
        self.bidirectional = bidirectional

        if bidirectional:
            assert output_dim % 2 == 0
            hidden_size = output_dim // 2
        else:
            hidden_size = output_dim

        if rnn_name == "lstm":
            self._rnn = torch.nn.LSTM(
                input_dim,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional

            )
        elif rnn_name == "gru":
            self._rnn = torch.nn.GRU(
                input_dim,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            self._rnn = torch.nn.RNN(
                input_dim,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._norm = None
        if normalizer == "batch_norm":
            self._norm = nn.BatchNorm1d(
                kernel_size,
                affine=affine_for_normalizer
            )
        elif normalizer == "layer_norm":
            self._norm = nn.LayerNorm(
                kernel_size,
                elementwise_affine=affine_for_normalizer
            )

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        if mask is not None:
            input_tensors = input_tensors * mask.unsqueeze(-1).float()

        encoded_output, _ = self._rnn(input_tensors)

        if self._dropout:
            encoded_output = self._dropout(encoded_output)

        if self._norm:
            encoded_output = self._norm(encoded_output)

        return encoded_output


if __name__ == "__main__":

    batch_size = 3
    num_tokens = 4
    input_dim = 10
    mask = [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]
    mask = torch.tensor(mask)

    input_tensors_ = torch.randn([batch_size, num_tokens, input_dim])

    output_dim = 32
    bidirectional = True
    rnn_ = RnnEncoder(
        input_dim,
        output_dim,
        rnn_name="rnn",  # pylint: disable=bad-whitespace
        bidirectional=False,
        dropout=0.3,
        normalizer="layer_norm",
        affine_for_normalizer=False
    )

    encoded_output = rnn_(input_tensors_, mask)
    print(encoded_output.size())
