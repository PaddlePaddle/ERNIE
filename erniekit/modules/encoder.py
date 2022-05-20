# -*- coding: utf-8 -*
"""
常见的文本编码器（encoder）
"""
import paddle
import paddle.nn as nn


class BoWEncoder(nn.Layer):
    """BOW
    """
    def __init__(self, emb_dim):
        """emb_dim
        """
        nn.Layer.__init__(self)
        self._emb_dim = emb_dim

    def get_input_dim(self):
        """input_dim
        """
        return self._emb_dim

    def get_output_dim(self):
        """output_dim
        """
        return self._emb_dim

    def forward(self, inputs):
        """inputs
        """
        # Shape: (batch_size, embedding_dim)
        summed = paddle.sum(inputs, axis=1)
        return summed


class CNNEncoder(nn.Layer):
    """CNN
    """
    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        """
        """
        nn.Layer.__init__(self)

        self._emb_dim = emb_dim
        self._num_filter = num_filter
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self.convs = nn.LayerList([
            nn.Conv2D(
                in_channels=1,
                out_channels=self._num_filter,
                kernel_size=(i, self._emb_dim),
                **kwargs) for i in self._ngram_filter_sizes
        ])

        maxpool_output_dim = self._num_filter * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def forward(self, inputs):
        """inputs
        """
        # Shape: (batch_size, 1, num_tokens, emb_dim) = (N, C, H, W)
        inputs = inputs.unsqueeze(1)

        # If output_dim is None, result shape of (batch_size, len(ngram_filter_sizes) * num_filter));
        # else, result shape of (batch_size, output_dim).
        convs_out = [
            self._activation(conv(inputs)).squeeze(3) for conv in self.convs
        ]
        maxpool_out = [
            nn.functional.adaptive_max_pool1d(t, output_size=1).squeeze(2) for t in convs_out
        ]
        result = paddle.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)

        return result

    def get_input_dim(self):
        """input_dim
        """
        return self._emb_dim

    def get_output_dim(self):
        """output_dim
        """
        return self._output_dim


class LSTMEncoder(nn.Layer):
    """LSTM
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        """
        """
        nn.Layer.__init__(self)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs)

    def get_input_dim(self):
        """input_dim
        """
        return self._input_size

    def get_output_dim(self):
        """output_dim
        """
        if self._direction == "bidirect":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        """
        :params inputs
        :params sequence_length
        """
        encoded_text, (last_hidden, last_cell) = self.lstm_layer(inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If lstm is not bidirection, then output is the hidden state of the last time step
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If lstm is bidirection, then output is concatenation of the forward and backward hidden state
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size * 2)`.
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat((last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class GRUEncoder(nn.Layer):
    """GRU
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        """
        """
        nn.Layer.__init__(self)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                direction=direction,
                                dropout=dropout,
                                **kwargs)

    def get_input_dim(self):
        """input_dim
        """
        return self._input_size

    def get_output_dim(self):
        """output_dim
        """
        if self._direction == "bidirect":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        """
        :params inputs
        :params sequence_length
        """
        encoded_text, last_hidden = self.gru_layer(inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # The output shape is `(batch_size, hidden_size * 2)` if use bidirectional GRU,
            # otherwise the output shape is `(batch_size, hidden_size * 2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output
