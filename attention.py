# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from common.tensor_utils import weighted_sum, last_dim_softmax, masked_softmax

class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, input_size_encoder, input_size_decoder, num_labels=1, biaffine=True, **kwargs):
        '''

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(BiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_d = nn.Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = nn.Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = nn.Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = nn.Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.constant_(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        '''

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        '''
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            mask_d = mask_d.float()
            mask_e = mask_e.float()
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        if self.num_labels == 1:
            # [batch, length_decoder, length_encoder]
            return output.squeeze(1)
        else:
            return output

def cosine_attn(mat_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    a_norm = mat_a / (mat_a.norm(p=2, dim=-1, keepdim=True) + 1e-13)
    b_norm = vec_b / (vec_b.norm(p=2, dim=-1, keepdim=True) + 1e-13)

    return torch.matmul(a_norm.float(), b_norm.transpose(-1, -2).float())


class BilinearAttention(nn.Module):

    def __init__(self, hidden_dim) -> None:
        super(BilinearAttention, self).__init__()
        self._weight_matrix = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self._bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix: torch.Tensor, mask: torch.Tensor, adj) -> torch.Tensor:
        '''

        :param matrix: (batch_size, seq_len, hidden_dim)
        :return:
        '''
        intermediate = matrix.matmul(self._weight_matrix)

        similarities = intermediate.bmm(matrix.transpose(1, 2)) #+ self._bias
        similarities = similarities * adj
        attn = masked_softmax(similarities, mask)
        output = attn.matmul(matrix)

        return output



class MultiHeadSelfAttention(torch.nn.Module):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")


        self._combined_projection = nn.Linear(input_dim, 2 * attention_dim + values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = nn.Linear(values_dim, self._output_dim)
        self._attention_dropout = nn.Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim


    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None,
                adj = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new(batch_size, timesteps).fill_(1.0)

        # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
        combined_projection = self._combined_projection(inputs)
        # split by attention dim - if values_dim > attention_dim, we will get more
        # than 3 elements returned. All of the rest are the values vector, so we
        # just concatenate them back together again below.
        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()
        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, num_heads, int(self._attention_dim/num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, int(self._attention_dim/num_heads))

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(queries_per_head, keys_per_head.transpose(1, 2)) / self._scale

        if adj is not None:
            adj = adj.unsqueeze(1).expand(batch_size, num_heads, timesteps, timesteps).reshape(batch_size*num_heads, timesteps, timesteps)
            scaled_similarities = scaled_similarities * adj

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        attention = last_dim_softmax(scaled_similarities, mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps))
        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, timesteps, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs


class MultiHeadSoftmax(torch.nn.Module):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadSoftmax, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")


        self._combined_projection = nn.Linear(input_dim, 2 * attention_dim + values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = nn.Linear(values_dim, self._output_dim)
        self._attention_dropout = nn.Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim


    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new(batch_size, timesteps).fill_(1.0)

        # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
        combined_projection = self._combined_projection(inputs)
        # split by attention dim - if values_dim > attention_dim, we will get more
        # than 3 elements returned. All of the rest are the values vector, so we
        # just concatenate them back together again below.
        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        # values = torch.cat(values, -1).contiguous()
        # # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        # values_per_head = values.view(batch_size, timesteps, num_heads, int(self._values_dim/num_heads))
        # values_per_head = values_per_head.transpose(1, 2).contiguous()
        # values_per_head = values_per_head.view(batch_size * num_heads, timesteps, int(self._values_dim/num_heads))

        dim_per_head = int(self._attention_dim/num_heads)

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, num_heads, dim_per_head)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, timesteps, dim_per_head)

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(batch_size, timesteps, num_heads, dim_per_head)
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, timesteps, dim_per_head)

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(queries_per_head, keys_per_head.transpose(1, 2)) / self._scale

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        attention = last_dim_softmax(scaled_similarities, mask.repeat(1, num_heads).view(batch_size * num_heads, timesteps))
        attention = self._attention_dropout(attention)


        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, timesteps)
        attention = attention.view(batch_size, num_heads, timesteps, timesteps)

        return attention