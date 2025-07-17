import numpy as np
import torch
import torch.nn as nn


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)
    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False,
                 use_spectral_norm=False):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=not use_spectral_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out