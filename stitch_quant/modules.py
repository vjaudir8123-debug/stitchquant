"""
Drop-in nn.Module replacements that use StitchedTensors for weights.

These modules dequantize + stitch on every forward pass, keeping only
the compressed representation in memory. The dequantized FP16 tensor
is an ephemeral temporary that lives only for the duration of the matmul/conv.

Supports:
- StitchedLinear (standard nn.Linear)
- StitchedConv1D (Hugging Face Transformer Conv1D e.g. GPT-2 attention & MLP projections)
- StitchedConv2d (Standard 2D convolution)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stitched_tensor import StitchedTensor


class StitchedLinear(nn.Module):
    """Drop-in replacement for nn.Linear using StitchQuant weights."""

    def __init__(self, stitched_weight: StitchedTensor, bias: torch.Tensor | None):
        super().__init__()
        self.stitched_weight = stitched_weight
        self._use_cache = False
        if bias is not None:
            self.register_buffer("bias", bias.clone().half())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.stitched_weight.dequantize(use_cache=self._use_cache)
        orig_dtype = x.dtype
        # Matrix multiply in FP16 precision
        bias_fp16 = self.bias if self.bias is not None else None
        out = F.linear(x.half(), w, bias_fp16)
        return out.to(orig_dtype)

    def set_cache_mode(self, enabled: bool = True) -> None:
        """Enable or disable persistent weight caching."""
        self._use_cache = enabled
        if not enabled:
            self.stitched_weight.clear_cache()

    def clear_cache(self) -> None:
        """Clear any cached dequantized weights."""
        self.stitched_weight.clear_cache()

    @property
    def weight_memory_bytes(self) -> int:
        b = self.stitched_weight.memory_bytes
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return b

    def extra_repr(self) -> str:
        s = self.stitched_weight.qdata.shape
        return (
            f"in={s[1]}, out={s[0]}, "
            f"bits={self.stitched_weight.qdata.bit_width}, "
            f"overlay={self.stitched_weight.overlay_fraction:.1%}, "
            f"compression={self.stitched_weight.compression_ratio:.2f}x"
        )


class StitchedConv1D(nn.Module):
    """Drop-in replacement for HuggingFace Conv1D (used in GPT-2 and other LLMs).
    
    In Hugging Face Conv1D, the weight matrix has shape [in_features, out_features]
    and computes: x @ w + b
    """

    def __init__(self, stitched_weight: StitchedTensor, bias: torch.Tensor | None):
        super().__init__()
        self.stitched_weight = stitched_weight
        self.nf = stitched_weight.qdata.shape[1]
        self._use_cache = False
        if bias is not None:
            self.register_buffer("bias", bias.clone().half())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.stitched_weight.dequantize(use_cache=self._use_cache)
        orig_dtype = x.dtype
        size_out = x.size()[:-1] + (self.nf,)
        x_flat = x.view(-1, x.size(-1)).half()
        if self.bias is not None:
            out = torch.addmm(self.bias, x_flat, w)
        else:
            out = torch.mm(x_flat, w)
        return out.view(size_out).to(orig_dtype)

    def set_cache_mode(self, enabled: bool = True) -> None:
        self._use_cache = enabled
        if not enabled:
            self.stitched_weight.clear_cache()

    def clear_cache(self) -> None:
        self.stitched_weight.clear_cache()

    @property
    def weight_memory_bytes(self) -> int:
        b = self.stitched_weight.memory_bytes
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return b

    def extra_repr(self) -> str:
        s = self.stitched_weight.qdata.shape
        return (
            f"in={s[0]}, out={s[1]}, "
            f"bits={self.stitched_weight.qdata.bit_width}, "
            f"overlay={self.stitched_weight.overlay_fraction:.1%}, "
            f"compression={self.stitched_weight.compression_ratio:.2f}x"
        )


class StitchedConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d using StitchQuant weights."""

    def __init__(
        self,
        stitched_weight: StitchedTensor,
        bias: torch.Tensor | None,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
    ):
        super().__init__()
        self.stitched_weight = stitched_weight
        self._use_cache = False
        if bias is not None:
            self.register_buffer("bias", bias.clone().half())
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.stitched_weight.dequantize(use_cache=self._use_cache)
        orig_dtype = x.dtype
        bias_fp16 = self.bias if self.bias is not None else None
        out = F.conv2d(x.half(), w, bias_fp16, self.stride, self.padding, self.dilation, self.groups)
        return out.to(orig_dtype)

    def set_cache_mode(self, enabled: bool = True) -> None:
        self._use_cache = enabled
        if not enabled:
            self.stitched_weight.clear_cache()

    def clear_cache(self) -> None:
        self.stitched_weight.clear_cache()

    @property
    def weight_memory_bytes(self) -> int:
        b = self.stitched_weight.memory_bytes
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return b

    def extra_repr(self) -> str:
        s = self.stitched_weight.qdata.shape
        return (
            f"shape={s}, "
            f"bits={self.stitched_weight.qdata.bit_width}, "
            f"overlay={self.stitched_weight.overlay_fraction:.1%}, "
            f"compression={self.stitched_weight.compression_ratio:.2f}x, "
            f"stride={self.stride}, pad={self.padding}"
        )
