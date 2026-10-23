"""
Multi-bit-width uniform quantizer with proper bit-packing.

Supports arbitrary bit widths including fractional (1.58-bit ternary, 2.5-bit, etc.)
by computing the effective number of quantization levels as round(2^bit_width).

Bit-packing is implemented for 1, 2, 3, 4, 5, 6, 8-bit widths.
Group quantization divides weights into contiguous groups, each with its own
scale and zero-point for higher precision.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass


@dataclass
class QuantizedData:
    """Container for quantized weight data."""
    packed: torch.Tensor        # Bit-packed uint8 tensor
    scales: torch.Tensor        # FP16 per-group scales
    zeros: torch.Tensor         # FP16 per-group zero points
    shape: tuple                # Original tensor shape
    numel: int                  # Original number of elements
    num_groups: int             # Number of quantization groups
    group_size: int             # Elements per group
    bit_width: float            # Quantization bit width


class Quantizer:
    """Arbitrary bit-width group quantizer with bit-packing."""

    def __init__(
        self,
        bit_width: float = 4.0,
        group_size: int = 128,
        symmetric: bool = True,
    ):
        self.bit_width = bit_width
        self.group_size = group_size
        self.symmetric = symmetric

        # Effective number of quantization levels
        self.num_levels = max(2, int(round(2 ** bit_width)))

        # Integer bits needed to represent a single quantized value
        self._storage_bits = math.ceil(math.log2(self.num_levels))

        if symmetric:
            self.qmin = -(self.num_levels // 2)
            self.qmax = max(1, self.num_levels // 2 - 1)  # Ensure qmax >= 1
        else:
            self.qmin = 0
            self.qmax = max(1, self.num_levels - 1)  # Ensure range >= 1

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def quantize(self, weight: torch.Tensor) -> QuantizedData:
        """Quantize a weight tensor with group quantization and bit-packing."""
        orig_shape = weight.shape
        orig_numel = weight.numel()

        # Work in float32 for quantization math
        w = weight.detach().float().reshape(-1)

        # Pad to multiple of group_size
        gs = self.group_size
        padded = math.ceil(w.numel() / gs) * gs
        if padded > w.numel():
            w = torch.nn.functional.pad(w, (0, padded - w.numel()))

        num_groups = w.numel() // gs
        w = w.reshape(num_groups, gs)

        if self.symmetric:
            amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scales = amax / self.qmax
            zeros = torch.zeros(num_groups, 1, dtype=torch.float32, device=w.device)
            quant = torch.clamp(torch.round(w / scales), self.qmin, self.qmax)
            # Shift to unsigned for packing: value - qmin
            quant_unsigned = (quant - self.qmin).to(torch.int32)
        else:
            wmin = w.amin(dim=1, keepdim=True)
            wmax = w.amax(dim=1, keepdim=True)
            rng = (wmax - wmin).clamp(min=1e-12)
            scales = rng / (self.num_levels - 1)
            zeros = torch.round(-wmin / scales)
            quant_unsigned = torch.clamp(
                torch.round(w / scales + zeros), 0, self.num_levels - 1
            ).to(torch.int32)

        packed = self._pack(quant_unsigned.reshape(-1))

        return QuantizedData(
            packed=packed,
            scales=scales.reshape(-1).half(),
            zeros=zeros.reshape(-1).half(),
            shape=orig_shape,
            numel=orig_numel,
            num_groups=num_groups,
            group_size=gs,
            bit_width=self.bit_width,
        )

    def dequantize(self, qdata: QuantizedData) -> torch.Tensor:
        """Dequantize packed data back to FP16."""
        total_packed = qdata.num_groups * qdata.group_size
        quant_unsigned = self._unpack(qdata.packed, total_packed)
        quant_unsigned = quant_unsigned.reshape(qdata.num_groups, qdata.group_size).float()

        scales = qdata.scales.float().unsqueeze(1)   # [G, 1]
        zeros = qdata.zeros.float().unsqueeze(1)      # [G, 1]

        if self.symmetric:
            quant_signed = quant_unsigned + self.qmin  # Shift back
            w = quant_signed * scales
        else:
            w = (quant_unsigned - zeros) * scales

        w = w.reshape(-1)[: qdata.numel]
        return w.reshape(qdata.shape).half()

    def simulate(self, weight: torch.Tensor) -> torch.Tensor:
        """Simulate quantization without bit-packing for fast O(1) sensitivity analysis."""
        w = weight.detach().float().reshape(-1)
        gs = self.group_size
        padded = math.ceil(w.numel() / gs) * gs
        pad_len = padded - w.numel()
        if pad_len > 0:
            w = torch.nn.functional.pad(w, (0, pad_len))
        num_groups = w.numel() // gs
        w_grp = w.reshape(num_groups, gs)

        if self.symmetric:
            amax = w_grp.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scales = amax / self.qmax
            quant = torch.clamp(torch.round(w_grp / scales), self.qmin, self.qmax)
            recon = quant * scales
        else:
            wmin = w_grp.amin(dim=1, keepdim=True)
            wmax = w_grp.amax(dim=1, keepdim=True)
            rng = (wmax - wmin).clamp(min=1e-12)
            scales = rng / (self.num_levels - 1)
            zeros = torch.round(-wmin / scales)
            quant = torch.clamp(torch.round(w_grp / scales + zeros), 0, self.num_levels - 1)
            recon = (quant - zeros) * scales

        recon = recon.reshape(-1)[: weight.numel()]
        return recon.reshape(weight.shape).half()

    def bits_per_weight(self) -> float:
        """Effective bits per weight including group metadata overhead."""
        data_bits = self._storage_bits
        # 2 x FP16 values (scale + zero) per group
        meta_bits = (16 + 16) / self.group_size
        return data_bits + meta_bits

    # ------------------------------------------------------------------ #
    #  Bit-packing internals                                              #
    # ------------------------------------------------------------------ #

    def _pack(self, data: torch.Tensor) -> torch.Tensor:
        """Pack integer values into bit-packed uint8 bytes."""
        bits = self._storage_bits
        data = data.to(torch.int32)
        mask = (1 << bits) - 1
        data = data & mask

        if bits == 8:
            return data.to(torch.uint8)

        if bits == 4:
            # 2 values per byte
            n = data.numel()
            if n % 2:
                data = torch.nn.functional.pad(data, (0, 1))
            lo = data[0::2] & 0x0F
            hi = (data[1::2] & 0x0F) << 4
            return (lo | hi).to(torch.uint8)

        if bits == 2:
            # 4 values per byte
            n = data.numel()
            pad = (4 - n % 4) % 4
            if pad:
                data = torch.nn.functional.pad(data, (0, pad))
            a = data[0::4] & 0x03
            b = (data[1::4] & 0x03) << 2
            c = (data[2::4] & 0x03) << 4
            d = (data[3::4] & 0x03) << 6
            return (a | b | c | d).to(torch.uint8)

        if bits == 1:
            # 8 values per byte
            n = data.numel()
            pad = (8 - n % 8) % 8
            if pad:
                data = torch.nn.functional.pad(data, (0, pad))
            result = torch.zeros(data.numel() // 8, dtype=torch.int32, device=data.device)
            for i in range(8):
                result |= (data[i::8] & 0x01) << i
            return result.to(torch.uint8)

        if bits == 3:
            # Pack 8 x 3-bit values into 3 bytes (24 bits)
            n = data.numel()
            pad = (8 - n % 8) % 8
            if pad:
                data = torch.nn.functional.pad(data, (0, pad))
            groups = data.reshape(-1, 8)
            # Combine 8 values × 3 bits = 24 bits = 3 bytes
            packed = torch.zeros(groups.shape[0], 3, dtype=torch.int32, device=data.device)
            # Byte 0: v0[2:0] | v1[2:0] | v2[1:0]
            packed[:, 0] = (groups[:, 0] & 0x07) | ((groups[:, 1] & 0x07) << 3) | ((groups[:, 2] & 0x03) << 6)
            # Byte 1: v2[2] | v3[2:0] | v4[2:0] | v5[1:0]
            packed[:, 1] = ((groups[:, 2] >> 2) & 0x01) | ((groups[:, 3] & 0x07) << 1) | ((groups[:, 4] & 0x07) << 4) | ((groups[:, 5] & 0x01) << 7)
            # Byte 2: v5[2:1] | v6[2:0] | v7[2:0]
            packed[:, 2] = ((groups[:, 5] >> 1) & 0x03) | ((groups[:, 6] & 0x07) << 2) | ((groups[:, 7] & 0x07) << 5)
            return packed.reshape(-1).to(torch.uint8)

        if bits == 5:
            # Pack 8 x 5-bit values into 5 bytes (40 bits)
            n = data.numel()
            pad = (8 - n % 8) % 8
            if pad:
                data = torch.nn.functional.pad(data, (0, pad))
            groups = data.reshape(-1, 8)
            packed = torch.zeros(groups.shape[0], 5, dtype=torch.int32, device=data.device)
            packed[:, 0] = (groups[:, 0] & 0x1F) | ((groups[:, 1] & 0x07) << 5)
            packed[:, 1] = ((groups[:, 1] >> 3) & 0x03) | ((groups[:, 2] & 0x1F) << 2) | ((groups[:, 3] & 0x01) << 7)
            packed[:, 2] = ((groups[:, 3] >> 1) & 0x0F) | ((groups[:, 4] & 0x0F) << 4)
            packed[:, 3] = ((groups[:, 4] >> 4) & 0x01) | ((groups[:, 5] & 0x1F) << 1) | ((groups[:, 6] & 0x03) << 6)
            packed[:, 4] = ((groups[:, 6] >> 2) & 0x07) | ((groups[:, 7] & 0x1F) << 3)
            return packed.reshape(-1).to(torch.uint8)

        if bits == 6:
            # Pack 4 x 6-bit values into 3 bytes (24 bits)
            n = data.numel()
            pad = (4 - n % 4) % 4
            if pad:
                data = torch.nn.functional.pad(data, (0, pad))
            groups = data.reshape(-1, 4)
            packed = torch.zeros(groups.shape[0], 3, dtype=torch.int32, device=data.device)
            packed[:, 0] = (groups[:, 0] & 0x3F) | ((groups[:, 1] & 0x03) << 6)
            packed[:, 1] = ((groups[:, 1] >> 2) & 0x0F) | ((groups[:, 2] & 0x0F) << 4)
            packed[:, 2] = ((groups[:, 2] >> 4) & 0x03) | ((groups[:, 3] & 0x3F) << 2)
            return packed.reshape(-1).to(torch.uint8)

        # Fallback: store as uint8 (wasteful but correct)
        return data.to(torch.uint8)

    def _unpack(self, packed: torch.Tensor, total: int) -> torch.Tensor:
        """Unpack bit-packed uint8 bytes back to individual integer values."""
        bits = self._storage_bits
        packed = packed.to(torch.int32)

        if bits == 8:
            return packed[:total]

        if bits == 4:
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            unpacked = torch.stack([lo, hi], dim=1).reshape(-1)
            return unpacked[:total]

        if bits == 2:
            a = packed & 0x03
            b = (packed >> 2) & 0x03
            c = (packed >> 4) & 0x03
            d = (packed >> 6) & 0x03
            unpacked = torch.stack([a, b, c, d], dim=1).reshape(-1)
            return unpacked[:total]

        if bits == 1:
            parts = []
            for i in range(8):
                parts.append((packed >> i) & 0x01)
            unpacked = torch.stack(parts, dim=1).reshape(-1)
            return unpacked[:total]

        if bits == 3:
            groups = packed.reshape(-1, 3)
            b0, b1, b2 = groups[:, 0], groups[:, 1], groups[:, 2]
            v = torch.zeros(groups.shape[0], 8, dtype=torch.int32, device=packed.device)
            v[:, 0] = b0 & 0x07
            v[:, 1] = (b0 >> 3) & 0x07
            v[:, 2] = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
            v[:, 3] = (b1 >> 1) & 0x07
            v[:, 4] = (b1 >> 4) & 0x07
            v[:, 5] = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
            v[:, 6] = (b2 >> 2) & 0x07
            v[:, 7] = (b2 >> 5) & 0x07
            return v.reshape(-1)[:total]

        if bits == 5:
            groups = packed.reshape(-1, 5)
            b0, b1, b2, b3, b4 = groups[:, 0], groups[:, 1], groups[:, 2], groups[:, 3], groups[:, 4]
            v = torch.zeros(groups.shape[0], 8, dtype=torch.int32, device=packed.device)
            v[:, 0] = b0 & 0x1F
            v[:, 1] = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3)
            v[:, 2] = (b1 >> 2) & 0x1F
            v[:, 3] = ((b1 >> 7) & 0x01) | ((b2 & 0x0F) << 1)
            v[:, 4] = ((b2 >> 4) & 0x0F) | ((b3 & 0x01) << 4)
            v[:, 5] = (b3 >> 1) & 0x1F
            v[:, 6] = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2)
            v[:, 7] = (b4 >> 3) & 0x1F
            return v.reshape(-1)[:total]

        if bits == 6:
            groups = packed.reshape(-1, 3)
            b0, b1, b2 = groups[:, 0], groups[:, 1], groups[:, 2]
            v = torch.zeros(groups.shape[0], 4, dtype=torch.int32, device=packed.device)
            v[:, 0] = b0 & 0x3F
            v[:, 1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
            v[:, 2] = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
            v[:, 3] = (b2 >> 2) & 0x3F
            return v.reshape(-1)[:total]

        # Fallback
        return packed[:total]
