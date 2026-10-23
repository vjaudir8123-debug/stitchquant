"""
StitchedTensor — the core data structure.

A StitchedTensor stores a weight matrix as:

    ┌──────────────────────────────────────────┐
    │  Quantized Base  (N-bit, bit-packed)     │  ← bulk storage, low memory
    │  + group scales & zeros (FP16)           │
    ├──────────────────────────────────────────┤
    │  Sparse FP16 Overlay                     │  ← sorted flat indices (int32)
    │  + original FP16 values                  │     + FP16 values for those indices
    └──────────────────────────────────────────┘

Dequantization:
    1. Standard group dequant: int → FP16 (approximate)
    2. Scatter overlay: overwrite selected positions with exact FP16 values
    Result: quantized-memory footprint with FP16 precision where it matters.
"""

from __future__ import annotations

import torch

from .quantizer import Quantizer, QuantizedData


class StitchedTensor:
    """Quantized weight tensor with sparse FP16 overlay stitching."""

    __slots__ = (
        "qdata",
        "overlay_indices",
        "overlay_values",
        "_quantizer",
        "_cached",
    )

    def __init__(
        self,
        qdata: QuantizedData,
        overlay_indices: torch.Tensor,
        overlay_values: torch.Tensor,
        quantizer: Quantizer,
    ):
        """
        Args:
            qdata: Quantized weight data from Quantizer.quantize().
            overlay_indices: Sorted int32 flat indices of FP16 overlay weights.
            overlay_values: FP16 values for those indices.
            quantizer: The Quantizer instance used (for dequantization).
        """
        self.qdata = qdata
        self.overlay_indices = overlay_indices
        self.overlay_values = overlay_values
        self._quantizer = quantizer
        self._cached = None

    @torch.no_grad()
    def dequantize(self, use_cache: bool = False) -> torch.Tensor:
        """Reconstruct the full weight tensor.
        
        Step 1: Dequantize the N-bit base → FP16 approximation.
        Step 2: Scatter the sparse FP16 overlay (exact values) over the base.
        
        Returns:
            FP16 tensor with the original shape.
        """
        if use_cache and self._cached is not None:
            return self._cached

        # Step 1: Standard group dequant
        w = self._quantizer.dequantize(self.qdata)  # [shape], FP16

        # Step 2: Stitch FP16 overlay via scatter
        if self.overlay_indices.numel() > 0:
            w_flat = w.view(-1)
            w_flat[self.overlay_indices.long()] = self.overlay_values
            # w is already modified in-place via the view

        if use_cache:
            self._cached = w

        return w

    def clear_cache(self) -> None:
        """Release cached reconstructed tensor to immediately free RAM."""
        self._cached = None

    def to(self, device: torch.device) -> "StitchedTensor":
        """Move all tensors to the given device."""
        self.qdata = QuantizedData(
            packed=self.qdata.packed.to(device),
            scales=self.qdata.scales.to(device),
            zeros=self.qdata.zeros.to(device),
            shape=self.qdata.shape,
            numel=self.qdata.numel,
            num_groups=self.qdata.num_groups,
            group_size=self.qdata.group_size,
            bit_width=self.qdata.bit_width,
        )
        self.overlay_indices = self.overlay_indices.to(device)
        self.overlay_values = self.overlay_values.to(device)
        return self

    # ------------------------------------------------------------------ #
    #  Memory accounting                                                  #
    # ------------------------------------------------------------------ #

    @property
    def memory_bytes(self) -> int:
        """Total memory footprint in bytes (all components)."""
        packed_bytes = self.qdata.packed.numel() * self.qdata.packed.element_size()
        scales_bytes = self.qdata.scales.numel() * self.qdata.scales.element_size()
        zeros_bytes = self.qdata.zeros.numel() * self.qdata.zeros.element_size()
        idx_bytes = self.overlay_indices.numel() * self.overlay_indices.element_size()
        val_bytes = self.overlay_values.numel() * self.overlay_values.element_size()
        return packed_bytes + scales_bytes + zeros_bytes + idx_bytes + val_bytes

    @property
    def fp16_equivalent_bytes(self) -> int:
        """What this tensor would cost in pure FP16."""
        return self.qdata.numel * 2

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        return self.fp16_equivalent_bytes / max(1, self.memory_bytes)

    @property
    def overlay_fraction(self) -> float:
        """Fraction of weights stored at FP16 in the overlay."""
        return self.overlay_indices.numel() / max(1, self.qdata.numel)

    @property
    def effective_bits_per_weight(self) -> float:
        """Effective bits per weight element (actual, not theoretical)."""
        return (self.memory_bytes * 8) / max(1, self.qdata.numel)

    def summary(self) -> dict:
        """Return a summary dict of this stitched tensor's properties."""
        return {
            "shape": self.qdata.shape,
            "numel": self.qdata.numel,
            "bit_width": self.qdata.bit_width,
            "group_size": self.qdata.group_size,
            "overlay_count": self.overlay_indices.numel(),
            "overlay_fraction": f"{self.overlay_fraction:.2%}",
            "memory_bytes": self.memory_bytes,
            "fp16_bytes": self.fp16_equivalent_bytes,
            "compression_ratio": f"{self.compression_ratio:.2f}x",
            "effective_bpw": f"{self.effective_bits_per_weight:.2f}",
        }
