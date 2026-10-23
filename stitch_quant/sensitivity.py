"""
Weight sensitivity analysis for FP16 overlay candidate selection.

Three strategies determine which weights get the sparse FP16 overlay:

1. quant_error  — Weights where quantization introduces the largest error.
                  Directly targets the source of accuracy loss. (Default)
2. magnitude   — Weights with the largest absolute value. Large weights
                  tend to dominate activations and are most impactful.
3. hybrid      — Weighted combination of both signals.
"""

from __future__ import annotations

import torch

from .quantizer import Quantizer


class SensitivityAnalyzer:
    """Identifies the most critical weights for FP16 overlay preservation."""

    @staticmethod
    def by_quantization_error(
        weight: torch.Tensor,
        quantizer: Quantizer,
        fraction: float,
    ) -> torch.Tensor:
        """Select weights where quantization error is largest.
        
        This is the most principled method: it directly measures how much
        each weight is distorted by quantization and preserves the ones
        that suffer the most. We use squared error to amplify outliers — 
        a single weight with 10x error is far more damaging than 10 weights
        with 1x error because matmul accumulates errors.
        
        Args:
            weight: Original FP16/FP32 weight tensor.
            quantizer: Configured Quantizer instance.
            fraction: Fraction of weights to select (0.0 to 1.0).
            
        Returns:
            Sorted 1-D int64 tensor of flat indices into weight.view(-1).
        """
        reconstructed = quantizer.simulate(weight)
        # Squared error amplifies outliers — they dominate output error
        error = (weight.float() - reconstructed.float()).pow(2).view(-1)

        k = max(1, int(fraction * error.numel()))
        _, indices = error.topk(k)
        return indices.sort().values

    @staticmethod
    def by_magnitude(
        weight: torch.Tensor,
        fraction: float,
    ) -> torch.Tensor:
        """Select weights with largest absolute magnitude.
        
        Args:
            weight: Original weight tensor.
            fraction: Fraction of weights to select.
            
        Returns:
            Sorted 1-D int64 tensor of flat indices.
        """
        w_flat = weight.float().abs().view(-1)
        k = max(1, int(fraction * w_flat.numel()))
        _, indices = w_flat.topk(k)
        return indices.sort().values

    @staticmethod
    def by_hybrid(
        weight: torch.Tensor,
        quantizer: Quantizer,
        fraction: float,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """Hybrid selection: alpha × quant_error + (1-alpha) × magnitude.
        
        Balances two signals:
        - Quantization error: which weights are most distorted?
        - Magnitude: which weights have the most influence?
        
        Args:
            weight: Original weight tensor.
            quantizer: Configured Quantizer.
            fraction: Fraction of weights to select.
            alpha: Weight for quant_error signal (1-alpha for magnitude).
            
        Returns:
            Sorted 1-D int64 tensor of flat indices.
        """
        reconstructed = quantizer.simulate(weight)
        error = (weight.float() - reconstructed.float()).pow(2).view(-1)
        magnitude = weight.float().abs().view(-1)

        # Normalize both to [0, 1]
        e_max = error.max().clamp(min=1e-12)
        m_max = magnitude.max().clamp(min=1e-12)
        error_norm = error / e_max
        mag_norm = magnitude / m_max

        score = alpha * error_norm + (1 - alpha) * mag_norm
        k = max(1, int(fraction * score.numel()))
        _, indices = score.topk(k)
        return indices.sort().values

    @staticmethod
    def select(
        weight: torch.Tensor,
        quantizer: Quantizer,
        fraction: float,
        method: str = "quant_error",
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch sensitivity selection by method name.
        
        Args:
            weight: Original weight tensor.
            quantizer: Configured Quantizer.
            fraction: Fraction of weights to select.
            method: One of 'quant_error', 'magnitude', 'hybrid'.
            
        Returns:
            Sorted 1-D int64 tensor of flat indices.
        """
        if method == "quant_error":
            return SensitivityAnalyzer.by_quantization_error(weight, quantizer, fraction)
        elif method == "magnitude":
            return SensitivityAnalyzer.by_magnitude(weight, fraction)
        elif method == "hybrid":
            return SensitivityAnalyzer.by_hybrid(weight, quantizer, fraction, **kwargs)
        else:
            raise ValueError(f"Unknown sensitivity method: {method!r}")
