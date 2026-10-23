"""
StitchQuant Engine — converts a full-precision model to StitchQuant format.

Usage:
    engine = StitchQuantEngine(bit_width=4, overlay_fraction=0.05)
    sq_model = engine.quantize_model(model)
    # sq_model is a drop-in replacement with lower memory and ~FP16 accuracy
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .quantizer import Quantizer
from .sensitivity import SensitivityAnalyzer
from .stitched_tensor import StitchedTensor
from .modules import StitchedLinear, StitchedConv1D, StitchedConv2d


class StitchQuantEngine:
    """Main engine: quantize a model with sparse FP16 overlay stitching."""

    def __init__(
        self,
        bit_width: float = 4.0,
        overlay_fraction: float = 0.05,
        group_size: int = 128,
        sensitivity: str = "quant_error",
        min_tensor_numel: int = 64,
    ):
        """
        Args:
            bit_width: Quantization bit width (supports 1, 1.58, 2, 3, 4, 5, 6, 8, etc.)
            overlay_fraction: Fraction of weights to keep at FP16 (0.0 to 1.0).
                             0.05 means 5% of weights get FP16 overlay.
            group_size: Quantization group size. Smaller = more precise, more overhead.
            sensitivity: Method for selecting overlay weights.
                        'quant_error' | 'magnitude' | 'hybrid'
            min_tensor_numel: Skip quantization for tiny tensors below this size.
        """
        self.bit_width = bit_width
        self.overlay_fraction = overlay_fraction
        self.group_size = group_size
        self.sensitivity = sensitivity
        self.min_tensor_numel = min_tensor_numel
        self.quantizer = Quantizer(bit_width, group_size)

    def quantize_tensor(self, weight: torch.Tensor) -> StitchedTensor:
        """Quantize a single weight tensor with FP16 overlay.
        
        Args:
            weight: FP16 or FP32 weight tensor.
            
        Returns:
            StitchedTensor containing quantized base + FP16 overlay table.
        """
        weight_fp = weight.detach().float()

        # Quantize base frame
        qdata = self.quantizer.quantize(weight_fp)

        # Identify critical outlier/sensitive weights
        if self.overlay_fraction > 0:
            indices = SensitivityAnalyzer.select(
                weight_fp, self.quantizer, self.overlay_fraction, self.sensitivity
            )
            # Extract exact original FP16 values for the overlay
            values = weight_fp.view(-1)[indices].half()
            indices = indices.to(torch.int32)
        else:
            indices = torch.empty(0, dtype=torch.int32, device=weight_fp.device)
            values = torch.empty(0, dtype=torch.float16, device=weight_fp.device)

        return StitchedTensor(qdata, indices, values, self.quantizer)

    def quantize_model(
        self,
        model: nn.Module,
        skip_layers: Optional[set[str]] = None,
    ) -> nn.Module:
        """Convert an entire model to StitchQuant.
        
        Replaces nn.Linear, HuggingFace Conv1D (in Transformers/LLMs),
        and nn.Conv2d with stitched equivalents.
        
        Embeddings, LayerNorm, and tiny layers are preserved at full precision.
        
        Args:
            model: The original FP16/FP32 model.
            skip_layers: Optional set of layer names or substrings to skip.
            
        Returns:
            A new model with StitchQuant layers and caching control methods.
        """
        model = copy.deepcopy(model)
        model = model.cpu().float()
        skip_layers = skip_layers or set()

        replacements = {}
        for name, module in model.named_modules():
            # Check skip
            if name in skip_layers or any(skip in name for skip in skip_layers):
                continue

            # Standard nn.Linear
            if isinstance(module, nn.Linear):
                if module.weight.numel() < self.min_tensor_numel:
                    continue
                st = self.quantize_tensor(module.weight.data)
                bias = module.bias.data if module.bias is not None else None
                replacements[name] = StitchedLinear(st, bias)

            # Hugging Face Transformer Conv1D (e.g., GPT-2 c_attn, c_proj, c_fc)
            elif module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
                if module.weight.numel() < self.min_tensor_numel:
                    continue
                st = self.quantize_tensor(module.weight.data)
                bias = module.bias.data if getattr(module, "bias", None) is not None else None
                replacements[name] = StitchedConv1D(st, bias)

            # 2D Convolution
            elif isinstance(module, nn.Conv2d):
                if module.weight.numel() < self.min_tensor_numel:
                    continue
                st = self.quantize_tensor(module.weight.data)
                bias = module.bias.data if module.bias is not None else None
                replacements[name] = StitchedConv2d(
                    st, bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )

        # Apply replacements
        for name, new_module in replacements.items():
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_module)

        # Attach cache controls to model
        def enable_cache():
            for m in model.modules():
                if isinstance(m, (StitchedLinear, StitchedConv1D, StitchedConv2d)):
                    m.set_cache_mode(True)

        def disable_cache():
            for m in model.modules():
                if isinstance(m, (StitchedLinear, StitchedConv1D, StitchedConv2d)):
                    m.set_cache_mode(False)

        def clear_cache():
            for m in model.modules():
                if isinstance(m, (StitchedLinear, StitchedConv1D, StitchedConv2d)):
                    m.clear_cache()

        model.enable_cache = enable_cache
        model.disable_cache = disable_cache
        model.clear_cache = clear_cache

        return model

    def model_summary(self, original: nn.Module, stitched: nn.Module) -> dict:
        """Compare original and stitched model statistics."""
        orig_bytes = sum(p.numel() * p.element_size() for p in original.parameters())
        
        sq_bytes = 0
        overlay_total = 0
        weight_total = 0
        for module in stitched.modules():
            if isinstance(module, (StitchedLinear, StitchedConv1D, StitchedConv2d)):
                sq_bytes += module.weight_memory_bytes
                st = module.stitched_weight
                overlay_total += st.overlay_indices.numel()
                weight_total += st.qdata.numel
            elif hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                # Unquantized layers (embeddings, LN, etc.)
                sq_bytes += module.weight.numel() * module.weight.element_size()
                if hasattr(module, 'bias') and module.bias is not None:
                    sq_bytes += module.bias.numel() * module.bias.element_size()

        return {
            "original_bytes": orig_bytes,
            "stitched_bytes": sq_bytes,
            "compression_ratio": orig_bytes / max(1, sq_bytes),
            "original_mb": orig_bytes / (1024 * 1024),
            "stitched_mb": sq_bytes / (1024 * 1024),
            "overlay_weights": overlay_total,
            "total_quantized_weights": weight_total,
            "overlay_fraction": overlay_total / max(1, weight_total),
            "bit_width": self.bit_width,
            "group_size": self.group_size,
        }
