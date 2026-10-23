"""
AutoStitchQuant: Automatic Architecture-Aware & Hardware-Adaptive Engine.

Automatically inspects:
1. Model Family & Architecture (GPT, LLaMA/Mistral/Qwen, BERT/Encoder, ViT, Custom).
2. Weight Topology (Tied embeddings, Gated MLPs, Attention Projections, Norms).
3. Host Hardware Profile (RAM availability, CPU vs CUDA/MPS device).
4. Automatic Parameter Tuning (Optimal base bit-width 1.0b/2.0b/4.0b, group size, and streaming mode).

Usage:
    from stitch_quant import AutoStitchQuant
    
    # 1-line automatic adaptation for ANY model:
    auto_model = AutoStitchQuant.from_model(model)
    
    # Or from HuggingFace directly:
    auto_model = AutoStitchQuant.from_pretrained("gpt2")
"""

from __future__ import annotations

import copy
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quantizer
from .stitched_tensor import StitchedTensor
from .jit_engine import ContiguousFP16MetadataTable, PointerEntry, JITStitchedModule


@dataclass
class HardwareProfile:
    """Hardware telemetry and capacity assessment."""
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    device_type: str
    device_name: str
    recommended_mode: str
    recommended_base_bits: float
    recommended_group_size: int


@dataclass
class ArchitectureProfile:
    """Model architecture taxonomy and topological insights."""
    model_class_name: str
    family: str  # 'gpt', 'llama_style', 'encoder', 'vision', 'generic'
    total_parameters: int
    trainable_parameters: int
    uncompressed_fp16_mb: float
    num_layers_total: int
    num_convertible_layers: int
    has_tied_weights: bool
    tied_pairs: List[Tuple[str, str]]
    detected_layer_types: Set[str]


class ArchitectureDetector:
    """Inspects arbitrary PyTorch models and identifies architectural characteristics."""

    @staticmethod
    def inspect(model: nn.Module) -> ArchitectureProfile:
        model_class = model.__class__.__name__
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fp16_mb = (total_params * 2) / (1024 * 1024)

        # Detect layer names and modules
        named_mods = dict(model.named_modules())
        detected_types = {m.__class__.__name__ for m in named_mods.values()}
        
        # Check for tied weights (e.g. lm_head.weight is wte.weight)
        tied_pairs = []
        data_ptrs: Dict[int, str] = {}
        for name, param in model.named_parameters():
            ptr = param.data.data_ptr()
            if ptr in data_ptrs:
                tied_pairs.append((data_ptrs[ptr], name))
            else:
                data_ptrs[ptr] = name

        has_tied = len(tied_pairs) > 0

        # Family classification
        name_lower = model_class.lower()
        module_names_joined = " ".join(named_mods.keys()).lower()

        if "llama" in name_lower or "mistral" in name_lower or "qwen" in name_lower or "gemma" in name_lower or "gate_proj" in module_names_joined:
            family = "llama_style"
        elif "gpt" in name_lower or "conv1d" in " ".join(detected_types).lower() or "c_attn" in module_names_joined:
            family = "gpt"
        elif "bert" in name_lower or "roberta" in name_lower or "encoder" in name_lower:
            family = "encoder"
        elif "vit" in name_lower or "vision" in name_lower or "conv2d" in " ".join(detected_types).lower():
            family = "vision"
        else:
            family = "generic"

        # Count convertible layers
        convertible_count = 0
        for name, m in named_mods.items():
            is_linear = isinstance(m, nn.Linear)
            is_conv1d = m.__class__.__name__ == "Conv1D" and hasattr(m, "weight")
            is_embed = isinstance(m, nn.Embedding)
            if (is_linear or is_conv1d or is_embed) and hasattr(m, "weight") and m.weight is not None:
                convertible_count += 1

        return ArchitectureProfile(
            model_class_name=model_class,
            family=family,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            uncompressed_fp16_mb=fp16_mb,
            num_layers_total=len(named_mods),
            num_convertible_layers=convertible_count,
            has_tied_weights=has_tied,
            tied_pairs=tied_pairs,
            detected_layer_types=detected_types,
        )


class HardwareDetector:
    """Profiles system memory and hardware accelerators to determine optimal settings."""

    @staticmethod
    def profile(model_fp16_mb: float) -> HardwareProfile:
        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024**3)
        avail_ram = vm.available / (1024**3)
        used_pct = vm.percent

        if torch.cuda.is_available():
            dev_type = "cuda"
            dev_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev_type = "mps"
            dev_name = "Apple Silicon GPU (MPS)"
        else:
            dev_type = "cpu"
            dev_name = "Host CPU (POSIX)"

        # Calculate memory pressure ratio: available RAM vs uncompressed model size
        avail_ram_mb = avail_ram * 1024
        headroom_ratio = avail_ram_mb / max(1.0, model_fp16_mb)

        # Automatic Decision Rules:
        if headroom_ratio < 2.0:
            # Constrained memory: aggressive 1-bit base structural frame with streaming
            rec_mode = "jit"
            rec_bits = 1.0
            rec_group = 128
        elif headroom_ratio < 4.0:
            # Moderate memory: 2-bit or 3-bit base frame
            rec_mode = "jit"
            rec_bits = 2.0
            rec_group = 128
        else:
            # Abundant memory: 4-bit base with dynamic session caching
            rec_mode = "pinned_fp16"
            rec_bits = 4.0
            rec_group = 128

        return HardwareProfile(
            total_ram_gb=total_ram,
            available_ram_gb=avail_ram,
            ram_usage_percent=used_pct,
            device_type=dev_type,
            device_name=dev_name,
            recommended_mode=rec_mode,
            recommended_base_bits=rec_bits,
            recommended_group_size=rec_group,
        )


class AutoStitchQuant:
    """Universal Automated Factory for Architecture-Aware & Hardware-Adaptive StitchQuant."""

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        target_device: Optional[torch.device] = None,
        storage_path: Optional[str] = None,
        force_mode: Optional[str] = None,
        force_bits: Optional[float] = None,
        force_group_size: Optional[int] = None,
    ) -> nn.Module:
        """Automatically analyze, tune, and convert ANY neural network to StitchQuant.
        
        Args:
            model: PyTorch model instance (e.g. GPT-2, LLaMA, BERT, ViT, etc.)
            target_device: Device for computation ('cpu', 'cuda', 'mps'). Defaults to auto-detect.
            storage_path: Path for contiguous FP16 binary table. Defaults to auto-managed cache.
            force_mode: Override auto-selected mode ('jit', 'pinned_fp16', 'quant_base').
            force_bits: Override auto-selected base bits (e.g. 1.0, 2.0, 4.0).
            force_group_size: Override auto-selected group size (e.g. 64, 128).
            
        Returns:
            Stitched model instance with auto-adjustment telemetry attached.
        """
        # Step 1: Detect Architecture
        arch = ArchitectureDetector.inspect(model)
        
        # Step 2: Profile Host Hardware
        hw = HardwareDetector.profile(arch.uncompressed_fp16_mb)

        # Step 3: Compute Adjusted Configuration
        base_bits = force_bits if force_bits is not None else hw.recommended_base_bits
        mode = force_mode if force_mode is not None else hw.recommended_mode
        group_size = force_group_size if force_group_size is not None else hw.recommended_group_size

        # Architecture-specific adjustments
        if arch.family == "llama_style":
            # LLaMA SwiGLU gates are wide; 128 group size is optimal
            group_size = 128
        elif arch.family == "encoder":
            # Encoders like BERT are small (100M); pin directly for maximum latency speed
            if force_mode is None and hw.available_ram_gb > 2.0:
                mode = "pinned_fp16"
        elif arch.family == "vision":
            # Vision models benefit from smaller group size for spatial filters
            group_size = 64

        device = target_device or torch.device(hw.device_type if hw.device_type != "mps" else "cpu")

        # Step 4: Setup Storage Directory
        if storage_path is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "autostitch_cache")
            os.makedirs(cache_dir, exist_ok=True)
            safe_name = arch.model_class_name.lower()
            storage_path = os.path.join(cache_dir, f"{safe_name}_{int(time.time())}.bin")

        # Step 5: Convert Model using Metadata Table & Dynamic AST
        quantizer = Quantizer(bit_width=base_bits, group_size=group_size)
        meta_table = ContiguousFP16MetadataTable(storage_path=storage_path)

        # Harvest weights
        named_weights: Dict[str, torch.Tensor] = {}
        targets: Dict[str, nn.Module] = {}
        
        # Handle tied weights gracefully: map aliases
        pointer_aliases: Dict[str, str] = {}
        for source_name, target_name in arch.tied_pairs:
            pointer_aliases[target_name] = source_name

        for name, module in model.named_modules():
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")
            is_embed = isinstance(module, nn.Embedding)
            
            if (is_linear or is_conv1d or is_embed) and hasattr(module, "weight") and module.weight is not None:
                # If this layer is tied to an earlier layer, we don't duplicate its recording!
                if name in pointer_aliases:
                    targets[name] = module
                else:
                    named_weights[name] = module.weight.data
                    targets[name] = module

        # Record exact weights to binary table on disk
        meta_table.record_weights(named_weights, storage_path=storage_path)

        # Build stitched replacement modules
        stitched_model = copy.deepcopy(model).cpu()
        replacements = {}

        for name, module in targets.items():
            ref_name = pointer_aliases.get(name, name)
            orig_weight = named_weights[ref_name].float()

            # Create low-bit quantized base frame (e.g. 1-bit or 4-bit)
            qdata = quantizer.quantize(orig_weight)
            dummy_idx = torch.empty(0, dtype=torch.int32)
            dummy_val = torch.empty(0, dtype=torch.float16)
            st = StitchedTensor(qdata, dummy_idx, dummy_val, quantizer)

            bias = module.bias.data if getattr(module, "bias", None) is not None else None
            
            jit_mod = JITStitchedModule(
                name=ref_name,  # Uses the source pointer entry if tied
                original_module=module,
                stitched_weight=st,
                pointer_table=meta_table,
                bias=bias,
            )
            jit_mod.mode = mode
            replacements[name] = jit_mod

        # Inject into model tree
        for name, new_module in replacements.items():
            parts = name.split(".")
            parent = stitched_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_module)

        # Attach management & auto-tuning telemetry methods
        def explain_adjustments() -> Dict[str, Any]:
            return {
                "detected_family": arch.family,
                "model_class": arch.model_class_name,
                "total_params": arch.total_parameters,
                "original_fp16_mb": arch.uncompressed_fp16_mb,
                "auto_selected_base_bits": base_bits,
                "auto_selected_group_size": group_size,
                "auto_selected_mode": mode,
                "tied_weights_detected": arch.tied_pairs,
                "host_hardware": {
                    "total_ram_gb": round(hw.total_ram_gb, 2),
                    "available_ram_gb": round(hw.available_ram_gb, 2),
                    "device": hw.device_name,
                },
                "structural_ram_compression": f"~{16.0 / base_bits:.1f}x theoretical base compression",
            }

        def set_mode(m: str):
            for mod in stitched_model.modules():
                if isinstance(mod, JITStitchedModule):
                    mod.mode = m

        def release_all():
            for mod in stitched_model.modules():
                if isinstance(mod, JITStitchedModule):
                    mod.release_fp16()

        def cleanup_storage():
            if os.path.exists(storage_path):
                os.remove(storage_path)

        stitched_model.explain_adjustments = explain_adjustments
        stitched_model.set_stitch_mode = set_mode
        stitched_model.release_all_fp16 = release_all
        stitched_model.cleanup_storage = cleanup_storage
        stitched_model.meta_table = meta_table
        stitched_model.auto_arch = arch
        stitched_model.auto_hw = hw

        return stitched_model

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> nn.Module:
        """Download or load any HuggingFace model and automatically convert it."""
        from transformers import AutoModelForCausalLM, AutoModel
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        except Exception:
            model = AutoModel.from_pretrained(model_name_or_path)
            
        return cls.from_model(model, **kwargs)
