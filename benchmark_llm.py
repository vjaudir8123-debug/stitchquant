#!/usr/bin/env python3
"""
StitchQuant — Production LLM Benchmark Suite
============================================
Benchmarks StitchQuant on real LLM weights (GPT-2, 124M parameters)
across arbitrary bit-widths:
  - FP16 Baseline
  - 5-bit (Pure vs 5% Overlay)
  - 4-bit (Pure vs 3% Overlay vs 5% Overlay)
  - 3-bit (Pure vs 5% Overlay vs 10% Overlay)
  - 2-bit (Pure vs 5% Overlay vs 10% Overlay)
  - 1.58-bit Ternary (Pure vs 5% Overlay vs 10% Overlay)

Evaluates:
  1. Real Text Generation Coherence
  2. Perplexity (PPL) on evaluation corpus
  3. Memory footprint and True Compression Ratio
  4. Layer-by-layer Cosine Similarity & Outlier Error Suppression
  5. Per-token Latency
"""

from __future__ import annotations

import os
import sys
import time
import math
from typing import List, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stitch_quant import StitchQuantEngine, Profiler


# ===================================================================== #
#  Evaluation Corpus for Perplexity Measurement                          #
# ===================================================================== #

EVAL_TEXTS = [
    "Artificial intelligence and deep learning have fundamentally reshaped modern computational science and engineering.",
    "Quantization reduces the numerical precision of weights and activations to accelerate matrix multiplication on modern accelerators.",
    "In large language models, outlier activations emerge in specific hidden dimensions and dominate the representation space.",
    "Sparse overlay tables preserve high-precision floating point parameters for the most sensitive coordinates of the neural network.",
    "The transformer architecture relies on multi-head scaled dot-product attention and feed-forward networks across stacked layers.",
    "Operating systems manage hardware resources and provide abstract system call interfaces for user-space applications.",
    "Distributed computing requires consensus protocols like Raft or Paxos to guarantee fault tolerance in asynchronous networks.",
    "Theoretical physics seeks to reconcile general relativity with quantum field theory into a consistent framework of quantum gravity.",
]

BENCHMARK_PROMPT = "Artificial intelligence is transforming science by"


# ===================================================================== #
#  Memory Accounting                                                     #
# ===================================================================== #

def get_backbone_memory_bytes(model: nn.Module) -> int:
    """Calculate exact memory of transformer blocks (quantizable weights)."""
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_memory_bytes"):
            total += module.weight_memory_bytes
        elif module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
            total += module.weight.numel() * module.weight.element_size()
            if getattr(module, "bias", None) is not None:
                total += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, nn.Linear) and not any(k in name for k in ["lm_head"]):
            total += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                total += module.bias.numel() * module.bias.element_size()
    return total


# ===================================================================== #
#  Benchmark Runner                                                      #
# ===================================================================== #

def run_llm_benchmark(device: torch.device = torch.device("cpu")):
    print("=" * 110)
    print("           StitchQuant LLM Production Benchmark (GPT-2 124M Parameters)           ")
    print("       Hybrid Low-Bit Quantized Frame + Sparse FP16 Overlay Pointer Stitching      ")
    print("=" * 110)

    # 1. Load FP16 Base Model
    print("\n[1/3] Loading Full-Precision FP16 GPT-2 Model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16).to(device)
    base_model.eval()

    total_params = sum(p.numel() for p in base_model.parameters())
    backbone_base_bytes = get_backbone_memory_bytes(base_model)
    print(f"      Total Parameters: {total_params:,}")
    print(f"      Transformer Backbone Memory: {backbone_base_bytes / (1024*1024):.2f} MB (FP16)")

    # Baseline PPL & Generation
    print("\n[2/3] Evaluating FP16 Baseline...")
    _ = Profiler.generate_text(base_model, tokenizer, "Test", max_new_tokens=3, device=device)
    t0 = time.perf_counter()
    fp16_gen = Profiler.generate_text(base_model, tokenizer, BENCHMARK_PROMPT, max_new_tokens=25, device=device)
    gen_time_fp16 = (time.perf_counter() - t0) * 1000 / 25
    fp16_ppl = Profiler.evaluate_perplexity(base_model, tokenizer, EVAL_TEXTS, device=device)

    print(f"      FP16 Perplexity: {fp16_ppl['perplexity']:.2f}")
    print(f"      FP16 Generation: {repr(fp16_gen[:80])}...")
    print(f"      FP16 Speed:      {gen_time_fp16:.2f} ms/token")

    # Define Configurations to Test
    # (bit_width, overlay_fraction, label)
    configs = [
        # FP16 Baseline
        (16.0, 0.0, "FP16 Baseline"),
        
        # 5-bit
        (5.0, 0.0, "Pure 5-bit"),
        (5.0, 0.05, "StitchQuant 5-bit + 5% OV"),

        # 4-bit
        (4.0, 0.0, "Pure 4-bit"),
        (4.0, 0.03, "StitchQuant 4-bit + 3% OV"),
        (4.0, 0.05, "StitchQuant 4-bit + 5% OV"),

        # 3-bit
        (3.0, 0.0, "Pure 3-bit"),
        (3.0, 0.05, "StitchQuant 3-bit + 5% OV"),
        (3.0, 0.10, "StitchQuant 3-bit + 10% OV"),

        # 2-bit
        (2.0, 0.0, "Pure 2-bit"),
        (2.0, 0.05, "StitchQuant 2-bit + 5% OV"),
        (2.0, 0.10, "StitchQuant 2-bit + 10% OV"),

        # 1.58-bit Ternary (BitNet style)
        (1.58, 0.0, "Pure 1.58-bit Ternary"),
        (1.58, 0.05, "StitchQuant 1.58b + 5% OV"),
        (1.58, 0.10, "StitchQuant 1.58b + 10% OV"),
    ]

    results = []

    # Record FP16 baseline row
    results.append({
        "label": "FP16 Baseline",
        "bits": 16.0,
        "overlay": 0.0,
        "backbone_mb": backbone_base_bytes / (1024 * 1024),
        "compression": 1.0,
        "ppl": fp16_ppl["perplexity"],
        "cos_sim": 1.0,
        "max_err": 0.0,
        "generation": fp16_gen,
        "ms_per_token": gen_time_fp16,
    })

    skip = {"lm_head", "wte", "wpe"}

    print("\n[3/3] Running StitchQuant Matrix Quantization & Evaluation...")
    print("-" * 110)

    for bw, ov, label in configs[1:]:
        print(f"\n  --> Testing: {label} (bits={bw}, overlay={ov:.0%})")
        
        engine = StitchQuantEngine(
            bit_width=bw,
            overlay_fraction=ov,
            group_size=128,
            sensitivity="quant_error",
        )

        # Quantize
        t_quant_start = time.perf_counter()
        q_model = engine.quantize_model(base_model, skip_layers=skip)
        t_quant = time.perf_counter() - t_quant_start

        # Measure layer reconstruction accuracy
        recon_errors = Profiler.measure_reconstruction_error(base_model, q_model)
        avg_cos = sum(r["cosine_sim"] for r in recon_errors) / max(1, len(recon_errors))
        max_err = max(r["max_error"] for r in recon_errors) if recon_errors else 0.0
        avg_rmse = sum(r["rmse"] for r in recon_errors) / max(1, len(recon_errors))

        # Memory accounting
        bb_bytes = get_backbone_memory_bytes(q_model)
        bb_mb = bb_bytes / (1024 * 1024)
        compression = backbone_base_bytes / max(1, bb_bytes)

        # Production mode: enable cache once for both PPL and Generation
        q_model.enable_cache()

        # Evaluate PPL
        ppl_res = Profiler.evaluate_perplexity(q_model, tokenizer, EVAL_TEXTS, device=device)
        ppl_val = ppl_res["perplexity"]

        # Evaluate Generation
        _ = Profiler.generate_text(q_model, tokenizer, "Test", max_new_tokens=3, device=device)
        t0 = time.perf_counter()
        gen_text = Profiler.generate_text(q_model, tokenizer, BENCHMARK_PROMPT, max_new_tokens=25, device=device)
        gen_time = (time.perf_counter() - t0) * 1000 / 25
        q_model.clear_cache()

        results.append({
            "label": label,
            "bits": bw,
            "overlay": ov,
            "backbone_mb": bb_mb,
            "compression": compression,
            "ppl": ppl_val,
            "cos_sim": avg_cos,
            "max_err": max_err,
            "rmse": avg_rmse,
            "generation": gen_text,
            "ms_per_token": gen_time,
        })

        print(f"      Compression: {compression:.2f}x ({bb_mb:.2f} MB vs {backbone_base_bytes/(1024*1024):.2f} MB FP16)")
        print(f"      Perplexity:  {ppl_val:.2f} (FP16 baseline: {fp16_ppl['perplexity']:.2f})")
        print(f"      Weight Cosine Similarity: {avg_cos:.6f}")
        print(f"      Sample Output: {repr(gen_text[:75])}...")

    # ================================================================= #
    #  Format Output Tables                                              #
    # ================================================================= #

    print("\n" + "=" * 115)
    print("                                STITCHQUANT COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 115)
    print(
        f"  {'Configuration':<30s} │ {'Backbone':>9s} │ {'Compress':>8s} │ "
        f"{'Perplexity':>10s} │ {'Cos Sim':>9s} │ {'Max Error':>10s} │ {'Speed (ms/tok)':>14s}"
    )
    print("─" * 115)

    base_ppl = fp16_ppl["perplexity"]

    for r in results:
        ppl_str = f"{r['ppl']:9.2f}" if r["ppl"] < 9999 else " >10,000 (FAIL)"
        print(
            f"  {r['label']:<30s} │ "
            f"{r['backbone_mb']:7.2f} MB │ "
            f"{r['compression']:6.2f}x  │ "
            f"{ppl_str:>10s} │ "
            f"{r['cos_sim']:9.6f} │ "
            f"{r['max_err']:10.5f} │ "
            f"{r['ms_per_token']:12.2f} ms"
        )

    print("─" * 115)

    # Comparison summary: Pure Quant vs StitchQuant
    print("\n" + "=" * 115)
    print("                 THE STITCHQUANT ADVANTAGE: PURE QUANT vs STITCHQUANT OVERLAY COMPARISON")
    print("=" * 115)

    pure_map = {r["bits"]: r for r in results if r["overlay"] == 0 and r["bits"] < 16}
    sq_list = [r for r in results if r["overlay"] > 0]

    for sq in sq_list:
        pure = pure_map.get(sq["bits"])
        if pure:
            ppl_drop = pure["ppl"] - sq["ppl"]
            cos_gain = (sq["cos_sim"] - pure["cos_sim"])
            err_reduction = ((pure["max_err"] - sq["max_err"]) / max(1e-6, pure["max_err"])) * 100
            print(f"\n  [{sq['bits']} bits] {sq['label']}:")
            print(f"    • Perplexity: {pure['ppl']:.2f} (pure) → {sq['ppl']:.2f} (stitched)  [Δ -{ppl_drop:.2f} PPL]")
            print(f"    • Weight Cosine Sim: {pure['cos_sim']:.6f} → {sq['cos_sim']:.6f}  [+{cos_gain:.6f}]")
            print(f"    • Max Outlier Error: {pure['max_err']:.5f} → {sq['max_err']:.5f}  [{err_reduction:.1f}% reduction]")
            print(f"    • True Compression:  {sq['compression']:.2f}x (Memory: {sq['backbone_mb']:.2f} MB)")

    print("\n" + "=" * 115)
    print("                                   QUALITATIVE GENERATION SAMPLES")
    print("=" * 115)
    print(f"Prompt: {repr(BENCHMARK_PROMPT)}\n")
    for r in results:
        print(f"[{r['label']}]:")
        print(f"  \"{r['generation']}\"\n")

    return results


if __name__ == "__main__":
    run_llm_benchmark()
