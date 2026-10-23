"""
Profiler — measures memory footprint, inference latency, layer-by-layer
reconstruction error, and Perplexity / Accuracy for StitchQuant models.
"""

from __future__ import annotations

import time
import math
from typing import Optional, List

import torch
import torch.nn as nn

from .modules import StitchedLinear, StitchedConv1D, StitchedConv2d


class Profiler:
    """Profile StitchQuant models for memory, speed, and accuracy/perplexity."""

    @staticmethod
    def measure_memory(model: nn.Module) -> dict:
        """Measure total parameter + buffer memory in the model."""
        param_bytes = 0
        param_count = 0

        for module in model.modules():
            if isinstance(module, (StitchedLinear, StitchedConv1D, StitchedConv2d)):
                param_bytes += module.weight_memory_bytes
                st = module.stitched_weight
                param_count += st.qdata.numel
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                for p in module.parameters(recurse=False):
                    param_bytes += p.numel() * p.element_size()
                    param_count += p.numel()
            elif module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
                for p in module.parameters(recurse=False):
                    param_bytes += p.numel() * p.element_size()
                    param_count += p.numel()
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.Embedding)):
                for p in module.parameters(recurse=False):
                    param_bytes += p.numel() * p.element_size()
                    param_count += p.numel()
                for b in module.buffers():
                    param_bytes += b.numel() * b.element_size()

        return {
            "total_bytes": param_bytes,
            "total_mb": param_bytes / (1024 * 1024),
            "total_params": param_count,
        }

    @staticmethod
    @torch.no_grad()
    def measure_latency(
        model: nn.Module,
        input_tensor: torch.Tensor,
        warmup: int = 5,
        repeats: int = 20,
    ) -> dict:
        """Measure inference latency (forward pass).
        
        Args:
            model: Model to benchmark.
            input_tensor: Example input tensor.
            warmup: Number of warmup passes.
            repeats: Number of timed passes.
            
        Returns:
            Dict with mean, std, min, max latency in milliseconds.
        """
        model.eval()

        # Warmup
        for _ in range(warmup):
            _ = model(input_tensor)

        # Timed
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        times_t = torch.tensor(times)
        return {
            "mean_ms": times_t.mean().item(),
            "std_ms": times_t.std().item(),
            "min_ms": times_t.min().item(),
            "max_ms": times_t.max().item(),
        }

    @staticmethod
    @torch.no_grad()
    def measure_reconstruction_error(
        original: nn.Module,
        stitched: nn.Module,
    ) -> list[dict]:
        """Layer-by-layer weight reconstruction error between original and stitched."""
        orig_weights = {}
        for name, module in original.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                orig_weights[name] = module.weight.data.float()
            elif module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
                orig_weights[name] = module.weight.data.float()

        results = []
        for name, module in stitched.named_modules():
            if isinstance(module, (StitchedLinear, StitchedConv1D, StitchedConv2d)):
                if name not in orig_weights:
                    continue
                orig_w = orig_weights[name]
                recon_w = module.stitched_weight.dequantize().float()

                diff = orig_w - recon_w
                rmse = diff.pow(2).mean().sqrt().item()
                max_err = diff.abs().max().item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_w.reshape(1, -1), recon_w.reshape(1, -1)
                ).item()

                results.append({
                    "layer": name,
                    "shape": list(orig_w.shape),
                    "numel": orig_w.numel(),
                    "rmse": rmse,
                    "max_error": max_err,
                    "cosine_sim": cos_sim,
                    "overlay_frac": module.stitched_weight.overlay_fraction,
                    "compression": module.stitched_weight.compression_ratio,
                })

        return results

    @staticmethod
    @torch.no_grad()
    def evaluate_perplexity(
        model: nn.Module,
        tokenizer,
        texts: List[str],
        device: torch.device,
        max_length: int = 128,
    ) -> dict:
        """Evaluate Perplexity (PPL) of an autoregressive language model on real text.
        
        Perplexity is the gold standard benchmark metric for LLM quantization:
        PPL = exp(mean cross-entropy loss)
        Lower is better. Catastrophic quantization causes PPL -> 1000+ or NaN.
        """
        model.eval()
        model.to(device)

        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue

            # In causal language modeling, target is input_ids shifted
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            tokens = input_ids.size(1)

            total_loss += loss * tokens
            total_tokens += tokens

        if total_tokens == 0:
            return {"loss": float("nan"), "perplexity": float("nan")}

        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20.0))  # clamp to avoid overflow if collapsed

        return {
            "loss": avg_loss,
            "perplexity": ppl,
            "total_tokens": total_tokens,
        }

    @staticmethod
    @torch.no_grad()
    def generate_text(
        model: nn.Module,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 30,
        device: torch.device = torch.device("cpu"),
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a prompt to verify output coherence."""
        model.eval()
        model.to(device)

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Greedy / top-1 generation for reproducibility
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(generated[0], skip_special_tokens=True)
