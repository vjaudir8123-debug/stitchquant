#!/usr/bin/env python3
"""
StitchQuant — Full Benchmark Demo
==================================
Trains a real VGG-style CNN on CIFAR-10, then applies StitchQuant at multiple
bit widths and overlay fractions. Produces a comprehensive comparison table.

Zero external dependencies beyond torch. Downloads CIFAR-10 directly.

Usage:
    python demo.py
    python demo.py --epochs 30 --bit-widths 2 3 4 5 --overlays 0.0 0.05 0.10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pickle
import tarfile
import urllib.request
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stitch_quant import StitchQuantEngine, Profiler


# ===================================================================== #
#  CIFAR-10 loader (no torchvision)                                      #
# ===================================================================== #

def download_cifar10(data_dir: str) -> tuple:
    """Download and load CIFAR-10 without torchvision."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.isdir(extracted):
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.isfile(archive):
            print(f"    Downloading CIFAR-10 from {url}...")
            urllib.request.urlretrieve(url, archive)
        print("    Extracting...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir)

    # Load training batches
    train_data, train_labels = [], []
    for i in range(1, 6):
        batch_file = os.path.join(extracted, f"data_batch_{i}")
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        train_data.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    train_data = np.concatenate(train_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    train_labels = np.array(train_labels, dtype=np.int64)

    # Load test batch
    with open(os.path.join(extracted, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    test_data = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = np.array(batch[b"labels"], dtype=np.int64)

    # Normalize: mean/std per channel
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1).astype(np.float32)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    return (
        torch.from_numpy(train_data), torch.from_numpy(train_labels),
        torch.from_numpy(test_data), torch.from_numpy(test_labels),
    )


def get_cifar10(batch_size: int = 128):
    """Get CIFAR-10 DataLoaders."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_x, train_y, test_x, test_y = download_cifar10(data_dir)

    trainset = TensorDataset(train_x, train_y)
    testset = TensorDataset(test_x, test_y)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader


# ===================================================================== #
#  Model: VGG-style CNN for CIFAR-10                                     #
# ===================================================================== #

class VGGNet(nn.Module):
    """VGG-style CNN for CIFAR-10.  ~2.4M parameters."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3→64, 32×32
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 16×16

            # Block 2: 64→128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 8×8

            # Block 3: 128→256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 4×4

            # Block 4: 256→512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → 1×1
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ===================================================================== #
#  Data augmentation (on-the-fly, no torchvision)                        #
# ===================================================================== #

def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip + random crop with padding=4."""
    B, C, H, W = images.shape
    
    # Random horizontal flip
    flip_mask = torch.rand(B) > 0.5
    images[flip_mask] = images[flip_mask].flip(-1)
    
    # Random crop: pad 4, then crop back to 32x32
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    crops = []
    for i in range(B):
        top = torch.randint(0, 9, (1,)).item()
        left = torch.randint(0, 9, (1,)).item()
        crops.append(padded[i, :, top:top+H, left:left+W])
    return torch.stack(crops)


# ===================================================================== #
#  Training                                                              #
# ===================================================================== #

def train_model(
    model: nn.Module,
    trainloader,
    testloader,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str | None = None,
) -> nn.Module:
    """Train the model with SGD + cosine annealing."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n  ✓ Loading pre-trained checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        print(f"    Checkpoint accuracy: {state['accuracy']:.2f}%")
        return model.to(device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    print(f"\n  Training VGGNet on CIFAR-10 ({epochs} epochs, device={device})")
    print(f"  {'─' * 62}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = augment_batch(inputs)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        elapsed = time.time() - start_time

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch+1:3d}/{epochs} │ "
            f"Loss {train_loss:.4f} │ "
            f"Train {train_acc:6.2f}% │ "
            f"Test {test_acc:6.2f}% │ "
            f"LR {lr:.5f} │ "
            f"{elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            if checkpoint_path:
                torch.save({"model": model.state_dict(), "accuracy": best_acc}, checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        print(f"\n  ✓ Best test accuracy: {state['accuracy']:.2f}%")

    return model


# ===================================================================== #
#  Benchmark                                                             #
# ===================================================================== #

def run_benchmark(
    model_fp: nn.Module,
    testloader,
    device: torch.device,
    bit_widths: list[float],
    overlay_fractions: list[float],
    group_size: int = 128,
):
    """Run the full StitchQuant benchmark."""
    results = []

    # ── FP32 Baseline (our training precision) ──
    print("\n  Evaluating FP32 baseline...")
    model_base = model_fp.float().to(device)
    model_base.eval()

    fp32_mem = Profiler.measure_memory(model_base)

    # For accuracy eval, keep FP32 (our model was trained in FP32)
    # We'll use a wrapper that handles the FP16 input cast
    base_acc = evaluate_fp32(model_base, testloader, device)

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    base_lat = Profiler.measure_latency(model_base, dummy_input, warmup=10, repeats=50)

    results.append({
        "config": "FP32 Baseline",
        "bits": 32.0,
        "overlay": 0.0,
        "accuracy": base_acc["accuracy_pct"],
        "loss": base_acc["loss"],
        "memory_mb": fp32_mem["total_mb"],
        "compression": 1.0,
        "latency_ms": base_lat["mean_ms"],
        "latency_std": base_lat["std_ms"],
    })
    print(f"    Accuracy: {base_acc['accuracy_pct']:.2f}%  Memory: {fp32_mem['total_mb']:.3f} MB  Latency: {base_lat['mean_ms']:.2f} ms")

    baseline_acc = base_acc["accuracy_pct"]
    baseline_bytes = fp32_mem["total_bytes"]

    # ── StitchQuant configurations ──
    model_fp32 = model_fp.float().cpu()

    for bw in bit_widths:
        for ov in overlay_fractions:
            config_name = f"{bw:g}-bit"
            if ov > 0:
                config_name += f" + {ov:.0%} overlay"
            else:
                config_name += " pure quant"

            print(f"\n  Evaluating {config_name}...")
            engine = StitchQuantEngine(
                bit_width=bw,
                overlay_fraction=ov,
                group_size=group_size,
                sensitivity="quant_error",
            )

            try:
                sq_model = engine.quantize_model(model_fp32)
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                import traceback; traceback.print_exc()
                continue

            sq_model = sq_model.to(device)
            sq_model.eval()

            sq_mem = Profiler.measure_memory(sq_model)
            sq_acc = evaluate_stitched(sq_model, testloader, device)

            dummy_sq = torch.randn(1, 3, 32, 32, device=device)
            sq_lat = Profiler.measure_latency(sq_model, dummy_sq, warmup=10, repeats=50)

            compression = baseline_bytes / max(1, sq_mem["total_bytes"])

            results.append({
                "config": config_name,
                "bits": bw,
                "overlay": ov,
                "accuracy": sq_acc["accuracy_pct"],
                "loss": sq_acc["loss"],
                "memory_mb": sq_mem["total_mb"],
                "compression": compression,
                "latency_ms": sq_lat["mean_ms"],
                "latency_std": sq_lat["std_ms"],
            })

            delta = sq_acc["accuracy_pct"] - baseline_acc
            print(
                f"    Accuracy: {sq_acc['accuracy_pct']:.2f}% ({delta:+.2f}%)  "
                f"Memory: {sq_mem['total_mb']:.3f} MB  "
                f"Compression: {compression:.2f}x  "
                f"Latency: {sq_lat['mean_ms']:.2f} ms"
            )

            # Layer-by-layer reconstruction analysis for overlay configs
            if ov > 0:
                recon_errors = Profiler.measure_reconstruction_error(model_fp32, sq_model.cpu())
                if recon_errors:
                    avg_cos = sum(r["cosine_sim"] for r in recon_errors) / len(recon_errors)
                    avg_rmse = sum(r["rmse"] for r in recon_errors) / len(recon_errors)
                    print(f"    Reconstruction: avg cosine_sim={avg_cos:.6f}  avg RMSE={avg_rmse:.6f}")
                sq_model = sq_model.to(device)

    return results


@torch.no_grad()
def evaluate_fp32(model, dataloader, device):
    """Evaluate a standard FP32 model."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in dataloader:
        inputs = inputs.to(device).float()
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "accuracy": correct / total,
        "accuracy_pct": 100.0 * correct / total,
        "loss": total_loss / total,
    }


@torch.no_grad()
def evaluate_stitched(model, dataloader, device):
    """Evaluate a StitchQuant model (handles FP16/FP32 input casting)."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets)
        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "accuracy": correct / total,
        "accuracy_pct": 100.0 * correct / total,
        "loss": total_loss / total,
    }


def print_results_table(results: list[dict]):
    """Print a beautiful comparison table."""
    print("\n")
    print("=" * 108)
    print("                            StitchQuant Benchmark Results")
    print("=" * 108)
    print(
        f"  {'Configuration':<28s} │ {'Accuracy':>8s} │ {'Δ Base':>8s} │ "
        f"{'Memory':>9s} │ {'Compress':>8s} │ {'Latency':>10s} │ {'Loss':>7s}"
    )
    print("─" * 108)

    baseline_acc = results[0]["accuracy"] if results else 0

    for r in results:
        delta = r["accuracy"] - baseline_acc
        delta_str = f"{delta:+.2f}%" if r["config"] != "FP32 Baseline" else "   ─── "
        print(
            f"  {r['config']:<28s} │ "
            f"{r['accuracy']:7.2f}% │ "
            f"{delta_str:>8s} │ "
            f"{r['memory_mb']:7.3f} MB │ "
            f"{r['compression']:6.2f}x  │ "
            f"{r['latency_ms']:7.2f} ms  │ "
            f"{r['loss']:7.4f}"
        )

    print("─" * 108)

    # Highlight best StitchQuant result
    sq_results = [r for r in results if r["overlay"] > 0]
    if sq_results:
        best = max(sq_results, key=lambda r: r["accuracy"])
        print(
            f"\n  ★ Best StitchQuant: {best['config']} → "
            f"{best['accuracy']:.2f}% accuracy at {best['compression']:.2f}x compression"
        )
        if baseline_acc > 0:
            recovery = (best['accuracy'] / baseline_acc) * 100
            print(f"    FP32 accuracy recovery: {recovery:.1f}%")

    # Pure quant vs stitchquant comparison
    print(f"\n  {'─' * 74}")
    print(f"  StitchQuant Advantage (overlay vs pure quant at same bit width):")
    print(f"  {'─' * 74}")

    pure_quants = {r["bits"]: r for r in results if r["overlay"] == 0 and r["bits"] < 32}
    for r in sorted(sq_results, key=lambda x: (x["bits"], x["overlay"])):
        pure = pure_quants.get(r["bits"])
        if pure:
            acc_gain = r["accuracy"] - pure["accuracy"]
            mem_overhead = ((r["memory_mb"] - pure["memory_mb"]) / max(0.001, pure["memory_mb"])) * 100
            print(
                f"    {r['config']:<28s}: +{acc_gain:.2f}% accuracy, "
                f"{mem_overhead:+.1f}% memory vs pure {r['bits']:g}-bit"
            )

    print()


def print_layer_analysis(model_fp: nn.Module, sq_model: nn.Module, config: str):
    """Print layer-by-layer analysis for one config."""
    from stitch_quant.modules import StitchedLinear, StitchedConv2d

    recon = Profiler.measure_reconstruction_error(model_fp, sq_model)
    if not recon:
        return

    print(f"\n  Layer-by-layer reconstruction ({config}):")
    print(f"  {'Layer':<35s} │ {'Shape':<20s} │ {'RMSE':>10s} │ {'Cos Sim':>10s} │ {'Overlay':>8s} │ {'Compr':>6s}")
    print(f"  {'─' * 100}")
    for r in recon:
        shape_str = "×".join(str(s) for s in r["shape"])
        print(
            f"  {r['layer']:<35s} │ "
            f"{shape_str:<20s} │ "
            f"{r['rmse']:10.6f} │ "
            f"{r['cosine_sim']:10.8f} │ "
            f"{r['overlay_frac']:7.1%} │ "
            f"{r['compression']:5.2f}x"
        )


# ===================================================================== #
#  Main                                                                  #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="StitchQuant Benchmark Demo")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument(
        "--bit-widths", nargs="+", type=float, default=[2.0, 3.0, 4.0, 5.0],
        help="Bit widths to benchmark"
    )
    parser.add_argument(
        "--overlays", nargs="+", type=float, default=[0.0, 0.05, 0.10],
        help="Overlay fractions to benchmark"
    )
    parser.add_argument("--no-train", action="store_true", help="Skip training (requires checkpoint)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║               StitchQuant v1.0 — Live Benchmark            ║")
    print("║   Hybrid Quantized Weights + Sparse FP16 Overlay Stitching ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    device = torch.device("cpu")
    print(f"\n  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")

    # Data
    print("\n  Loading CIFAR-10...")
    trainloader, testloader = get_cifar10(args.batch_size)

    # Model
    model = VGGNet(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: VGGNet ({param_count:,} parameters, {param_count*4/(1024*1024):.2f} MB FP32)")

    # Train
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggnet_cifar10.pt")
    if not args.no_train:
        model = train_model(model, trainloader, testloader, args.epochs, device, ckpt_path)
    else:
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
            print(f"  Loaded checkpoint: {state['accuracy']:.2f}%")
        else:
            print("  ERROR: No checkpoint found. Run without --no-train first.")
            sys.exit(1)

    model = model.cpu().float()

    # Benchmark
    print("\n" + "=" * 70)
    print("  BENCHMARK: Comparing quantization configurations")
    print("=" * 70)

    results = run_benchmark(
        model, testloader, device,
        bit_widths=args.bit_widths,
        overlay_fractions=args.overlays,
        group_size=args.group_size,
    )

    # Results table
    print_results_table(results)

    # Detailed layer analysis for the best StitchQuant config
    sq_results = [r for r in results if r["overlay"] > 0]
    if sq_results:
        best = max(sq_results, key=lambda r: r["accuracy"])
        engine = StitchQuantEngine(
            bit_width=best["bits"],
            overlay_fraction=best["overlay"],
            group_size=args.group_size,
        )
        sq_model = engine.quantize_model(model)
        print_layer_analysis(model, sq_model, best["config"])

    print("\n  ✓ Benchmark complete.\n")


if __name__ == "__main__":
    main()
