"""
StitchQuant — Hybrid Quantized + Sparse FP16 Overlay Weight Compression

Architecture:
    1. Quantize all weights to N-bit (arbitrary: 1.58, 2, 3, 4, 5, 6, 8-bit)
    2. Identify the most sensitive weights via quantization-error analysis
    3. Store those weights' original FP16 values in a sparse overlay table with pointers
    4. During inference, dequantize + stitch the FP16 overlay in O(K) time
    
Result: Low memory advantage of the quant version as the structural frame,
        and the precision of FP16 on critical weights where it matters.
"""

__version__ = "1.0.0"

from .quantizer import Quantizer, QuantizedData
from .sensitivity import SensitivityAnalyzer
from .stitched_tensor import StitchedTensor
from .modules import StitchedLinear, StitchedConv1D, StitchedConv2d
from .engine import StitchQuantEngine
from .jit_engine import ContiguousFP16MetadataTable, JITStitchedModule, StitchQuantPerfectEngine
from .auto import AutoStitchQuant, ArchitectureDetector, HardwareDetector
from .profiler import Profiler

__all__ = [
    "Quantizer",
    "QuantizedData",
    "SensitivityAnalyzer", 
    "StitchedTensor",
    "StitchedLinear",
    "StitchedConv1D",
    "StitchedConv2d",
    "StitchQuantEngine",
    "ContiguousFP16MetadataTable",
    "JITStitchedModule",
    "StitchQuantPerfectEngine",
    "AutoStitchQuant",
    "ArchitectureDetector",
    "HardwareDetector",
    "Profiler",
]
