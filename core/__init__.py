"""
Color Align & Evaluation Core Package.
Provides ISO/CIE standard color metrics, color space conversions, and adaptive color harmonization.
"""

from .metrics import (
    ciede2000,
    delta_e76,
    bhattacharyya_3d,
    fast_w1_channel,
    cosine_sim_3d,
    compute_ssim,
    compute_psnr,
    extract_histograms,
    generate_diff_heatmap,
)
from .aligner import AdaptiveColorAligner, color_transfer_align
from .pipeline import ColorPipeline, process_image_pairs

__all__ = [
    "ciede2000",
    "delta_e76",
    "bhattacharyya_3d",
    "fast_w1_channel",
    "cosine_sim_3d",
    "compute_ssim",
    "compute_psnr",
    "extract_histograms",
    "generate_diff_heatmap",
    "AdaptiveColorAligner",
    "color_transfer_align",
    "ColorPipeline",
    "process_image_pairs",
]
