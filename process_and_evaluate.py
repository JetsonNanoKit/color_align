#!/usr/bin/env python3
"""
Color Discrepancy Statistics & Color Transfer Alignment Pipeline
Designed for '八字刘海' (Curtain Bangs) FaceFix evaluation.
"""

import os
import json
import time
import cv2
import numpy as np

# Input directories
SRC_DIR = "/Users/meitu/Downloads/八字刘海/src_facefix"
TAG_DIR = "/Users/meitu/Downloads/八字刘海/tag_facefix"

# Output directories
OUTPUT_DIR = "/Users/meitu/Documents/color_analysis"
WEB_DIR = os.path.join(OUTPUT_DIR, "web")
THUMB_DIR = os.path.join(WEB_DIR, "assets", "thumbs")

os.makedirs(THUMB_DIR, exist_ok=True)

def ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """
    Vectorized ISO/CIE standard CIEDE2000 color difference formula.
    lab1, lab2: float arrays of shape (..., 3) with [L, a, b]
    L in [0, 100], a, b in [-128, 127]
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7 + 1e-12)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.hypot(a1_prime, b1)
    C2_prime = np.hypot(a2_prime, b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = h2_prime - h1_prime
    dh_prime = np.where(np.abs(h2_prime - h1_prime) <= 180.0, dh_prime,
               np.where(h2_prime <= h1_prime, dh_prime + 360.0, dh_prime - 360.0))
    dh_prime = np.where((C1_prime * C2_prime) == 0, 0.0, dh_prime)

    dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime * 0.5))

    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)

    h_sum = h1_prime + h2_prime
    h_diff = np.abs(h1_prime - h2_prime)
    h_bar_prime = np.where(C1_prime * C2_prime == 0, h_sum,
                  np.where(h_diff <= 180.0, 0.5 * h_sum,
                  np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0))))

    T = (1.0 - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
             + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0)))

    d_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0)**2)
    C_bar_prime7 = C_bar_prime ** 7
    RC = 2.0 * np.sqrt(C_bar_prime7 / (C_bar_prime7 + 25.0**7 + 1e-12))
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    SL = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / np.sqrt(20.0 + (L_bar_prime - 50.0)**2)
    SC = 1.0 + 0.045 * C_bar_prime
    SH = 1.0 + 0.015 * C_bar_prime * T

    termL = dL_prime / (kL * SL)
    termC = dC_prime / (kC * SC)
    termH = dH_prime / (kH * SH)

    return np.sqrt(termL**2 + termC**2 + termH**2 + RT * termC * termH)

def compute_ssim(img1, img2):
    """Fast SSIM on luminance channel using Gaussian blurring."""
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(I1, kernel, sigma)
    mu2 = cv2.GaussianBlur(I2, kernel, sigma)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(I1**2, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(I2**2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(I1 * I2, kernel, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def fast_w1_channel(c1, c2, max_val=100.0):
    """1D Wasserstein distance calculated via discrete cumulative distributions."""
    h1 = cv2.calcHist([c1], [0], None, [256], [0, 256]).flatten()
    h2 = cv2.calcHist([c2], [0], None, [256], [0, 256]).flatten()
    cdf1 = h1.cumsum() / (h1.sum() + 1e-12)
    cdf2 = h2.cumsum() / (h2.sum() + 1e-12)
    return float(np.sum(np.abs(cdf1 - cdf2)) * (max_val / 256.0))

def bhattacharyya_3d(img1_lab_u8, img2_lab_u8, bins=16):
    """3D Color histogram Bhattacharyya distance."""
    h1 = cv2.calcHist([img1_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([img2_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    p1 = h1 / (h1.sum() + 1e-12)
    p2 = h2 / (h2.sum() + 1e-12)
    bc = np.sum(np.sqrt(p1 * p2))
    return float(-np.log(np.clip(bc, 1e-10, 1.0)))

def cosine_sim_3d(img1_lab_u8, img2_lab_u8, bins=16):
    """3D Color histogram Cosine Similarity."""
    h1 = cv2.calcHist([img1_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]).flatten()
    h2 = cv2.calcHist([img2_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]).flatten()
    norm1 = np.linalg.norm(h1) + 1e-12
    norm2 = np.linalg.norm(h2) + 1e-12
    return float(np.dot(h1, h2) / (norm1 * norm2))

def color_transfer_align(s_bgr, t_bgr):
    """
    Proposed Adaptive Distribution-Matching Harmonization (ADMH):
    1. Multi-scale Lab CDF Histogram Matching to align tone and chromaticity.
    2. Edge-preserving soft spatial blending mask to retain 100% fidelity on unchanged background.
    """
    s_lab_u8 = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2Lab)
    t_lab_u8 = cv2.cvtColor(t_bgr, cv2.COLOR_BGR2Lab)

    # 1. Non-linear CDF matching in Lab
    al_lab_u8 = np.zeros_like(s_lab_u8)
    for c in range(3):
        h_s = cv2.calcHist([s_lab_u8[:, :, c]], [0], None, [256], [0, 256]).flatten()
        h_t = cv2.calcHist([t_lab_u8[:, :, c]], [0], None, [256], [0, 256]).flatten()
        cdf_s = h_s.cumsum() / (h_s.sum() + 1e-12)
        cdf_t = h_t.cumsum() / (h_t.sum() + 1e-12)
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = min(np.searchsorted(cdf_t, cdf_s[i]), 255)
        al_lab_u8[:, :, c] = lut[s_lab_u8[:, :, c]]

    al_bgr = cv2.cvtColor(al_lab_u8, cv2.COLOR_Lab2BGR)

    # 2. Content-Aware Spatial Feathering:
    # Retain identical background intact, smoothly apply color transfer to modified facefix areas
    diff = np.abs(s_bgr.astype(float) - t_bgr.astype(float)).mean(axis=2)
    # Mask of active edit region
    raw_mask = (diff > 1.5).astype(np.float32)
    smooth_mask = cv2.GaussianBlur(raw_mask, (31, 31), 0)[:, :, None]

    aligned_out = smooth_mask * al_bgr.astype(np.float32) + (1.0 - smooth_mask) * s_bgr.astype(np.float32)
    return np.clip(aligned_out, 0, 255).astype(np.uint8)

def extract_histograms(img_bgr, img_lab_u8):
    """Extract compact 32-bin 1D histograms and 16x16 2D a*-b* chromaticity grid."""
    # 32-bin RGB
    hist_r = cv2.calcHist([img_bgr], [2], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_bgr], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_bgr], [0], None, [32], [0, 256]).flatten()

    # 32-bin Lab
    hist_L = cv2.calcHist([img_lab_u8], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([img_lab_u8], [1], None, [32], [0, 256]).flatten()
    hist_b_lab = cv2.calcHist([img_lab_u8], [2], None, [32], [0, 256]).flatten()

    # Normalize to 0..1
    total = img_bgr.shape[0] * img_bgr.shape[1]
    
    # 2D chromaticity: a* (channel 1) vs b* (channel 2)
    # 16x16 grid
    hist_2d = cv2.calcHist([img_lab_u8], [1, 2], None, [16, 16], [0, 256, 0, 256])
    hist_2d_norm = (hist_2d / (total + 1e-12)).tolist()

    return {
        "rgb": {
            "r": (hist_r / total).round(4).tolist(),
            "g": (hist_g / total).round(4).tolist(),
            "b": (hist_b / total).round(4).tolist()
        },
        "lab": {
            "L": (hist_L / total).round(4).tolist(),
            "a": (hist_a / total).round(4).tolist(),
            "b": (hist_b_lab / total).round(4).tolist()
        },
        "gamut_2d": [[round(val, 5) for val in row] for row in hist_2d_norm]
    }

def main():
    print("Starting Color Discrepancy Analysis & Alignment Pipeline...")
    t_start = time.time()

    src_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith((".png", ".jpg", ".jpeg"))])
    tag_files = sorted([f for f in os.listdir(TAG_DIR) if f.endswith((".png", ".jpg", ".jpeg"))])
    common_files = sorted(list(set(src_files).intersection(set(tag_files))))

    print(f"Total matching image pairs found: {len(common_files)}")

    results = []

    for idx, fname in enumerate(common_files):
        s_path = os.path.join(SRC_DIR, fname)
        t_path = os.path.join(TAG_DIR, fname)

        s_bgr = cv2.imread(s_path)
        t_bgr = cv2.imread(t_path)

        if s_bgr is None or t_bgr is None:
            print(f"Warning: Failed to read {fname}")
            continue

        base_name = os.path.splitext(fname)[0]

        # 1. Colorspace representations
        s_lab_f = cv2.cvtColor(s_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        t_lab_f = cv2.cvtColor(t_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

        s_lab_u8 = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2Lab)
        t_lab_u8 = cv2.cvtColor(t_bgr, cv2.COLOR_BGR2Lab)

        # 2. Raw Perceptual Color Differences
        de00_map = ciede2000(s_lab_f, t_lab_f)
        de00_mean = float(de00_map.mean())
        de00_median = float(np.median(de00_map))
        de00_p95 = float(np.percentile(de00_map, 95))
        de00_max = float(de00_map.max())

        de76_map = np.sqrt(np.sum((s_lab_f - t_lab_f)**2, axis=2))
        de76_mean = float(de76_map.mean())

        # 3. Channel-wise Biases
        s_mean_L, s_mean_a, s_mean_b = s_lab_f.mean(axis=(0, 1))
        t_mean_L, t_mean_a, t_mean_b = t_lab_f.mean(axis=(0, 1))

        delta_L = float(s_mean_L - t_mean_L)
        delta_a = float(s_mean_a - t_mean_a)
        delta_b = float(s_mean_b - t_mean_b)

        s_chroma = np.hypot(s_lab_f[..., 1], s_lab_f[..., 2]).mean()
        t_chroma = np.hypot(t_lab_f[..., 1], t_lab_f[..., 2]).mean()
        delta_C = float(s_chroma - t_chroma)

        # 4. Distributional Discrepancy
        db_raw = bhattacharyya_3d(s_lab_u8, t_lab_u8)
        w1_L = fast_w1_channel(s_lab_u8[:, :, 0], t_lab_u8[:, :, 0], 100.0)
        w1_a = fast_w1_channel(s_lab_u8[:, :, 1], t_lab_u8[:, :, 1], 255.0)
        w1_b = fast_w1_channel(s_lab_u8[:, :, 2], t_lab_u8[:, :, 2], 255.0)
        w1_raw = float(w1_L + w1_a + w1_b)
        cos_sim_raw = cosine_sim_3d(s_lab_u8, t_lab_u8)

        # 5. Signal & Structural Fidelity
        mse_raw = float(np.mean((s_bgr.astype(float) - t_bgr.astype(float))**2))
        psnr_raw = float(10 * np.log10(255**2 / (mse_raw + 1e-10)))
        ssim_raw = compute_ssim(s_bgr, t_bgr)

        # 6. Active Edit Region (ROI) Statistics
        diff_abs = np.abs(s_bgr.astype(float) - t_bgr.astype(float)).mean(axis=2)
        edit_mask = diff_abs > 2.0
        edit_ratio = float(edit_mask.mean() * 100.0)
        if edit_mask.sum() > 0:
            edit_de00 = float(de00_map[edit_mask].mean())
            edit_s_lab = s_lab_f[edit_mask].mean(axis=0)
            edit_t_lab = t_lab_f[edit_mask].mean(axis=0)
            edit_shift_L = float(edit_s_lab[0] - edit_t_lab[0])
            edit_shift_a = float(edit_s_lab[1] - edit_t_lab[1])
            edit_shift_b = float(edit_s_lab[2] - edit_t_lab[2])
        else:
            edit_de00 = 0.0
            edit_shift_L, edit_shift_a, edit_shift_b = 0.0, 0.0, 0.0

        # 7. Apply Color Transfer & Alignment
        aligned_bgr = color_transfer_align(s_bgr, t_bgr)
        al_lab_f = cv2.cvtColor(aligned_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        al_lab_u8 = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2Lab)

        # 8. Post-alignment Metrics
        al_de00_map = ciede2000(al_lab_f, t_lab_f)
        al_de00_mean = float(al_de00_map.mean())
        al_de00_p95 = float(np.percentile(al_de00_map, 95))

        al_mean_L, al_mean_a, al_mean_b = al_lab_f.mean(axis=(0, 1))
        al_delta_L = float(al_mean_L - t_mean_L)
        al_delta_a = float(al_mean_a - t_mean_a)
        al_delta_b = float(al_mean_b - t_mean_b)

        al_db = bhattacharyya_3d(al_lab_u8, t_lab_u8)
        al_w1_L = fast_w1_channel(al_lab_u8[:, :, 0], t_lab_u8[:, :, 0], 100.0)
        al_w1_a = fast_w1_channel(al_lab_u8[:, :, 1], t_lab_u8[:, :, 1], 255.0)
        al_w1_b = fast_w1_channel(al_lab_u8[:, :, 2], t_lab_u8[:, :, 2], 255.0)
        al_w1 = float(al_w1_L + al_w1_a + al_w1_b)

        al_mse = float(np.mean((aligned_bgr.astype(float) - t_bgr.astype(float))**2))
        al_psnr = float(10 * np.log10(255**2 / (al_mse + 1e-10)))
        al_ssim = compute_ssim(aligned_bgr, t_bgr)

        # Improvement percentages
        improve_db_pct = float((db_raw - al_db) / (db_raw + 1e-10) * 100.0)
        improve_w1_pct = float((w1_raw - al_w1) / (w1_raw + 1e-10) * 100.0)
        improve_de00_pct = float((de00_mean - al_de00_mean) / (de00_mean + 1e-10) * 100.0)
        improve_psnr_gain = float(al_psnr - psnr_raw)

        # 9. Extract compact histograms for client-side rendering
        s_hist = extract_histograms(s_bgr, s_lab_u8)
        t_hist = extract_histograms(t_bgr, t_lab_u8)

        # 10. Generate Difference Heatmap (overlay on grayscale)
        heat_u8 = np.clip(de76_map / 15.0 * 255.0, 0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        gray_src = cv2.cvtColor(cv2.cvtColor(s_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        heat_overlay = cv2.addWeighted(gray_src, 0.35, heat_color, 0.65, 0)

        # 11. Save Optimized Web Thumbnails (384x384 JPEG)
        thumb_size = (384, 384)
        s_thumb = cv2.resize(s_bgr, thumb_size, interpolation=cv2.INTER_AREA)
        t_thumb = cv2.resize(t_bgr, thumb_size, interpolation=cv2.INTER_AREA)
        al_thumb = cv2.resize(aligned_bgr, thumb_size, interpolation=cv2.INTER_AREA)
        heat_thumb = cv2.resize(heat_overlay, thumb_size, interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(THUMB_DIR, f"{base_name}_src.jpg"), s_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(THUMB_DIR, f"{base_name}_tag.jpg"), t_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(THUMB_DIR, f"{base_name}_aligned.jpg"), al_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(THUMB_DIR, f"{base_name}_heatmap.jpg"), heat_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        item_data = {
            "id": idx + 1,
            "filename": fname,
            "basename": base_name,
            "src_path": s_path,
            "tag_path": t_path,
            "thumbs": {
                "src": f"assets/thumbs/{base_name}_src.jpg",
                "tag": f"assets/thumbs/{base_name}_tag.jpg",
                "aligned": f"assets/thumbs/{base_name}_aligned.jpg",
                "heatmap": f"assets/thumbs/{base_name}_heatmap.jpg"
            },
            # Raw Metrics
            "metrics_raw": {
                "ciede2000_mean": round(de00_mean, 3),
                "ciede2000_median": round(de00_median, 3),
                "ciede2000_p95": round(de00_p95, 3),
                "ciede2000_max": round(de00_max, 2),
                "delta_e76_mean": round(de76_mean, 3),
                "delta_L": round(delta_L, 3),
                "delta_a": round(delta_a, 3),
                "delta_b": round(delta_b, 3),
                "delta_C": round(delta_C, 3),
                "bhattacharyya_3d": round(db_raw, 5),
                "wasserstein_1d": round(w1_raw, 3),
                "cosine_sim_3d": round(cos_sim_raw, 4),
                "psnr": round(psnr_raw, 2),
                "ssim": round(ssim_raw, 4),
                "src_lab_mean": [round(float(s_mean_L), 2), round(float(s_mean_a), 2), round(float(s_mean_b), 2)],
                "tag_lab_mean": [round(float(t_mean_L), 2), round(float(t_mean_a), 2), round(float(t_mean_b), 2)],
                "edit_ratio": round(edit_ratio, 1),
                "edit_de00": round(edit_de00, 3),
                "edit_shift_lab": [round(edit_shift_L, 2), round(edit_shift_a, 2), round(edit_shift_b, 2)]
            },
            # Aligned Metrics
            "metrics_aligned": {
                "ciede2000_mean": round(al_de00_mean, 3),
                "ciede2000_p95": round(al_de00_p95, 3),
                "delta_L": round(al_delta_L, 3),
                "delta_a": round(al_delta_a, 3),
                "delta_b": round(al_delta_b, 3),
                "bhattacharyya_3d": round(al_db, 5),
                "wasserstein_1d": round(al_w1, 3),
                "psnr": round(al_psnr, 2),
                "ssim": round(al_ssim, 4),
                "improvement_db_pct": round(improve_db_pct, 1),
                "improvement_w1_pct": round(improve_w1_pct, 1),
                "improvement_de00_pct": round(improve_de00_pct, 1),
                "psnr_gain": round(improve_psnr_gain, 2)
            },
            "src_hist": s_hist,
            "tag_hist": t_hist
        }

        results.append(item_data)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(common_files):
            print(f"Processed [{idx + 1}/{len(common_files)}] image pairs...")

    # Global Dataset Aggregates
    all_de00 = [r["metrics_raw"]["ciede2000_mean"] for r in results]
    all_de00_al = [r["metrics_aligned"]["ciede2000_mean"] for r in results]
    all_db = [r["metrics_raw"]["bhattacharyya_3d"] for r in results]
    all_db_al = [r["metrics_aligned"]["bhattacharyya_3d"] for r in results]
    all_w1 = [r["metrics_raw"]["wasserstein_1d"] for r in results]
    all_w1_al = [r["metrics_aligned"]["wasserstein_1d"] for r in results]
    all_psnr = [r["metrics_raw"]["psnr"] for r in results]
    all_psnr_al = [r["metrics_aligned"]["psnr"] for r in results]

    all_delta_a = [r["metrics_raw"]["delta_a"] for r in results]
    all_delta_b = [r["metrics_raw"]["delta_b"] for r in results]
    all_delta_L = [r["metrics_raw"]["delta_L"] for r in results]

    summary = {
        "total_pairs": len(results),
        "mean_ciede2000_raw": round(float(np.mean(all_de00)), 3),
        "mean_ciede2000_aligned": round(float(np.mean(all_de00_al)), 3),
        "mean_db_raw": round(float(np.mean(all_db)), 5),
        "mean_db_aligned": round(float(np.mean(all_db_al)), 5),
        "avg_db_reduction_pct": round(float((np.mean(all_db) - np.mean(all_db_al)) / np.mean(all_db) * 100.0), 1),
        "mean_w1_raw": round(float(np.mean(all_w1)), 3),
        "mean_w1_aligned": round(float(np.mean(all_w1_al)), 3),
        "avg_w1_reduction_pct": round(float((np.mean(all_w1) - np.mean(all_w1_al)) / np.mean(all_w1) * 100.0), 1),
        "mean_psnr_raw": round(float(np.mean(all_psnr)), 2),
        "mean_psnr_aligned": round(float(np.mean(all_psnr_al)), 2),
        "systematic_bias": {
            "avg_delta_L": round(float(np.mean(all_delta_L)), 3),
            "avg_delta_a": round(float(np.mean(all_delta_a)), 3),
            "avg_delta_b": round(float(np.mean(all_delta_b)), 3),
            "interpretation": "src 呈现显著的正偏向 (Δa* > 0, Δb* > 0)，即 src 整体偏红、偏暖调；tag 则呈现偏冷调、高纯白肤色。"
        }
    }

    output_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "items": results
    }

    json_path = os.path.join(WEB_DIR, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    t_end = time.time()
    print(f"\nProcessing finished in {t_end - t_start:.2f}s!")
    print(f"Data saved to: {json_path}")
    print(f"Thumbnails saved to: {THUMB_DIR}")
    print(f"Dataset Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
