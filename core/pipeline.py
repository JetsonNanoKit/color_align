"""
Batch Evaluation & Color Alignment Processing Pipeline.
Handles pair matching, perceptual color metrics calculation, alignment execution,
thumbnail generation, and packaging of aligned results into ZIP archives.
"""

import os
import glob
import json
import time
import zipfile
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

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
from .aligner import AdaptiveColorAligner

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_image_files(root_dir: str) -> Dict[str, str]:
    """Recursively finds all valid image files, ignoring macOS metadata."""
    images = {}
    for dirpath, _, filenames in os.walk(root_dir):
        if "__MACOSX" in dirpath:
            continue
        for fname in filenames:
            if fname.startswith("._") or fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                images[fname] = os.path.join(dirpath, fname)
    return images


def match_image_pairs(src_dir: str, tag_dir: str) -> List[Tuple[str, str, str]]:
    """
    Matches source and target images by filename.
    Returns: list of tuples (filename, src_full_path, tag_full_path)
    """
    src_map = find_image_files(src_dir)
    tag_map = find_image_files(tag_dir)

    common_names = sorted(list(set(src_map.keys()).intersection(set(tag_map.keys()))))
    return [(name, src_map[name], tag_map[name]) for name in common_names]


class ColorPipeline:
    """Manages batch processing, evaluation, and aligned archive creation."""

    def __init__(self, mode: str = "admh", thumb_size: Tuple[int, int] = (384, 384)):
        self.aligner = AdaptiveColorAligner(mode=mode)
        self.thumb_size = thumb_size

    def process(
        self,
        src_dir: str,
        tag_dir: str,
        output_dir: str,
        session_id: Optional[str] = None,
        save_aligned: bool = True,
        create_zip: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        """
        Executes complete analysis and alignment pipeline.
        """
        os.makedirs(output_dir, exist_ok=True)
        aligned_dir = os.path.join(output_dir, "aligned_images")
        thumbs_dir = os.path.join(output_dir, "thumbs")

        if save_aligned:
            os.makedirs(aligned_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)

        pairs = match_image_pairs(src_dir, tag_dir)
        total_pairs = len(pairs)
        if total_pairs == 0:
            raise ValueError(f"No matching image pairs found between '{src_dir}' and '{tag_dir}'.")

        results = []
        t0 = time.time()

        for idx, (fname, s_path, t_path) in enumerate(pairs):
            if progress_cb:
                progress_cb(idx + 1, total_pairs, f"Processing {fname} ({idx+1}/{total_pairs})")

            s_bgr = cv2.imread(s_path)
            t_bgr = cv2.imread(t_path)

            if s_bgr is None or t_bgr is None:
                continue

            # Ensure matching resolutions
            if s_bgr.shape != t_bgr.shape:
                t_bgr = cv2.resize(t_bgr, (s_bgr.shape[1], s_bgr.shape[0]), interpolation=cv2.INTER_AREA)

            base_name = os.path.splitext(fname)[0]

            # Colorspace representations
            s_lab_f = cv2.cvtColor(s_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            t_lab_f = cv2.cvtColor(t_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            s_lab_u8 = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2Lab)
            t_lab_u8 = cv2.cvtColor(t_bgr, cv2.COLOR_BGR2Lab)

            # 1. Raw Metrics
            de00_map = ciede2000(s_lab_f, t_lab_f)
            de00_mean = float(de00_map.mean())
            de00_median = float(np.median(de00_map))
            de00_p95 = float(np.percentile(de00_map, 95))
            de00_max = float(de00_map.max())

            de76_map = delta_e76(s_lab_f, t_lab_f)
            de76_mean = float(de76_map.mean())

            # Channel-wise biases
            s_mean_L, s_mean_a, s_mean_b = s_lab_f.mean(axis=(0, 1))
            t_mean_L, t_mean_a, t_mean_b = t_lab_f.mean(axis=(0, 1))
            delta_L = float(s_mean_L - t_mean_L)
            delta_a = float(s_mean_a - t_mean_a)
            delta_b = float(s_mean_b - t_mean_b)

            s_chroma = np.hypot(s_lab_f[..., 1], s_lab_f[..., 2]).mean()
            t_chroma = np.hypot(t_lab_f[..., 1], t_lab_f[..., 2]).mean()
            delta_C = float(s_chroma - t_chroma)

            # Distributional divergence
            db_raw = bhattacharyya_3d(s_lab_u8, t_lab_u8)
            w1_L = fast_w1_channel(s_lab_u8[:, :, 0], t_lab_u8[:, :, 0], 100.0)
            w1_a = fast_w1_channel(s_lab_u8[:, :, 1], t_lab_u8[:, :, 1], 255.0)
            w1_b = fast_w1_channel(s_lab_u8[:, :, 2], t_lab_u8[:, :, 2], 255.0)
            w1_raw = float(w1_L + w1_a + w1_b)
            cos_sim_raw = cosine_sim_3d(s_lab_u8, t_lab_u8)

            # Structural & signal fidelity
            psnr_raw = compute_psnr(s_bgr, t_bgr)
            ssim_raw = compute_ssim(s_bgr, t_bgr)

            # Active Edit Region (ROI)
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

            # 2. Alignment Transfer
            aligned_bgr = self.aligner.align(s_bgr, t_bgr)

            if save_aligned:
                out_img_path = os.path.join(aligned_dir, fname)
                cv2.imwrite(out_img_path, aligned_bgr)

            # 3. Post-alignment Metrics
            al_lab_f = cv2.cvtColor(aligned_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            al_lab_u8 = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2Lab)

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

            al_psnr = compute_psnr(aligned_bgr, t_bgr)
            al_ssim = compute_ssim(aligned_bgr, t_bgr)

            # Improvement Ratios
            improve_db_pct = float((db_raw - al_db) / (db_raw + 1e-10) * 100.0)
            improve_w1_pct = float((w1_raw - al_w1) / (w1_raw + 1e-10) * 100.0)
            improve_de00_pct = float((de00_mean - al_de00_mean) / (de00_mean + 1e-10) * 100.0)
            improve_psnr_gain = float(al_psnr - psnr_raw)

            # 4. Histograms for client canvas
            s_hist = extract_histograms(s_bgr, s_lab_u8)
            t_hist = extract_histograms(t_bgr, t_lab_u8)

            # 5. Turbo Heatmap
            heat_overlay = generate_diff_heatmap(s_bgr, t_bgr)

            # 6. Save Thumbnails
            s_thumb = cv2.resize(s_bgr, self.thumb_size, interpolation=cv2.INTER_AREA)
            t_thumb = cv2.resize(t_bgr, self.thumb_size, interpolation=cv2.INTER_AREA)
            al_thumb = cv2.resize(aligned_bgr, self.thumb_size, interpolation=cv2.INTER_AREA)
            heat_thumb = cv2.resize(heat_overlay, self.thumb_size, interpolation=cv2.INTER_AREA)

            s_th_name = f"{base_name}_src.jpg"
            t_th_name = f"{base_name}_tag.jpg"
            al_th_name = f"{base_name}_aligned.jpg"
            heat_th_name = f"{base_name}_heatmap.jpg"

            cv2.imwrite(os.path.join(thumbs_dir, s_th_name), s_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(os.path.join(thumbs_dir, t_th_name), t_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(os.path.join(thumbs_dir, al_th_name), al_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(os.path.join(thumbs_dir, heat_th_name), heat_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Path prefix for browser access
            prefix = f"sessions/{session_id}/thumbs" if session_id else "assets/thumbs"

            results.append({
                "id": idx + 1,
                "filename": fname,
                "basename": base_name,
                "src_path": s_path,
                "tag_path": t_path,
                "thumbs": {
                    "src": f"{prefix}/{s_th_name}",
                    "tag": f"{prefix}/{t_th_name}",
                    "aligned": f"{prefix}/{al_th_name}",
                    "heatmap": f"{prefix}/{heat_th_name}",
                },
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
                    "edit_shift_lab": [round(edit_shift_L, 2), round(edit_shift_a, 2), round(edit_shift_b, 2)],
                },
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
                    "psnr_gain": round(improve_psnr_gain, 2),
                },
                "src_hist": s_hist,
                "tag_hist": t_hist,
            })

        # Global dataset aggregates
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

        mean_da = float(np.mean(all_delta_a))
        mean_db = float(np.mean(all_delta_b))
        mean_dL = float(np.mean(all_delta_L))

        bias_desc = []
        if mean_dL > 0.5:
            bias_desc.append(f"src 偏亮 (+{mean_dL:.2f})")
        elif mean_dL < -0.5:
            bias_desc.append(f"src 偏暗 ({mean_dL:.2f})")

        if mean_da > 0.05:
            bias_desc.append("整体偏红/洋红 (Δa* > 0)")
        elif mean_da < -0.05:
            bias_desc.append("整体偏绿 (Δa* < 0)")

        if mean_db > 0.05:
            bias_desc.append("偏暖黄调 (Δb* > 0)")
        elif mean_db < -0.05:
            bias_desc.append("偏冷蓝调 (Δb* < 0)")

        interpretation = "，".join(bias_desc) if bias_desc else "色彩整体分布相对均衡"

        summary = {
            "total_pairs": len(results),
            "mean_ciede2000_raw": round(float(np.mean(all_de00)), 3),
            "mean_ciede2000_aligned": round(float(np.mean(all_de00_al)), 3),
            "mean_db_raw": round(float(np.mean(all_db)), 5),
            "mean_db_aligned": round(float(np.mean(all_db_al)), 5),
            "avg_db_reduction_pct": round(float((np.mean(all_db) - np.mean(all_db_al)) / (np.mean(all_db) + 1e-10) * 100.0), 1),
            "mean_w1_raw": round(float(np.mean(all_w1)), 3),
            "mean_w1_aligned": round(float(np.mean(all_w1_al)), 3),
            "avg_w1_reduction_pct": round(float((np.mean(all_w1) - np.mean(all_w1_al)) / (np.mean(all_w1) + 1e-10) * 100.0), 1),
            "mean_psnr_raw": round(float(np.mean(all_psnr)), 2),
            "mean_psnr_aligned": round(float(np.mean(all_psnr_al)), 2),
            "systematic_bias": {
                "avg_delta_L": round(mean_dL, 3),
                "avg_delta_a": round(mean_da, 3),
                "avg_delta_b": round(mean_db, 3),
                "interpretation": f"系统性偏差诊断：{interpretation}。tag 基准则呈现高纯净肤色。"
            },
        }

        output_payload = {
            "session_id": session_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "items": results,
        }

        # Save JSON
        json_path = os.path.join(output_dir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        # Create ZIP archive of aligned images
        zip_path = None
        if create_zip and save_aligned:
            zip_path = os.path.join(output_dir, "aligned_src.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fname, _, _ in pairs:
                    src_file = os.path.join(aligned_dir, fname)
                    if os.path.exists(src_file):
                        zf.write(src_file, arcname=fname)

        return {
            "summary": summary,
            "items": results,
            "json_path": json_path,
            "zip_path": zip_path,
            "duration": round(time.time() - t0, 2),
        }


def process_image_pairs(
    src_dir: str,
    tag_dir: str,
    output_dir: str,
    session_id: Optional[str] = None,
    mode: str = "admh",
    create_zip: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """Convenience wrapper for ColorPipeline."""
    pipeline = ColorPipeline(mode=mode)
    return pipeline.process(
        src_dir=src_dir,
        tag_dir=tag_dir,
        output_dir=output_dir,
        session_id=session_id,
        save_aligned=True,
        create_zip=create_zip,
        progress_cb=progress_cb,
    )
