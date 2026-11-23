#!/usr/bin/env python3
"""
Color Align & Evaluation Command-Line Tool.
Usage:
  python cli.py --src path/to/src_dir_or_zip --tag path/to/tag_dir_or_zip --output ./aligned_results
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile

from core.pipeline import process_image_pairs


def extract_if_zip(path: str, temp_parent: str, prefix: str) -> str:
    """Extracts path if it is a zip file, otherwise returns the directory."""
    if os.path.isfile(path) and (path.lower().endswith(".zip") or zipfile.is_zipfile(path)):
        extract_dir = os.path.join(temp_parent, prefix)
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {path} -> {extract_dir} ...")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir
    elif os.path.isdir(path):
        return path
    else:
        raise ValueError(f"Invalid path: {path} is neither a directory nor a zip file.")


def main():
    parser = argparse.ArgumentParser(
        description="Professional Color Difference Evaluation & Color Alignment System"
    )
    parser.add_argument("--src", required=True, help="Path to src image directory or src.zip")
    parser.add_argument("--tag", required=True, help="Path to tag image directory or tag.zip")
    parser.add_argument(
        "--output", default="./aligned_results", help="Output directory (default: ./aligned_results)"
    )
    parser.add_argument(
        "--mode",
        choices=["admh", "global_cdf", "reinhard"],
        default="admh",
        help="Color alignment algorithm (default: admh)",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create aligned_src.zip")

    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp(prefix="color_align_cli_")
    try:
        src_dir = extract_if_zip(args.src, temp_dir, "src")
        tag_dir = extract_if_zip(args.tag, temp_dir, "tag")

        print("=" * 70)
        print(" Professional Color Difference Evaluation & Color Alignment Pipeline")
        print("=" * 70)
        print(f" Source: {src_dir}")
        print(f" Target: {tag_dir}")
        print(f" Output: {args.output}")
        print(f" Mode:   {args.mode}")
        print("-" * 70)

        def progress_cb(cur, total, msg):
            pct = int(cur / total * 100)
            bar = ("#" * (pct // 4)).ljust(25, "-")
            sys.stdout.write(f"\r[{bar}] {pct}% - {msg}")
            sys.stdout.flush()

        res = process_image_pairs(
            src_dir=src_dir,
            tag_dir=tag_dir,
            output_dir=args.output,
            mode=args.mode,
            create_zip=not args.no_zip,
            progress_cb=progress_cb,
        )

        print("\n" + "=" * 70)
        print(" Processing Completed Successfully!")
        print("=" * 70)
        s = res["summary"]
        print(f" Total Image Pairs Processed:       {s['total_pairs']}")
        print(f" Mean CIEDE2000 (ΔE₀₀):             {s['mean_ciede2000_raw']} ➔ {s['mean_ciede2000_aligned']}")
        print(f" 3D Bhattacharyya Divergence:       {s['mean_db_raw']} ➔ {s['mean_db_aligned']} (-{s['avg_db_reduction_pct']}%)")
        print(f" Wasserstein Distance (W₁):         {s['mean_w1_raw']} ➔ {s['mean_w1_aligned']} (-{s['avg_w1_reduction_pct']}%)")
        print(f" Mean PSNR Fidelity:                {s['mean_psnr_raw']} dB ➔ {s['mean_psnr_aligned']} dB")
        print(f" Systematic Channel Biases:         ΔL*={s['systematic_bias']['avg_delta_L']}, Δa*={s['systematic_bias']['avg_delta_a']}, Δb*={s['systematic_bias']['avg_delta_b']}")
        print(f" Diagnosis:                         {s['systematic_bias']['interpretation']}")
        print("-" * 70)
        print(f" JSON Evaluation Report:            {res['json_path']}")
        if res.get("zip_path"):
            print(f" Aligned Image Archive:             {res['zip_path']}")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
