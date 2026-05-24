"""
encode_distorted.py
-------------------
Encode multiple distorted video configurations from a reference video.
Run once to generate the dataset; re-run only when the reference changes.

Usage:
    python scripts/encode_distorted.py
    python scripts/encode_distorted.py --ref data/reference/reference.mp4 --workers 4
"""

import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REF = os.path.join(BASE_DIR, "data", "reference", "reference.mp4")
DISTORTED_DIR = os.path.join(BASE_DIR, "data", "distorted")

# (output_filename, video_filter or None, codec_args)
ENCODE_CONFIGS = [
    ("h264_500kbps.mp4",   None,          ["-c:v", "libx264", "-b:v", "500k",  "-an"]),
    ("h264_1mbps.mp4",     None,          ["-c:v", "libx264", "-b:v", "1000k", "-an"]),
    ("h264_2mbps.mp4",     None,          ["-c:v", "libx264", "-b:v", "2000k", "-an"]),
    ("h265_500kbps.mp4",   None,          ["-c:v", "libx265", "-b:v", "500k",  "-an", "-tag:v", "hvc1"]),
    ("h265_1mbps.mp4",     None,          ["-c:v", "libx265", "-b:v", "1000k", "-an", "-tag:v", "hvc1"]),
    ("vp9_1mbps.mp4",      None,          ["-c:v", "libvpx-vp9", "-b:v", "1000k", "-an"]),
    ("h264_500k_blur.mp4", "boxblur=2:1", ["-c:v", "libx264", "-b:v", "500k",  "-an"]),
    ("h264_1mbps_15fps.mp4", "fps=15",   ["-c:v", "libx264", "-b:v", "1000k", "-an"]),
]


def encode_one(args):
    ref_path, output_path, vf, codec_args = args
    name = os.path.basename(output_path)

    cmd = ["ffmpeg", "-y", "-i", ref_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += codec_args
    cmd.append(output_path)

    start = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        elapsed = round(time.time() - start, 2)
        print(f"[OK] {name} ({elapsed}s)")
        return {"name": name, "status": "success", "time": elapsed}
    except subprocess.CalledProcessError as e:
        elapsed = round(time.time() - start, 2)
        print(f"[ERROR] {name}: {e.stderr.decode('utf-8', errors='replace')[:200]}")
        return {"name": name, "status": "error", "time": elapsed}


def run_encode(ref_path, distorted_dir, configs, max_workers=None):
    os.makedirs(distorted_dir, exist_ok=True)

    task_args = [
        (ref_path, os.path.join(distorted_dir, filename), vf, codec_args)
        for filename, vf, codec_args in configs
    ]

    print(f"\n[INFO] Encoding {len(task_args)} configs from: {os.path.basename(ref_path)}")
    print(f"[INFO] Output: {distorted_dir}\n")

    total_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(encode_one, a): a for a in task_args}
        for future in as_completed(futures):
            results.append(future.result())

    total = round(time.time() - total_start, 2)
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n[DONE] {success}/{len(results)} succeeded — Total: {total}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Encode distorted videos for VMAF testing")
    parser.add_argument("--ref", default=DEFAULT_REF, help="Path to reference video")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print(f"[ERROR] Reference video not found: {args.ref}")
        return

    run_encode(args.ref, DISTORTED_DIR, ENCODE_CONFIGS, max_workers=args.workers)


if __name__ == "__main__":
    main()