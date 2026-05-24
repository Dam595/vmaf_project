"""
encode_distorted.py
-------------------
Tự động encode nhiều cấu hình distorted từ video gốc.
Chạy một lần để tạo dataset, không cần chạy lại trừ khi đổi reference.

Usage:
    python scripts/encode_distorted.py
    python scripts/encode_distorted.py --ref data/reference/reference.mp4
"""

import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REF = os.path.join(BASE_DIR, "data", "reference", "reference.mp4")
DISTORTED_DIR = os.path.join(BASE_DIR, "data", "distorted")

# Danh sách cấu hình encode
# Mỗi entry: (tên file output, ffmpeg video filter hoặc None, ffmpeg codec args)
ENCODE_CONFIGS = [
    (
        "h264_150kbps.mp4",
        None,
        ["-c:v", "libx264", "-b:v", "150k", "-an"]
    ),
    (
        "h264_100kbps.mp4",
        None,
        ["-c:v", "libx264", "-b:v", "100k", "-an"]
    ),
    (
        "h264_50kbps.mp4",
        None,
        ["-c:v", "libx264", "-b:v", "50k", "-an"]
    ),
    (
        "h265_200kbps.mp4",
        None,
        ["-c:v", "libx265", "-b:v", "200k", "-an", "-tag:v", "hvc1"]
    ),
    (
        "h265_100kbps.mp4",
        None,
        ["-c:v", "libx265", "-b:v", "100k", "-an", "-tag:v", "hvc1"]
    ),
    (
        "vp9_150kbps.mp4",
        None,
        ["-c:v", "libvpx-vp9", "-b:v", "150k", "-an"]
    ),
    (
        "h264_blur.mp4",
        "boxblur=2:1",
        ["-c:v", "libx264", "-b:v", "200k", "-an"]
    ),
    (
        "h264_15fps.mp4",
        "fps=15",
        ["-c:v", "libx264", "-b:v", "200k", "-an"]
    ),
]


def encode_one(args):
    """Worker encode 1 cấu hình. Trả về dict kết quả."""
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

    task_args = []
    for filename, vf, codec_args in configs:
        out = os.path.join(distorted_dir, filename)
        task_args.append((ref_path, out, vf, codec_args))

    print(f"\n[INFO] Encoding {len(task_args)} cấu hình từ: {os.path.basename(ref_path)}")
    print(f"[INFO] Output: {distorted_dir}\n")

    total_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(encode_one, a): a for a in task_args}
        for future in as_completed(futures):
            results.append(future.result())

    total = round(time.time() - total_start, 2)
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n[DONE] {success}/{len(results)} thành công — Tổng: {total}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Encode distorted videos for VMAF testing")
    parser.add_argument("--ref", default=DEFAULT_REF, help="Đường dẫn video gốc")
    parser.add_argument("--workers", type=int, default=None, help="Số luồng song song")
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print(f"[ERROR] Không tìm thấy video gốc: {args.ref}")
        return

    run_encode(args.ref, DISTORTED_DIR, ENCODE_CONFIGS, max_workers=args.workers)


if __name__ == "__main__":
    main()