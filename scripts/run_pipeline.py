import os
import subprocess
import json
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_VIDEO = os.path.join(BASE_DIR, "data", "reference", "reference.mp4")
DISTORTED_DIR = os.path.join(BASE_DIR, "data", "distorted")
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "json")


def run_vmaf(args):
    """
    Hàm xử lý VMAF cho 1 video - thiết kế để chạy song song.
    Nhận tuple (ref_path, dist_path, json_out_path) vì ProcessPoolExecutor
    chỉ truyền được 1 argument.
    """
    ref_path, dist_path, json_out_path = args
    video_name = os.path.basename(dist_path)

    print(f"[START] {video_name} (PID: {os.getpid()})")
    start_time = time.time()

    filter_spec = (
        "[0:v][1:v]scale2ref=w=iw:h=ih[dist_scaled][ref_scaled];"
        f"[dist_scaled][ref_scaled]libvmaf=log_fmt=json:log_path={json_out_path}"
    )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", dist_path,
        "-i", ref_path,
        "-filter_complex", filter_spec,
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        elapsed = round(time.time() - start_time, 2)
        print(f"[SUCCESS] {video_name} - Hoàn thành trong {elapsed}s")
        return {"video": video_name, "status": "success", "time": elapsed, "output": json_out_path}

    except subprocess.CalledProcessError as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"[ERROR] {video_name} - Thất bại sau {elapsed}s")
        print(e.stderr.decode("utf-8", errors="replace"))
        return {"video": video_name, "status": "error", "time": elapsed, "output": None}


def run_pipeline_parallel(ref_video, distorted_videos, json_output_dir, max_workers=None):
    """
    Chạy VMAF song song cho toàn bộ danh sách video.
    max_workers=None sẽ tự động dùng số CPU cores của máy.
    Trả về list kết quả để app.py có thể đọc.
    """
    os.makedirs(json_output_dir, exist_ok=True)

    # Chuẩn bị danh sách args cho từng worker
    task_args = []
    for dist_video in distorted_videos:
        video_name = os.path.splitext(os.path.basename(dist_video))[0]
        output_json = os.path.join(json_output_dir, f"vmaf_{video_name}.json")
        task_args.append((ref_video, dist_video, output_json))

    total = len(task_args)
    results = []

    print(f"\n[INFO] Bắt đầu xử lý {total} video song song (max_workers={max_workers or 'auto'})")
    pipeline_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_vmaf, args): args for args in task_args}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done = len(results)
            print(f"[PROGRESS] {done}/{total} hoàn thành")

    total_elapsed = round(time.time() - pipeline_start, 2)
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n[DONE] {success_count}/{total} video thành công - Tổng thời gian: {total_elapsed}s")

    return results


def main():
    if not os.path.exists(REF_VIDEO):
        print(f"[CRITICAL] Không tìm thấy video gốc tại: {REF_VIDEO}")
        return

    distorted_videos = glob.glob(os.path.join(DISTORTED_DIR, "*.mp4"))
    if not distorted_videos:
        print(f"[WARNING] Không tìm thấy video nén nào trong: {DISTORTED_DIR}")
        return

    print(f"[INFO] Tìm thấy {len(distorted_videos)} video cần đánh giá.")

    # So sánh tuần tự vs song song để demo performance improvement
    print("\n" + "="*60)
    print("  CHẾ ĐỘ: SONG SONG (Parallel Processing)")
    print("="*60)

    results = run_pipeline_parallel(
        ref_video=REF_VIDEO,
        distorted_videos=distorted_videos,
        json_output_dir=JSON_OUTPUT_DIR,
        max_workers=None  # Tự động theo số CPU
    )

    # In bảng tóm tắt
    print("\n" + "="*60)
    print("  KẾT QUẢ PIPELINE")
    print("="*60)
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['video']} | {r['status']} | {r['time']}s")
    print("="*60)


if __name__ == "__main__":
    main()