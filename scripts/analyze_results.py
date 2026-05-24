import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, "output", "json")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")

# Ngưỡng chất lượng VMAF chuẩn
VMAF_EXCELLENT = 93
VMAF_GOOD = 75
VMAF_FAIR = 50


def get_quality_label(score):
    if score >= VMAF_EXCELLENT:
        return "Excellent", "#2ecc71"
    elif score >= VMAF_GOOD:
        return "Good", "#f39c12"
    elif score >= VMAF_FAIR:
        return "Fair", "#e67e22"
    else:
        return "Poor", "#e74c3c"


def parse_vmaf_json(json_path):
    """Đọc file JSON và trả về frame scores + pooled metrics + sub-metrics."""
    with open(json_path, "r") as f:
        data = json.load(f)

    frames_data = data.get("frames", [])

    # Frame-level VMAF scores
    frame_scores = [f["metrics"]["vmaf"] for f in frames_data]

    # Pooled metrics (mean của toàn video)
    pooled = data.get("pooled_metrics", {})
    mean_vmaf = pooled.get("vmaf", {}).get("mean", pd.Series(frame_scores).mean())

    # Sub-metrics để vẽ heatmap
    sub_metrics = {
        "adm2": pooled.get("integer_adm2", {}).get("mean", None),
        "vif_scale0": pooled.get("integer_vif_scale0", {}).get("mean", None),
        "vif_scale1": pooled.get("integer_vif_scale1", {}).get("mean", None),
        "vif_scale2": pooled.get("integer_vif_scale2", {}).get("mean", None),
        "vif_scale3": pooled.get("integer_vif_scale3", {}).get("mean", None),
        "motion": pooled.get("integer_motion2", {}).get("mean", None),
    }

    return frame_scores, mean_vmaf, sub_metrics


def plot_qoe_over_time(all_data, plot_dir):
    """Biểu đồ đường biến thiên VMAF theo frame."""
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.set_theme(style="whitegrid")

    for config_name, (frames, mean_vmaf, _) in all_data.items():
        ax.plot(frames, label=f"{config_name} (Mean: {round(mean_vmaf, 1)})", alpha=0.85, linewidth=2)

    # Vẽ đường ngưỡng
    ax.axhline(y=VMAF_EXCELLENT, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7, label=f"Excellent ({VMAF_EXCELLENT})")
    ax.axhline(y=VMAF_GOOD, color="#f39c12", linestyle="--", linewidth=1, alpha=0.7, label=f"Good ({VMAF_GOOD})")

    ax.set_title("Biến Thiên Điểm Số QoE (VMAF) Theo Thời Gian Khung Hình", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("VMAF Score (0 - 100)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=9)

    path = os.path.join(plot_dir, "qoe_variation_over_time.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_bar_comparison(all_data, plot_dir):
    """Bar chart so sánh VMAF Mean/Min/Max giữa các cấu hình."""
    configs = list(all_data.keys())
    means = [all_data[c][1] for c in configs]
    mins = [min(all_data[c][0]) for c in configs]
    maxs = [max(all_data[c][0]) for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_mean = ax.bar(x - width, means, width, label="Mean", color="#3498db", alpha=0.9)
    bars_min = ax.bar(x, mins, width, label="Min", color="#e74c3c", alpha=0.9)
    bars_max = ax.bar(x + width, maxs, width, label="Max", color="#2ecc71", alpha=0.9)

    # Gán nhãn số lên từng cột
    for bars in [bars_mean, bars_min, bars_max]:
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )

    ax.axhline(y=VMAF_EXCELLENT, color="#27ae60", linestyle="--", linewidth=1, alpha=0.6, label=f"Excellent threshold ({VMAF_EXCELLENT})")
    ax.set_title("So Sánh VMAF Mean / Min / Max Giữa Các Cấu Hình Nén", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("VMAF Score")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)

    path = os.path.join(plot_dir, "bar_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_boxplot(all_data, plot_dir):
    """Box plot phân phối điểm VMAF - thể hiện độ ổn định."""
    fig, ax = plt.subplots(figsize=(10, 5))

    data_list = [all_data[c][0] for c in all_data]
    labels = list(all_data.keys())
    means = [all_data[c][1] for c in all_data]

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, notch=False, vert=True)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Đánh dấu mean
    for i, mean in enumerate(means, start=1):
        ax.plot(i, mean, marker="D", color="black", markersize=6, zorder=5, label="Mean" if i == 1 else "")

    ax.axhline(y=VMAF_EXCELLENT, color="#27ae60", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Phân Phối Điểm VMAF Theo Cấu Hình Nén (Box Plot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("VMAF Score")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)

    path = os.path.join(plot_dir, "boxplot_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def plot_submetrics_heatmap(all_data, plot_dir):
    """Heatmap tương quan sub-metrics (ADM, VIF) giữa các cấu hình."""
    records = []
    for config_name, (_, _, sub_metrics) in all_data.items():
        row = {"Config": config_name}
        row.update(sub_metrics)
        records.append(row)

    df = pd.DataFrame(records).set_index("Config")
    df = df.dropna(axis=1)  # Bỏ cột nào bị None

    if df.empty:
        print("[WARNING] Không đủ sub-metrics để vẽ heatmap.")
        return None

    fig, ax = plt.subplots(figsize=(11, max(3, len(df) * 1.2)))
    sns.heatmap(
        df,
        annot=True, fmt=".3f",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Score (0-1)"}
    )
    ax.set_title("Heatmap Sub-Metrics VMAF (ADM, VIF, Motion) Theo Cấu Hình", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sub-Metric")
    ax.set_ylabel("Cấu hình nén")

    path = os.path.join(plot_dir, "submetrics_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")
    return path


def build_summary_df(all_data):
    """Tạo DataFrame tổng hợp để in ra terminal và dùng trong Streamlit."""
    rows = []
    for config_name, (frames, mean_vmaf, _) in all_data.items():
        label, _ = get_quality_label(mean_vmaf)
        rows.append({
            "Configuration": config_name,
            "VMAF Mean": round(mean_vmaf, 2),
            "VMAF Min": round(min(frames), 2),
            "VMAF Max": round(max(frames), 2),
            "Std Dev": round(pd.Series(frames).std(), 2),
            "Quality": label,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    if not json_files:
        print("[ERROR] Không tìm thấy file JSON. Hãy chạy run_pipeline.py trước!")
        return

    all_data = {}
    for json_file in json_files:
        config_name = os.path.basename(json_file).replace("vmaf_", "").replace(".json", "")
        try:
            frames, mean_vmaf, sub_metrics = parse_vmaf_json(json_file)
            if frames:
                all_data[config_name] = (frames, mean_vmaf, sub_metrics)
        except Exception as e:
            print(f"[ERROR] Không đọc được {json_file}: {e}")

    if not all_data:
        print("[ERROR] Không có dữ liệu hợp lệ.")
        return

    print(f"[INFO] Đang phân tích {len(all_data)} cấu hình...")

    plot_qoe_over_time(all_data, PLOT_DIR)
    plot_bar_comparison(all_data, PLOT_DIR)
    plot_boxplot(all_data, PLOT_DIR)
    plot_submetrics_heatmap(all_data, PLOT_DIR)

    df = build_summary_df(all_data)
    print("\n" + "="*65)
    print("          BẢNG THỐNG KÊ SO SÁNH CHẤT LƯỢNG QoE")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65)


if __name__ == "__main__":
    main()