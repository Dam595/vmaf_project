import streamlit as st
import os
import json
import glob
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="VMAF QoE Analyzer",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF_VIDEO = os.path.join(BASE_DIR, "data", "reference", "reference.mp4")
DISTORTED_DIR = os.path.join(BASE_DIR, "data", "distorted")
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "json")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
UPLOAD_TEMP_DIR = os.path.join(BASE_DIR, "output", "temp_uploads")

for d in [JSON_OUTPUT_DIR, PLOT_DIR, UPLOAD_TEMP_DIR, DISTORTED_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------------------
# QUALITY THRESHOLDS
# -----------------------------------------------------------------------
VMAF_EXCELLENT = 93
VMAF_GOOD = 75
VMAF_FAIR = 50

PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#34495e"]


def get_quality_label(score):
    if score >= VMAF_EXCELLENT:
        return "Excellent", "#2ecc71"
    elif score >= VMAF_GOOD:
        return "Good", "#f39c12"
    elif score >= VMAF_FAIR:
        return "Fair", "#e67e22"
    else:
        return "Poor", "#e74c3c"


# -----------------------------------------------------------------------
# ENCODE PRESETS
# -----------------------------------------------------------------------
ENCODE_PRESETS = {
    "H.264 — 500kbps": {
        "filename": "h264_500kbps.mp4",
        "vf": None,
        "codec_args": ["-c:v", "libx264", "-b:v", "500k", "-an"]
    },
    "H.264 — 1Mbps": {
        "filename": "h264_1mbps.mp4",
        "vf": None,
        "codec_args": ["-c:v", "libx264", "-b:v", "1000k", "-an"]
    },
    "H.264 — 2Mbps": {
        "filename": "h264_2mbps.mp4",
        "vf": None,
        "codec_args": ["-c:v", "libx264", "-b:v", "2000k", "-an"]
    },
    "H.265 — 500kbps": {
        "filename": "h265_500kbps.mp4",
        "vf": None,
        "codec_args": ["-c:v", "libx265", "-b:v", "500k", "-an", "-tag:v", "hvc1"]
    },
    "H.265 — 1Mbps": {
        "filename": "h265_1mbps.mp4",
        "vf": None,
        "codec_args": ["-c:v", "libx265", "-b:v", "1000k", "-an", "-tag:v", "hvc1"]
    },
    "VP9 — 1Mbps": {
        "filename": "vp9_1mbps.mp4",
        "vf": None,
        "codec_args": ["-c:v", "libvpx-vp9", "-b:v", "1000k", "-an"]
    },
    "H.264 — 500kbps + Blur": {
        "filename": "h264_500k_blur.mp4",
        "vf": "boxblur=2:1",
        "codec_args": ["-c:v", "libx264", "-b:v", "500k", "-an"]
    },
    "H.264 — 1Mbps + 15fps": {
        "filename": "h264_1mbps_15fps.mp4",
        "vf": "fps=15",
        "codec_args": ["-c:v", "libx264", "-b:v", "1000k", "-an"]
    },
}


def _encode_worker(args):
    ref_path, output_path, vf, codec_args = args
    cmd = ["ffmpeg", "-y", "-i", ref_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += codec_args
    cmd.append(output_path)
    start = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return {"name": os.path.basename(output_path), "status": "success", "time": round(time.time() - start, 2)}
    except subprocess.CalledProcessError as e:
        return {"name": os.path.basename(output_path), "status": "error",
                "time": round(time.time() - start, 2),
                "error": e.stderr.decode("utf-8", errors="replace")[:300]}


# -----------------------------------------------------------------------
# VMAF WORKER
# -----------------------------------------------------------------------
def _vmaf_worker(args):
    ref_path, dist_path, json_out_path = args
    video_name = os.path.basename(dist_path)
    filter_spec = (
        "[0:v][1:v]scale2ref=w=iw:h=ih[dist_scaled][ref_scaled];"
        f"[dist_scaled][ref_scaled]libvmaf=log_fmt=json:log_path={json_out_path}"
    )
    cmd = ["ffmpeg", "-y", "-i", dist_path, "-i", ref_path,
           "-filter_complex", filter_spec, "-f", "null", "-"]
    start = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return {"video": video_name, "status": "success",
                "time": round(time.time() - start, 2), "output": json_out_path}
    except subprocess.CalledProcessError as e:
        return {"video": video_name, "status": "error",
                "time": round(time.time() - start, 2),
                "error_msg": e.stderr.decode("utf-8", errors="replace")}


# -----------------------------------------------------------------------
# PARSE JSON
# -----------------------------------------------------------------------
def parse_vmaf_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    frames_data = data.get("frames", [])
    frame_scores = [fr["metrics"]["vmaf"] for fr in frames_data]
    pooled = data.get("pooled_metrics", {})
    mean_vmaf = pooled.get("vmaf", {}).get("mean", float(np.mean(frame_scores)))
    sub_metrics = {
        "ADM2":   pooled.get("integer_adm2", {}).get("mean"),
        "VIF_s0": pooled.get("integer_vif_scale0", {}).get("mean"),
        "VIF_s1": pooled.get("integer_vif_scale1", {}).get("mean"),
        "VIF_s2": pooled.get("integer_vif_scale2", {}).get("mean"),
        "VIF_s3": pooled.get("integer_vif_scale3", {}).get("mean"),
        "Motion": pooled.get("integer_motion2", {}).get("mean"),
    }
    return frame_scores, mean_vmaf, sub_metrics


def load_all_results(json_dir):
    all_data = {}
    for jf in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        name = os.path.basename(jf).replace("vmaf_", "").replace(".json", "")
        try:
            frames, mean, sub = parse_vmaf_json(jf)
            if frames:
                all_data[name] = {"frames": frames, "mean": mean, "sub": sub}
        except Exception:
            pass
    return all_data


# -----------------------------------------------------------------------
# CHARTS
# -----------------------------------------------------------------------
def fig_qoe_over_time(all_data):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (name, d) in enumerate(all_data.items()):
        ax.plot(d["frames"], label=f"{name} (Mean: {d['mean']:.1f})",
                color=PALETTE[i % len(PALETTE)], alpha=0.85, linewidth=1.8)
    ax.axhline(VMAF_EXCELLENT, color="#27ae60", linestyle="--", linewidth=1, alpha=0.6, label=f"Excellent >={VMAF_EXCELLENT}")
    ax.axhline(VMAF_GOOD, color="#f39c12", linestyle="--", linewidth=1, alpha=0.6, label=f"Good >={VMAF_GOOD}")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("VMAF Score (0-100)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_bar_comparison(all_data):
    configs = list(all_data.keys())
    means = [all_data[c]["mean"] for c in configs]
    mins  = [min(all_data[c]["frames"]) for c in configs]
    maxs  = [max(all_data[c]["frames"]) for c in configs]
    stds  = [float(np.std(all_data[c]["frames"])) for c in configs]

    x = np.arange(len(configs)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.4), 5))

    for offset, vals, label, color in [
        (-1.5*w, means, "Mean", "#3498db"),
        (-0.5*w, mins,  "Min",  "#e74c3c"),
        ( 0.5*w, maxs,  "Max",  "#2ecc71"),
        ( 1.5*w, stds,  "Std",  "#9b59b6"),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.9)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.axhline(VMAF_EXCELLENT, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("VMAF Score")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_boxplot(all_data):
    fig, ax = plt.subplots(figsize=(max(10, len(all_data) * 1.4), 5))
    data_list = [all_data[c]["frames"] for c in all_data]
    labels = list(all_data.keys())

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (name, d) in enumerate(all_data.items(), 1):
        ax.plot(i, d["mean"], marker="D", color="black", markersize=6,
                zorder=5, label="Mean" if i == 1 else "")

    ax.axhline(VMAF_EXCELLENT, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("VMAF Score")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_heatmap(all_data):
    records = []
    for name, d in all_data.items():
        row = {"Config": name}
        row.update({k: v for k, v in d["sub"].items() if v is not None})
        records.append(row)
    df = pd.DataFrame(records).set_index("Config")
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, max(2.5, len(df) * 1.0)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Score (0-1)"})
    ax.set_xlabel("Sub-Metric")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def build_summary_df(all_data):
    rows = []
    for name, d in all_data.items():
        label, _ = get_quality_label(d["mean"])
        rows.append({
            "Configuration": name,
            "VMAF Mean": round(d["mean"], 2),
            "VMAF Min": round(min(d["frames"]), 2),
            "VMAF Max": round(max(d["frames"]), 2),
            "Std Dev": round(float(np.std(d["frames"])), 2),
            "Quality": label,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------
st.markdown("""
<style>
    .section-divider { border-top: 1px solid #333; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("🎬 VMAF QoE Analyzer")
st.caption("Video Multimethod Assessment Fusion — Frame-level Quality Evaluation Pipeline")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline Configuration")

    st.subheader("Reference Video")
    ref_mode = st.radio("Source", ["Use default file", "Upload new"], horizontal=True)

    ref_video_path = None
    if ref_mode == "Use default file":
        if os.path.exists(DEFAULT_REF_VIDEO):
            st.success(f"✅ {os.path.basename(DEFAULT_REF_VIDEO)}")
            ref_video_path = DEFAULT_REF_VIDEO
        else:
            st.error("❌ File not found: data/reference/reference.mp4")
    else:
        ref_upload = st.file_uploader("Upload reference video", type=["mp4", "mkv", "avi", "mov"])
        if ref_upload:
            ref_tmp = os.path.join(UPLOAD_TEMP_DIR, f"ref_{ref_upload.name}")
            with open(ref_tmp, "wb") as f:
                f.write(ref_upload.read())
            ref_video_path = ref_tmp
            st.success(f"✅ {ref_upload.name}")

    st.markdown("---")

    st.subheader("Distorted Videos")
    dist_mode = st.radio("Source", ["Use default folder", "Upload new"], horizontal=True, key="dist_mode")

    dist_video_paths = []
    if dist_mode == "Use default folder":
        dist_video_paths = sorted(glob.glob(os.path.join(DISTORTED_DIR, "*.mp4")))
        st.info(f"Found {len(dist_video_paths)} video(s)")
        for p in dist_video_paths:
            st.caption(f"• {os.path.basename(p)}")
    else:
        dist_uploads = st.file_uploader(
            "Upload distorted videos (multiple files allowed)",
            type=["mp4", "mkv", "avi", "mov"],
            accept_multiple_files=True
        )
        if dist_uploads:
            for up in dist_uploads:
                tmp_path = os.path.join(UPLOAD_TEMP_DIR, f"dist_{up.name}")
                with open(tmp_path, "wb") as f:
                    f.write(up.read())
                dist_video_paths.append(tmp_path)
            st.success(f"✅ {len(dist_video_paths)} video(s) uploaded")

    st.markdown("---")

    st.subheader("Processing Options")
    max_workers = st.slider(
        "Parallel workers",
        min_value=1,
        max_value=os.cpu_count() or 4,
        value=min(max(len(dist_video_paths), 2), os.cpu_count() or 4),
        help="Number of videos processed simultaneously."
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run VMAF Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("🗑️ Clear Data")
    clear_distorted = st.checkbox("Delete distorted videos")
    clear_json = st.checkbox("Delete VMAF results (JSON)")
    clear_temp = st.checkbox("Delete temp uploads")

    if st.button("Clear Selected", type="secondary", use_container_width=True):
        cleared = []
        if clear_distorted:
            files = glob.glob(os.path.join(DISTORTED_DIR, "*.mp4"))
            for f in files:
                os.remove(f)
            cleared.append(f"{len(files)} distorted video(s)")
        if clear_json:
            files = glob.glob(os.path.join(JSON_OUTPUT_DIR, "*.json"))
            for f in files:
                os.remove(f)
            cleared.append(f"{len(files)} JSON result(s)")
        if clear_temp:
            files = glob.glob(os.path.join(UPLOAD_TEMP_DIR, "*"))
            for f in files:
                os.remove(f)
            cleared.append(f"{len(files)} temp file(s)")
        if cleared:
            st.success("Cleared: " + ", ".join(cleared))
            st.rerun()
        else:
            st.warning("Nothing selected to clear.")

# -----------------------------------------------------------------------
# MAIN TABS
# -----------------------------------------------------------------------
tab_encode, tab_vmaf, tab_results = st.tabs([
    "🎞️ Encode Distorted Videos",
    "⚡ VMAF Pipeline",
    "📊 Results & Analysis"
])

# -----------------------------------------------------------------------
# TAB 1: ENCODE
# -----------------------------------------------------------------------
with tab_encode:
    st.subheader("Generate Distorted Videos from Reference")
    st.markdown(
        "Select compression presets to encode. Videos will be saved to "
        "`data/distorted/` and used as input for the VMAF pipeline."
    )

    if not ref_video_path:
        st.warning("⚠️ Please select a reference video in the sidebar first.")
    else:
        st.info(f"Reference: `{os.path.basename(ref_video_path)}`")

        selected_presets = st.multiselect(
            "Select encode presets",
            options=list(ENCODE_PRESETS.keys()),
            default=list(ENCODE_PRESETS.keys()),
        )

        col_workers, col_btn = st.columns([2, 1])
        with col_workers:
            encode_workers = st.slider(
                "Parallel encode workers",
                min_value=1, max_value=os.cpu_count() or 4,
                value=min(len(selected_presets) if selected_presets else 2, os.cpu_count() or 4)
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            encode_btn = st.button("▶️ Start Encoding", type="primary", use_container_width=True)

        if encode_btn:
            if not selected_presets:
                st.error("Please select at least one preset.")
            else:
                prog = st.progress(0, text="Initializing...")
                log = st.empty()
                log_lines = []

                task_args = []
                for preset_name in selected_presets:
                    cfg = ENCODE_PRESETS[preset_name]
                    out_path = os.path.join(DISTORTED_DIR, cfg["filename"])
                    task_args.append((ref_video_path, out_path, cfg["vf"], cfg["codec_args"]))

                results = []
                encode_start = time.time()

                with ProcessPoolExecutor(max_workers=encode_workers) as executor:
                    futures = {executor.submit(_encode_worker, a): a for a in task_args}
                    for future in as_completed(futures):
                        r = future.result()
                        results.append(r)
                        icon = "✅" if r["status"] == "success" else "❌"
                        log_lines.append(f"{icon} `{r['name']}` — {r['time']}s")
                        log.markdown("\n\n".join(log_lines))
                        prog.progress(len(results) / len(task_args),
                                      text=f"{len(results)}/{len(task_args)} completed...")

                total_encode_time = round(time.time() - encode_start, 2)
                success_n = sum(1 for r in results if r["status"] == "success")

                if success_n == len(results):
                    st.success(f"🎉 Encoding complete — {success_n} videos in **{total_encode_time}s** ({encode_workers} workers)")
                else:
                    st.warning(f"⚠️ {success_n}/{len(results)} succeeded — {total_encode_time}s")

                df_enc = pd.DataFrame([
                    {"File": r["name"], "Time (s)": r["time"], "Status": r["status"]}
                    for r in results
                ])
                st.dataframe(df_enc, use_container_width=True, hide_index=True)
                st.info("Videos saved to `data/distorted/`. Switch to the **VMAF Pipeline** tab to score them.")

# -----------------------------------------------------------------------
# TAB 2: VMAF PIPELINE
# -----------------------------------------------------------------------
with tab_vmaf:
    st.subheader("Run VMAF Scoring Pipeline")

    if run_btn:
        if not ref_video_path:
            st.error("⚠️ Please select a reference video in the sidebar.")
        elif not dist_video_paths:
            st.error("⚠️ No distorted videos found. Please encode or upload videos first.")
        else:
            st.markdown(f"**Reference:** `{os.path.basename(ref_video_path)}` | **{len(dist_video_paths)} video(s)** | **{max_workers} worker(s)**")

            progress_bar = st.progress(0, text="Initializing...")
            log_box = st.empty()
            log_lines = []

            task_args = []
            for dist in dist_video_paths:
                name = os.path.splitext(os.path.basename(dist))[0]
                out_json = os.path.join(JSON_OUTPUT_DIR, f"vmaf_{name}.json")
                task_args.append((ref_video_path, dist, out_json))

            results = []
            pipeline_start = time.time()

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_vmaf_worker, a): a for a in task_args}
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    icon = "✅" if r["status"] == "success" else "❌"
                    log_lines.append(f"{icon} `{r['video']}` — {r['time']}s")
                    log_box.markdown("\n\n".join(log_lines))
                    progress_bar.progress(len(results) / len(task_args),
                                          text=f"{len(results)}/{len(task_args)} videos scored...")

            total_time = round(time.time() - pipeline_start, 2)
            success_count = sum(1 for r in results if r["status"] == "success")

            if success_count == len(results):
                st.success(f"🎉 Done! {success_count}/{len(results)} videos scored — Total: **{total_time}s**")
            else:
                st.warning(f"⚠️ {success_count}/{len(results)} succeeded — {total_time}s")

            with st.expander("📊 Parallel processing performance details"):
                df_perf = pd.DataFrame([
                    {"Video": r["video"], "Time (s)": r["time"], "Status": r["status"]}
                    for r in results
                ])
                st.dataframe(df_perf, use_container_width=True, hide_index=True)
                avg = round(sum(r["time"] for r in results) / len(results), 2)
                st.caption(
                    f"{len(results)} videos processed in parallel — "
                    f"Total: {total_time}s | Avg: {avg}s/video | Workers: {max_workers}"
                )

            st.balloons()
    else:
        st.info("👈 Click **'Run VMAF Analysis'** in the sidebar to start the pipeline.")

# -----------------------------------------------------------------------
# TAB 3: RESULTS
# -----------------------------------------------------------------------
with tab_results:
    st.subheader("QoE Results & Analysis")

    all_data = load_all_results(JSON_OUTPUT_DIR)

    if not all_data:
        st.info("ℹ️ No results yet. Run the VMAF Pipeline first.")
    else:
        # Metric cards
        st.markdown("**Quick Overview**")
        cols = st.columns(min(len(all_data), 4))
        for i, (name, d) in enumerate(all_data.items()):
            label, _ = get_quality_label(d["mean"])
            with cols[i % len(cols)]:
                st.metric(label=name, value=f"{d['mean']:.2f}", delta=label)
                st.caption(f"Min: {min(d['frames']):.1f} | Max: {max(d['frames']):.1f} | Std: {np.std(d['frames']):.2f}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Summary table
        st.markdown("**Detailed Comparison Table**")
        df_summary = build_summary_df(all_data)

        def style_quality(val):
            if val == "Excellent": return "background-color: #1a4731; color: #2ecc71; font-weight: bold"
            elif val == "Good": return "background-color: #3d2e00; color: #f39c12; font-weight: bold"
            elif val == "Fair": return "background-color: #3d1f00; color: #e67e22; font-weight: bold"
            else: return "background-color: #3d0000; color: #e74c3c; font-weight: bold"

        styled_df = df_summary.style.map(style_quality, subset=["Quality"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("VMAF >= 93: Excellent | >= 75: Good | >= 50: Fair | < 50: Poor — Score of 93+ is considered perceptually transparent to source.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Charts
        st.markdown("**Analysis Charts**")
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "📉 QoE Over Time",
            "📊 Bar Comparison",
            "📦 Box Plot",
            "🌡️ Sub-Metrics Heatmap"
        ])

        with chart_tab1:
            st.pyplot(fig_qoe_over_time(all_data))
            st.caption("Dashed lines indicate Excellent (93) and Good (75) quality thresholds.")

        with chart_tab2:
            st.pyplot(fig_bar_comparison(all_data))
            st.caption("Comparison of Mean, Min, Max, and Std Dev VMAF scores across compression configurations.")

        with chart_tab3:
            st.pyplot(fig_boxplot(all_data))
            st.caption("◆ = Mean value. A narrower box indicates more stable quality over time.")

        with chart_tab4:
            hm = fig_heatmap(all_data)
            if hm:
                st.pyplot(hm)
                st.caption("ADM2: detail fidelity | VIF (scale 0-3): visibility at different spatial frequencies | Motion: temporal complexity")
            else:
                st.info("Not enough sub-metrics to render heatmap.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Export
        st.markdown("**Export Results**")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download summary table (.csv)", csv_data,
                               "vmaf_summary.csv", "text/csv")
        with col2:
            all_json = {}
            for name, d in all_data.items():
                all_json[name] = {
                    "mean": d["mean"],
                    "min": min(d["frames"]),
                    "max": max(d["frames"]),
                    "std": float(np.std(d["frames"])),
                    "sub_metrics": d["sub"],
                }
            json_bytes = json.dumps(all_json, indent=2).encode("utf-8")
            st.download_button("⬇️ Download full results (.json)", json_bytes,
                               "vmaf_all_results.json", "application/json")