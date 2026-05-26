# VMAF QoE Analyzer

A full pipeline for evaluating video compression quality using **VMAF (Video Multimethod Assessment Fusion)**. Features a Streamlit web dashboard with parallel processing, multi-codec encoding, and interactive QoE visualizations.

---

## Features

- **Encode distorted videos** directly from the dashboard (H.264, H.265, VP9, blur, frame-drop presets)
- **Parallel VMAF scoring** using `ProcessPoolExecutor` — scores multiple videos simultaneously
- **4 visualization types**: QoE over time, bar comparison, box plot, sub-metrics heatmap
- **Upload your own videos** (reference + distorted) via the UI
- **Export results** as CSV or JSON
- Runs entirely on Ubuntu with FFmpeg + libvmaf

---

## Installation & Setup

### 1. System dependencies

FFmpeg must be compiled with `libvmaf` support. Verify with:

```bash
ffmpeg -filters 2>/dev/null | grep vmaf
```

If `libvmaf` is not listed, install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg
```

If your distro's FFmpeg does not include libvmaf, build from source or use a static build from [https://johnvansickle.com/ffmpeg/](https://johnvansickle.com/ffmpeg/).

### 2. Clone the repository

We recommend using a virtual environment (with Python 3.12.3):

```bash
git clone https://github.com/Dam595/vmaf_project
cd <project-folder>
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
vmaf_project/
├── app.py                    # Streamlit web dashboard
├── requirements.txt
├── data/
│   ├── reference/
│   │   └── reference.mp4     # Source video (high quality)
│   └── distorted/            # Compressed/distorted videos (auto-generated or uploaded)
├── output/
│   ├── json/                 # VMAF JSON results (frame-level scores)
│   └── plots/                # Saved chart images
└── scripts/
    ├── run_pipeline.py       # CLI: parallel VMAF scoring
    ├── analyze_results.py    # CLI: generate charts and stats table
    └── encode_distorted.py   # CLI: encode distorted videos from reference
```

---

## Quick Start

### 1. Prepare reference video

Place your source video at:

```
data/reference/reference.mp4
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Workflow inside the app:**

1. **Tab "Encode Distorted Videos"** — select compression presets and encode distorted videos
2. **Sidebar** — click **"Run VMAF Analysis"** to start the scoring pipeline
3. **Tab "Results & Analysis"** — view results, charts, and download data

---

## CLI Usage (without dashboard)

### Encode distorted videos

```bash
python scripts/encode_distorted.py
# or specify a custom reference:
python scripts/encode_distorted.py --ref /path/to/video.mp4 --workers 4
```

### Run VMAF scoring

```bash
python scripts/run_pipeline.py
```

### Analyze results and generate charts

```bash
python scripts/analyze_results.py
```

Charts are saved to `output/plots/`. Stats table is printed to terminal.

---

## Parallel Processing

The pipeline uses Python's `concurrent.futures.ProcessPoolExecutor` to score multiple videos simultaneously. The number of parallel workers is configurable via the sidebar slider or the `--workers` CLI argument.

Example speedup on a machine with 4 CPU cores, 8 distorted videos:

| Mode       | Time   |
|------------|--------|
| Sequential | ~16s   |
| Parallel (4 workers) | ~5s |

Actual speedup depends on video resolution, duration, and CPU count.

---

## Encode Presets

| Preset | Codec | Bitrate / Filter |
|--------|-------|-----------------|
| H.264 — 500kbps | libx264 | 500 kbps |
| H.264 — 1Mbps | libx264 | 1000 kbps |
| H.264 — 2Mbps | libx264 | 2000 kbps |
| H.265 — 500kbps | libx265 | 500 kbps |
| H.265 — 1Mbps | libx265 | 1000 kbps |
| VP9 — 1Mbps | libvpx-vp9 | 1000 kbps |
| H.264 — 500kbps + Blur | libx264 | 500 kbps + boxblur filter |
| H.264 — 1Mbps + 15fps | libx264 | 1000 kbps + fps=15 filter |

---

## VMAF Score Thresholds

| Score | Quality |
|-------|---------|
| >= 93 | Excellent — nearly indistinguishable from source |
| >= 75 | Good |
| >= 50 | Fair |
| < 50  | Poor |

---

## requirements.txt

```
streamlit
pandas
matplotlib
seaborn
numpy
```