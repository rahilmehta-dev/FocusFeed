# FocusFeed

AI YouTube video pre-filter for macOS (Apple Silicon). Pick a mood, get the one video worth watching — no doom-scrolling required.

## How it works

1. Pick a mood (need energy / feeling lazy / want to focus / need discipline)
2. App fetches top 10 motivational/productivity videos from YouTube
3. Each thumbnail + metadata is sent to a local MLX vision model
4. Model returns a watch score (1–10), verdict (watch/skip), one-line reason, and vibe tag
5. Results are streamed to the UI as each video is analyzed — first result appears in ~15 s

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- YouTube Data API v3 key (free tier: ~10,000 units/day)

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate focusfeed
```

This creates an isolated `focusfeed` environment with Python 3.11 and all dependencies.

To update an existing environment after pulling changes:

```bash
conda env update -f environment.yml --prune
```

> **Alternative (pip only):** If you prefer not to use conda:
> ```bash
> pip install -r requirements.txt
> ```

### 2. Get a YouTube API key

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create a project
2. Go to **APIs & Services → Library**, search for **YouTube Data API v3**, enable it
3. Go to **APIs & Services → Credentials → Create Credentials → API Key**
4. Copy the key

### 3. Configure

```bash
cp .env.example .env
# Edit .env and paste your YouTube API key
```

### 4. Run

**As a native macOS app (recommended):**
```bash
conda activate focusfeed
python app.py
```
Opens a native window — no browser needed.

**As a web server (alternative):**
```bash
conda activate focusfeed
python main.py
```
Then open **http://localhost:8000** in your browser.

## MLX model

| Model | Size | Notes |
|---|---|---|
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | ~4 GB | Default. Better reasoning. |
| `mlx-community/llava-1.5-7b-4bit` | ~4 GB | Faster first load. |

The model is auto-downloaded from HuggingFace on first run and cached at `~/.cache/huggingface/`.
First inference: ~30–60 s to load. Subsequent runs: ~5–15 s per video.

To switch models, set `MLX_MODEL` in `.env`:

```
MLX_MODEL=mlx-community/llava-1.5-7b-4bit
```

## Folder structure

```
focusfeed/
  main.py           FastAPI backend — YouTube API + MLX inference + SSE streaming
  index.html        Single-page dark UI
  environment.yml   Conda environment definition
  requirements.txt  pip fallback dependencies
  .env.example      Environment variable template
  README.md         This file
```
