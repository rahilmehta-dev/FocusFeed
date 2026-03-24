"""FocusFeed — AI YouTube Video Pre-Filter (macOS / Apple Silicon)"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("focusfeed")

YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
MODEL_NAME: str = os.getenv("MLX_MODEL", "mlx-community/Qwen2-VL-7B-Instruct-4bit")
YT_BASE = "https://www.googleapis.com/youtube/v3"

# Lazy-loaded vision model (downloaded on first use, cached by HuggingFace)
_model = None
_processor = None

MOOD_MAP: dict[str, str] = {
    "need energy":    "high energy motivational pump up get moving",
    "feeling lazy":   "stop being lazy motivation discipline get up",
    "want to focus":  "deep focus flow state productivity concentration",
    "need discipline": "self discipline habits consistency success mindset",
}

app = FastAPI(title="FocusFeed")


# ── MLX vision inference ──────────────────────────────────────────────────────

def _load_model() -> tuple:
    global _model, _processor
    if _model is None:
        try:
            from mlx_vlm import load  # type: ignore
            log.info("Loading MLX model: %s  (first run auto-downloads ~4 GB)", MODEL_NAME)
            _model, _processor = load(MODEL_NAME)
            log.info("Model ready.")
        except ImportError as exc:
            raise RuntimeError("mlx_vlm not found. Run: pip install mlx-vlm") from exc
    return _model, _processor


def _infer(video: dict, thumb_path: str) -> dict:
    """Synchronous MLX vision inference — always called via asyncio.to_thread."""
    try:
        from mlx_vlm import generate  # type: ignore
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
        from mlx_vlm.utils import load_config  # type: ignore
    except ImportError:
        return _default_analysis()

    model, processor = _load_model()

    comments = "; ".join(video["comments"][:3]) or "none"
    prompt_text = (
        "You are a strict YouTube video curator. Evaluate this video before recommending it.\n\n"
        f"Title: {video['title']}\n"
        f"Channel: {video['channel']}\n"
        f"Views: {video['view_count']:,}   Likes: {video['like_count']:,}\n"
        f"Description: {video['description'][:300]}\n"
        f"Top comments: {comments}\n\n"
        "Look at the thumbnail image carefully:\n"
        "• Is it genuine content or pure clickbait?\n"
        "• Does the title match the likely actual content?\n"
        "• Do stats and comments confirm viewers found it valuable?\n\n"
        "Respond with ONLY a valid JSON object — no markdown, no explanation:\n"
        '{"score":<integer 1-10>,"verdict":"<watch|skip>",'
        '"reason":"<one sentence>","vibe":"<hype|calm|deep|intense>"}'
    )

    # Apply the model-specific chat template
    try:
        config = load_config(MODEL_NAME)
        prompt = apply_chat_template(processor, config, prompt_text, num_images=1)
    except Exception:
        prompt = prompt_text  # fallback: raw string

    # Generate — handle minor API version differences
    try:
        raw: str = generate(
            model, processor, prompt, [thumb_path],
            max_tokens=150, temp=0.1, verbose=False,
        )
    except TypeError:
        raw = generate(model, processor, prompt, thumb_path, max_tokens=150)  # type: ignore

    return _parse_analysis(str(raw))


def _parse_analysis(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            r = json.loads(raw[s:e])
            score = max(1, min(10, int(r.get("score", 5))))
            verdict = str(r.get("verdict", "")).lower()
            if verdict not in ("watch", "skip"):
                verdict = "watch" if score >= 6 else "skip"
            vibe = str(r.get("vibe", "")).lower()
            if vibe not in ("hype", "calm", "deep", "intense"):
                vibe = "calm"
            return {
                "score": score,
                "verdict": verdict,
                "reason": str(r.get("reason", "")).strip()[:200],
                "vibe": vibe,
            }
        except (json.JSONDecodeError, ValueError):
            pass
    log.warning("MLX parse failed. Raw output: %r", raw[:120])
    return _default_analysis()


def _default_analysis() -> dict:
    return {"score": 5, "verdict": "watch", "reason": "Analysis unavailable.", "vibe": "calm"}


# ── YouTube API ───────────────────────────────────────────────────────────────

async def _yt_search(mood: str) -> list[dict]:
    query = MOOD_MAP.get(mood, f"{mood} motivational productivity")
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(f"{YT_BASE}/search", params={
            "key": YOUTUBE_API_KEY, "q": query,
            "part": "snippet", "type": "video",
            "maxResults": 10, "videoDuration": "medium",
            "order": "relevance", "relevanceLanguage": "en",
        })
        r.raise_for_status()

        ids = [it["id"]["videoId"] for it in r.json().get("items", [])]
        if not ids:
            return []

        r2 = await c.get(f"{YT_BASE}/videos", params={
            "key": YOUTUBE_API_KEY, "id": ",".join(ids),
            "part": "statistics,snippet",
        })
        r2.raise_for_status()

        out: list[dict] = []
        for item in r2.json().get("items", []):
            vid = item["id"]
            sn = item["snippet"]
            st = item.get("statistics", {})
            th = sn.get("thumbnails", {})
            thumb = (
                th.get("maxres") or th.get("high") or th.get("medium") or {}
            ).get("url", "")

            comments = await _fetch_comments(c, vid)
            out.append({
                "id": vid,
                "title": sn["title"],
                "description": sn.get("description", "")[:500],
                "thumbnail_url": thumb,
                "channel": sn.get("channelTitle", ""),
                "view_count": int(st.get("viewCount", 0)),
                "like_count": int(st.get("likeCount", 0)),
                "comments": comments,
                "url": f"https://youtube.com/watch?v={vid}",
            })
        return out


async def _fetch_comments(c: httpx.AsyncClient, vid: str) -> list[str]:
    try:
        r = await c.get(f"{YT_BASE}/commentThreads", params={
            "key": YOUTUBE_API_KEY, "videoId": vid,
            "part": "snippet", "maxResults": 5, "order": "relevance",
        })
        r.raise_for_status()
        return [
            it["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for it in r.json().get("items", [])
        ]
    except Exception:
        return []


async def _dl_thumb(url: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(url)
        r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(r.content)
        return f.name


# ── Routes ────────────────────────────────────────────────────────────────────

class FeedRequest(BaseModel):
    mood: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.post("/api/feed")
async def api_feed(req: FeedRequest):
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="YOUTUBE_API_KEY not set. Copy .env.example → .env and add your key.",
        )

    async def stream() -> AsyncGenerator[bytes, None]:
        def evt(d: dict) -> bytes:
            return f"data: {json.dumps(d)}\n\n".encode()

        yield evt({"type": "status", "message": "Searching YouTube…"})

        try:
            videos = await _yt_search(req.mood)
        except Exception as exc:
            yield evt({"type": "error", "message": f"YouTube search failed: {exc}"})
            return

        if not videos:
            yield evt({"type": "error", "message": "No videos found — try a different mood."})
            return

        n = len(videos)
        yield evt({
            "type": "status",
            "message": f"Found {n} videos. Running AI analysis…",
            "total": n,
        })

        for i, video in enumerate(videos):
            short = video["title"][:55]
            yield evt({"type": "status", "message": f"[{i+1}/{n}] {short}…"})

            thumb_path = None
            try:
                thumb_path = await _dl_thumb(video["thumbnail_url"])
                analysis = await asyncio.to_thread(_infer, video, thumb_path)
            except Exception as exc:
                log.error("Analysis error for %s: %s", video["id"], exc)
                analysis = _default_analysis()
            finally:
                if thumb_path:
                    try:
                        os.unlink(thumb_path)
                    except OSError:
                        pass

            video["analysis"] = analysis
            video.pop("comments", None)  # trim payload
            yield evt({"type": "video", "video": video})

        yield evt({"type": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
