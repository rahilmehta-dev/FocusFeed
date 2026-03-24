"""FocusFeed — AI YouTube Video Pre-Filter (macOS / Apple Silicon)"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("focusfeed")

YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
MODEL_NAME: str = os.getenv("MLX_MODEL", "mlx-community/Qwen2-VL-7B-Instruct-4bit")
YT_BASE  = "https://www.googleapis.com/youtube/v3"
DB_PATH  = Path(__file__).parent / "focusfeed.db"

_model     = None
_processor = None
_load_log: list[tuple]  = []
_load_done = threading.Event()

MOOD_MAP: dict[str, str] = {
    "need energy":     "high energy motivational pump up get moving",
    "feeling lazy":    "stop being lazy motivation discipline get up",
    "want to focus":   "deep focus flow state productivity concentration",
    "need discipline": "self discipline habits consistency success mindset",
}


# ── Database ──────────────────────────────────────────────────────────────────

async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id            TEXT PRIMARY KEY,
                title         TEXT NOT NULL,
                channel       TEXT DEFAULT '',
                thumbnail_url TEXT DEFAULT '',
                url           TEXT DEFAULT '',
                view_count    INTEGER DEFAULT 0,
                like_count    INTEGER DEFAULT 0,
                score         INTEGER DEFAULT 5,
                verdict       TEXT DEFAULT 'watch',
                reason        TEXT DEFAULT '',
                vibe          TEXT DEFAULT 'calm',
                mood          TEXT DEFAULT '',
                suggested_at  TEXT DEFAULT (datetime('now')),
                liked         INTEGER DEFAULT 0,
                watched       INTEGER DEFAULT 0
            )
        """)
        await db.commit()


async def _save_video(video: dict, mood: str) -> dict:
    """Insert video (ignore if already exists to preserve liked/watched state).
    Returns the video's current liked/watched state."""
    a = video["analysis"]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR IGNORE INTO videos
                (id, title, channel, thumbnail_url, url, view_count, like_count,
                 score, verdict, reason, vibe, mood)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video["id"], video["title"], video.get("channel", ""),
            video.get("thumbnail_url", ""), video.get("url", ""),
            video.get("view_count", 0), video.get("like_count", 0),
            a["score"], a["verdict"], a["reason"], a["vibe"], mood,
        ))
        await db.commit()
        async with db.execute(
            "SELECT liked, watched FROM videos WHERE id = ?", (video["id"],)
        ) as cur:
            row = await cur.fetchone()
    return {"liked": bool(row[0]), "watched": bool(row[1])} if row else {"liked": False, "watched": False}


async def _get_user_preferences() -> str:
    """Build a preference context string from the user's liked/watched history."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT vibe, channel, score
            FROM videos
            WHERE liked = 1 OR watched = 1
            ORDER BY suggested_at DESC
            LIMIT 40
        """) as cur:
            rows = await cur.fetchall()

    if len(rows) < 3:
        return ""   # not enough signal yet

    vibes    = Counter(r[0] for r in rows if r[0])
    channels = Counter(r[1] for r in rows if r[1])
    avg_score = sum(r[2] for r in rows if r[2]) / len(rows)

    lines = ["User engagement history (calibrate scores accordingly):"]
    if vibes:
        top = ", ".join(f"{v} ({c}x)" for v, c in vibes.most_common(3))
        lines.append(f"• Preferred vibes: {top}")
    if channels:
        top = ", ".join(c for c, _ in channels.most_common(3))
        lines.append(f"• Channels they engage with: {top}")
    lines.append(f"• Avg score of content they liked: {avg_score:.1f}/10")
    lines.append("")
    return "\n".join(lines) + "\n"


# ── Model loading ─────────────────────────────────────────────────────────────

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _is_model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


def _hf_cached_bytes() -> int:
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    blobs = cache_dir / ("models--" + MODEL_NAME.replace("/", "--")) / "blobs"
    if not blobs.exists():
        return 0
    return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())


def _model_total_bytes() -> int:
    try:
        from huggingface_hub import model_info  # type: ignore
        info = model_info(MODEL_NAME)
        return sum(f.size or 0 for f in info.siblings)
    except Exception:
        return 4_670_000_000


def _download_progress_poller(total: int) -> None:
    last_pct = -1
    while not _load_done.is_set():
        downloaded = _hf_cached_bytes()
        if downloaded > 0 and total > 0:
            pct = min(99, int(downloaded / total * 100))
            if pct != last_pct:
                last_pct = pct
                _load_log.append((
                    "progress",
                    f"Downloading {_fmt_bytes(downloaded)} / {_fmt_bytes(total)}  ({pct}%)",
                    pct,
                ))
        time.sleep(2)


def _load_model_thread() -> None:
    global _model, _processor
    try:
        _load_log.append(("status", "Checking model cache…"))
        cached = _is_model_cached()

        if cached:
            _load_log.append(("status", f"Loading {MODEL_NAME} into memory…"))
        else:
            total = _model_total_bytes()
            _load_log.append((
                "status",
                f"Downloading {MODEL_NAME} ({_fmt_bytes(total)}) — first run only…",
            ))
            threading.Thread(target=_download_progress_poller, args=(total,), daemon=True).start()

        from mlx_vlm import load  # type: ignore
        _model, _processor = load(MODEL_NAME)
        log.info("MLX model ready.")
        _load_log.append(("ready", f"Model ready — {MODEL_NAME}"))
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        _load_log.append(("error", f"Failed to load model: {exc}"))
    finally:
        _load_done.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    threading.Thread(target=_load_model_thread, daemon=True).start()
    yield


app = FastAPI(title="FocusFeed", lifespan=lifespan)


# ── Model status endpoints ────────────────────────────────────────────────────

@app.get("/api/model/status")
async def model_status():
    return JSONResponse({"ready": _model is not None})


@app.get("/api/model/load")
async def model_load_stream():
    async def stream() -> AsyncGenerator[bytes, None]:
        def evt(d: dict) -> bytes:
            return f"data: {json.dumps(d)}\n\n".encode()

        if _model is not None:
            yield evt({"type": "ready", "message": f"Model ready — {MODEL_NAME}"})
            return

        sent = 0
        while True:
            while sent < len(_load_log):
                entry   = _load_log[sent]
                kind    = entry[0]
                payload = {"type": kind, "message": entry[1]}
                if len(entry) == 3:
                    payload["pct"] = entry[2]
                yield evt(payload)
                sent += 1
                if kind in ("ready", "error"):
                    return
            if _load_done.is_set():
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── MLX inference ─────────────────────────────────────────────────────────────

def _infer(video: dict, thumb_path: str, pref_context: str = "") -> dict:
    try:
        from mlx_vlm import generate  # type: ignore
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
        from mlx_vlm.utils import load_config  # type: ignore
    except ImportError:
        return _default_analysis()

    if _model is None:
        return _default_analysis()

    comments = "; ".join(video["comments"][:3]) or "none"
    prompt_text = (
        pref_context
        + "You are a strict YouTube video curator. Evaluate this video before recommending it.\n\n"
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

    try:
        config = load_config(MODEL_NAME)
        prompt = apply_chat_template(_processor, config, prompt_text, num_images=1)
    except Exception:
        prompt = prompt_text

    try:
        raw: str = generate(
            _model, _processor, prompt, [thumb_path],
            max_tokens=150, temp=0.1, verbose=False,
        )
    except TypeError:
        raw = generate(_model, _processor, prompt, thumb_path, max_tokens=150)  # type: ignore

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
                "score": score, "verdict": verdict,
                "reason": str(r.get("reason", "")).strip()[:200], "vibe": vibe,
            }
        except (json.JSONDecodeError, ValueError):
            pass
    log.warning("MLX parse failed. Raw: %r", raw[:120])
    return _default_analysis()


def _default_analysis() -> dict:
    return {"score": 5, "verdict": "watch", "reason": "Analysis unavailable.", "vibe": "calm"}


# ── YouTube API ───────────────────────────────────────────────────────────────

async def _yt_search(mood: str) -> list[dict]:
    query = MOOD_MAP.get(mood, f"{mood} motivational productivity")
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(f"{YT_BASE}/search", params={
            "key": YOUTUBE_API_KEY, "q": query, "part": "snippet",
            "type": "video", "maxResults": 10, "videoDuration": "medium",
            "order": "relevance", "relevanceLanguage": "en",
        })
        r.raise_for_status()

        ids = [it["id"]["videoId"] for it in r.json().get("items", [])]
        if not ids:
            return []

        r2 = await c.get(f"{YT_BASE}/videos", params={
            "key": YOUTUBE_API_KEY, "id": ",".join(ids), "part": "statistics,snippet",
        })
        r2.raise_for_status()

        out: list[dict] = []
        for item in r2.json().get("items", []):
            vid = item["id"]
            sn  = item["snippet"]
            st  = item.get("statistics", {})
            th  = sn.get("thumbnails", {})
            thumb = (th.get("maxres") or th.get("high") or th.get("medium") or {}).get("url", "")
            out.append({
                "id": vid, "title": sn["title"],
                "description": sn.get("description", "")[:500],
                "thumbnail_url": thumb,
                "channel": sn.get("channelTitle", ""),
                "view_count": int(st.get("viewCount", 0)),
                "like_count": int(st.get("likeCount", 0)),
                "comments": await _fetch_comments(c, vid),
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
        raise HTTPException(500, "YOUTUBE_API_KEY not set. Copy .env.example → .env and add your key.")
    if _model is None:
        raise HTTPException(503, "Model is still loading. Please wait.")

    pref_context = await _get_user_preferences()

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
        yield evt({"type": "status", "message": f"Found {n} videos. Running AI analysis…", "total": n})

        for i, video in enumerate(videos):
            yield evt({"type": "status", "message": f"[{i+1}/{n}] {video['title'][:55]}…"})

            thumb_path = None
            try:
                thumb_path = await _dl_thumb(video["thumbnail_url"])
                analysis  = await asyncio.to_thread(_infer, video, thumb_path, pref_context)
            except Exception as exc:
                log.error("Analysis error %s: %s", video["id"], exc)
                analysis = _default_analysis()
            finally:
                if thumb_path:
                    try:
                        os.unlink(thumb_path)
                    except OSError:
                        pass

            video["analysis"] = analysis
            state = await _save_video(video, req.mood)   # auto-save; returns liked/watched
            video["liked"]   = state["liked"]
            video["watched"] = state["watched"]
            video.pop("comments", None)
            yield evt({"type": "video", "video": video})

        yield evt({"type": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── History & reactions ───────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, title, channel, thumbnail_url, url,
                   view_count, like_count, score, verdict, reason,
                   vibe, mood, suggested_at, liked, watched
            FROM videos
            ORDER BY suggested_at DESC
            LIMIT 300
        """) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


@app.post("/api/videos/{vid}/like")
async def toggle_like(vid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE videos SET liked = NOT liked WHERE id = ?", (vid,))
        await db.commit()
        async with db.execute("SELECT liked FROM videos WHERE id = ?", (vid,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "Video not found in history")
    return {"liked": bool(row[0])}


@app.post("/api/videos/{vid}/watched")
async def toggle_watched(vid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE videos SET watched = NOT watched WHERE id = ?", (vid,))
        await db.commit()
        async with db.execute("SELECT watched FROM videos WHERE id = ?", (vid,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "Video not found in history")
    return {"watched": bool(row[0])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
