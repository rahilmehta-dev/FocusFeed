#!/usr/bin/env python3
"""Generate AppIcon.iconset for FocusFeed.app.
Called by build_app.sh — requires Pillow.
"""
import os
from PIL import Image, ImageDraw

ICONSET = "AppIcon.iconset"
os.makedirs(ICONSET, exist_ok=True)

# Sizes required by macOS iconset
SIZES = [16, 32, 128, 256, 512]


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # ── Gradient background: purple → blue (top → bottom) ──
    grad = Image.new("RGBA", (size, size))
    gd   = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / max(size - 1, 1)
        r = int(124 + (79  - 124) * t)   # 7c → 4f
        g = int(106 + (168 - 106) * t)   # 6a → a8
        b = int(247 + (255 - 247) * t)   # f7 → ff
        gd.line([(0, y), (size - 1, y)], fill=(r, g, b, 255))

    # ── Rounded-rect mask (macOS-style, ~22 % radius) ──
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=size // 4, fill=255
    )
    grad.putalpha(mask)
    img.paste(grad, (0, 0), grad)

    # ── White ▶ play triangle ──
    draw = ImageDraw.Draw(img)
    m = size * 0.24
    cx, cy = size / 2, size / 2
    pts = [
        (cx - m * 0.52, cy - m),
        (cx - m * 0.52, cy + m),
        (cx + m * 0.88, cy),
    ]
    draw.polygon(pts, fill=(255, 255, 255, 215))

    return img


for s in SIZES:
    make_icon(s).save(f"{ICONSET}/icon_{s}x{s}.png")
    make_icon(s * 2).save(f"{ICONSET}/icon_{s}x{s}@2x.png")

print(f"✓  {ICONSET}/ ready ({len(SIZES) * 2} files)")
