#!/usr/bin/env python3
"""Make the README images in docs/assets/ (needs matplotlib and Pillow).

    python3 docs/assets/make_assets.py [path/to/banner-source.jpg]

Every number below is copied from docs/BENCHMARKS.md (measured on one machine, greedy decoding).
Nothing here is estimated. If you re-measure, change BENCHMARKS.md first, then these constants.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# --- docs/BENCHMARKS.md, "Decode, real prompts (CLI)" and "Decode without speculation" -----------------
LLAMA_DECODE = 14.300132          # llama.cpp b11806 llama-server, same 6 prompts, median
KRAIT_NOSPEC = 20.679             # Krait without speculation (Phase 3 build), median of 18
KRAIT_MTP = 23.01585              # Krait with MTP speculation k = 1, median of 18
KRAIT_MTP_PROMPTS = (22.5289, 23.6092)   # lowest / highest per-prompt median

# --- docs/BENCHMARKS.md, "Prompt reading (synthetic prompts, median of 3)" -------------------------------
TTFT_LABELS = ["1K tokens", "4K tokens", "32K tokens"]
TTFT_LLAMA = [5.780, 20.103, 156.849]
TTFT_KRAIT = [2.250, 8.027, 63.378]

HARDWARE = ("Measured on one RTX 5090 32 GB (engine capped at 29,346 MiB VRAM), Ryzen 9 7950X3D, "
            "WSL2 Ubuntu 24.04, CUDA 12.9.\nModel: Unsloth Nemotron 3.5 Super 120B-A12B UD-Q4_K_XL. "
            "Greedy decoding (temperature 0). Source: docs/BENCHMARKS.md")

BG = "#05070a"
PANEL = "#0b1116"
GRID = "#1c2730"
TEXT = "#e8f0ea"
MUTED = "#8a9a92"
GREEN = "#5cf03a"
GREEN_DIM = "#2f9a2a"
TEAL = "#26c6c6"
GREY = "#5b6670"


def style(ax):
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=12)


def decode_chart(path):
    fig, ax = plt.subplots(figsize=(12.8, 6.0), dpi=100)
    fig.patch.set_facecolor(BG)
    style(ax)
    names = ["llama.cpp b11806", "Krait\n(no speculation)", "Krait\n(MTP speculation)"]
    vals = [LLAMA_DECODE, KRAIT_NOSPEC, KRAIT_MTP]
    colors = [GREY, GREEN_DIM, GREEN]
    y = range(len(vals))
    bars = ax.barh(list(y), vals, color=colors, height=0.58, zorder=3)
    lo, hi = KRAIT_MTP_PROMPTS
    ax.errorbar([KRAIT_MTP], [2], xerr=[[KRAIT_MTP - lo], [hi - KRAIT_MTP]], fmt="none",
                ecolor=TEXT, elinewidth=1.4, capsize=5, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=TEXT, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 34)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_xlabel("decode speed, tokens per second (higher is better)", color=MUTED, fontsize=12)
    for b, v in zip(bars, vals):
        label = f"{v:.2f} tok/s"
        if b is bars[2]:
            label += f"   {v / LLAMA_DECODE:.2f}x llama.cpp"
        ax.text(v + (1.3 if b is bars[2] else 0.4), b.get_y() + b.get_height() / 2, label,
                va="center", color=TEXT, fontsize=14, fontweight="bold", zorder=5)
    fig.suptitle("Nemotron 3.5 Super 120B-A12B decode on one 32 GB GPU (greedy)", color=TEXT,
                 fontsize=18, fontweight="bold", x=0.02, ha="left", y=0.965)
    ax.set_title("median of 6 real chat prompts x 3 runs, 256 tokens each; whisker = range of per-prompt medians.\n"
                 "No-speculation bar: an earlier build, measured in its own session (llama.cpp 14.32 tok/s there).",
                 color=MUTED, fontsize=11.5, loc="left", pad=10)
    fig.text(0.02, 0.025, HARDWARE, color=MUTED, fontsize=9.5, ha="left", va="bottom")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.25)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def ttft_chart(path):
    fig, ax = plt.subplots(figsize=(12.8, 6.0), dpi=100)
    fig.patch.set_facecolor(BG)
    style(ax)
    x = list(range(len(TTFT_LABELS)))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], TTFT_LLAMA, w, color=GREY, label="llama.cpp b11806", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], TTFT_KRAIT, w, color=GREEN, label="Krait", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, 400)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["1 s", "2 s", "5 s", "10 s", "20 s", "50 s", "100 s", "200 s"])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(TTFT_LABELS, color=TEXT, fontsize=13)
    ax.set_ylabel("time to first token (log scale, lower is better)", color=MUTED, fontsize=12)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.1f} s" if v >= 10 else f"{v:.2f} s",
                    ha="center", va="bottom", color=TEXT, fontsize=12.5, fontweight="bold", zorder=5)
    for i, (a, k) in enumerate(zip(TTFT_LLAMA, TTFT_KRAIT)):
        ax.text(i, 330, f"{a / k:.2f}x faster", ha="center", color=TEAL, fontsize=13, fontweight="bold")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=12, bbox_to_anchor=(0.0, 0.88))
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.suptitle("Time to first token: reading a long prompt (greedy)", color=TEXT, fontsize=18,
                 fontweight="bold", x=0.02, ha="left", y=0.965)
    ax.set_title("synthetic prompts, median of 3 runs (llama.cpp's 32K prompt: 32,512 tokens)", color=MUTED,
                 fontsize=11.5, loc="left", pad=10)
    fig.text(0.02, 0.025, HARDWARE, color=MUTED, fontsize=9.5, ha="left", va="bottom")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.25)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def images(src):
    im = Image.open(src).convert("RGB")
    assert im.size == (1280, 720), im.size
    im.save(os.path.join(HERE, "banner.jpg"), "JPEG", quality=84, optimize=True, progressive=True)
    top = (720 - 640) // 2
    im.crop((0, top, 1280, top + 640)).save(os.path.join(HERE, "social-preview.png"), "PNG", optimize=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        images(sys.argv[1])
    decode_chart(os.path.join(HERE, "decode-speed.png"))
    ttft_chart(os.path.join(HERE, "time-to-first-token.png"))
