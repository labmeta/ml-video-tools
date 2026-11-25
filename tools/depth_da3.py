"""
Depth Anything 3 -> 16-bit PNG depth sequence.  (run by process_clip.py)

Files are named after their source frame: box_0001.png -> box_depth_0001.png
Far = white, near = black by default (process_clip.py; flip with --near-white). Normalised across the WHOLE clip,
so values don't jump from frame to frame.

Two passes so long clips don't fill your memory:
  1. depth chunk by chunk, raw numbers saved to <out>/_raw  (resumable)
  2. one near/far range for the whole clip, then the PNGs   (resumable)
"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import worker_start, beat, fatal, list_frames, is_done, progress, out_name, match_depth_white

p = worker_start("Depth Anything 3 depth maps")
p.add_argument("--model", default="depth-anything/DA3-BASE",
               help="DA3-SMALL / DA3-BASE / DA3MONO-LARGE / DA3METRIC-LARGE (commercial OK), "
                    "DA3-LARGE-1.1 / DA3-GIANT-1.1 (non-commercial)")
p.add_argument("--chunk", type=int, default=16, help="frames processed together. Lower if you run out of GPU memory")
p.add_argument("--far-white", action="store_true", help="far = white instead of near = white")
p.add_argument("--keep-raw", action="store_true", help="keep the raw depth numbers (.npy) afterwards")
a = p.parse_args()

import numpy as np
import cv2

frames_dir, out_dir = os.path.abspath(a.frames), os.path.abspath(a.out)
raw_dir = os.path.join(out_dir, "_raw")
os.makedirs(raw_dir, exist_ok=True)

# Depth from one model must not be mixed with another's
model_note = os.path.join(out_dir, "_model.txt")
if os.path.isfile(model_note) and open(model_note).read().strip() != a.model:
    fatal(f"This clip's depth was started with {open(model_note).read().strip()}, but you asked for {a.model}.\n"
          f"Add  --redo depth  to throw the old depth away and start again with the new model.")
open(model_note, "w").write(a.model)

frames = list_frames(frames_dir)
n = len(frames)
if n == 0:
    fatal(f"No frames found in {frames_dir}")
h, w = cv2.imread(frames[0]).shape[:2]
print(f"{n} frames at {w}x{h}")

def png_path(i): return os.path.join(out_dir, out_name(frames[i], "depth"))  # box_depth_0001.png
match_depth_white(out_dir, [png_path(i) for i in range(n)], a.far_white)
def raw_path(i): return os.path.join(raw_dir, f"raw_{i + 1:04d}.npy")

if not all(is_done(png_path(i)) for i in range(n)):
    # Pass 1: raw depth, chunk by chunk
    chunks = [(s, min(s + a.chunk, n)) for s in range(0, n, a.chunk)]
    todo = [(s, e) for s, e in chunks if not all(is_done(raw_path(i)) for i in range(s, e))]
    left = sum(e - s for s, e in todo)
    if left < n:
        print(f"Resuming: {n - left} frames of depth already done, {left} to go.")
    if todo:
        import torch
        from depth_anything_3.api import DepthAnything3
        if not torch.cuda.is_available():
            fatal("No CUDA GPU found. Check this tool's environment has the cu128 PyTorch.")
        print(f"Loading {a.model}...")
        model = DepthAnything3.from_pretrained(a.model).to(device=torch.device("cuda"))
        beat()
        started, done = time.time(), 0
        for s, e in todo:
            pred = model.inference(frames[s:e])
            for k, d in enumerate(pred.depth):
                tmp = raw_path(s + k) + ".tmp.npy"
                np.save(tmp, d.astype(np.float16))
                os.replace(tmp, raw_path(s + k))
            done += e - s
            beat("frame")
            progress("depth", done, left, started)
            torch.cuda.empty_cache()
        print()

    # Pass 2: one near/far range for the whole clip, then the PNGs
    print("Working out the depth range across the whole clip...")
    sample = sorted(set(np.linspace(0, n - 1, min(n, 300)).astype(int)))
    vals = np.concatenate([np.load(raw_path(i)).astype(np.float32).ravel()[::7] for i in sample])
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    del vals
    with open(os.path.join(out_dir, "_depth_range.txt"), "w") as f:
        f.write(f"model {a.model}\nraw values used for black/white: {lo} {hi}\n")
    beat("frame")
    print("Writing depth PNGs...")
    for i in range(n):
        if is_done(png_path(i)):
            continue
        d = cv2.resize(np.load(raw_path(i)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        norm = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
        if not a.far_white:
            norm = 1.0 - norm  # near = white
        tmp = png_path(i) + ".tmp.png"
        cv2.imwrite(tmp, (norm * 65535).astype(np.uint16))
        os.replace(tmp, png_path(i))
        if i % 10 == 0:
            beat("frame")
            print(f"  png {i + 1}/{n}   ", end="\r", flush=True)
    print()

if not a.keep_raw:
    shutil.rmtree(raw_dir, ignore_errors=True)
print(f"Done -> {out_dir}")
