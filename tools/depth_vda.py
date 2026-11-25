"""
Video Depth Anything -> 16-bit PNG depth sequence.  (run by process_clip.py)

Video Depth Anything looks at runs of frames together, so depth stays steady over time
(no flicker). On its own it loads the whole clip into memory, which long clips can't
afford, so here the clip goes through in overlapping chunks:
  - each chunk starts with the last frames of the one before
  - the new chunk is scaled to match the old one on those shared frames
    (the same trick Video Depth Anything uses between its own 32-frame windows)
  - the shared frames are cross-faded, so there's no jump where chunks meet

Then, like the DA3 tool, one near/far range is worked out for the whole clip and the
PNGs are written. Far = white, near = black by default (process_clip.py; flip with --near-white).
Files are named after their source frame: box_0001.png -> box_depth_0001.png
"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import (worker_start, beat, fatal, list_frames, is_done, progress, out_name,
                        load_settings, tool_install, match_depth_white)

p = worker_start("Video Depth Anything depth maps")
p.add_argument("--install", help="the Video Depth Anything folder (default: from settings.json)")
p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"],
               help="vits = Small (commercial OK), vitb = Base, vitl = Large (both non-commercial)")
p.add_argument("--chunk", type=int, default=320, help="frames per chunk. Lower = less memory")
p.add_argument("--overlap", type=int, default=32, help="frames shared between chunks for matching")
p.add_argument("--max-res", type=int, default=1024,
               help="longest side the depth is worked out at (the PNGs are scaled back to full size)")
p.add_argument("--far-white", action="store_true", help="far = white instead of near = white")
p.add_argument("--keep-raw", action="store_true", help="keep the raw depth numbers (.npy) afterwards")
a = p.parse_args()
if a.overlap >= a.chunk:
    fatal("--overlap has to be smaller than --chunk")

import numpy as np
import cv2

frames_dir, out_dir = os.path.abspath(a.frames), os.path.abspath(a.out)
vda_dir = os.path.abspath(a.install or tool_install(load_settings(), "depth_vda"))
raw_dir = os.path.join(out_dir, "_raw")
os.makedirs(raw_dir, exist_ok=True)

ckpt = os.path.join(vda_dir, "checkpoints", f"video_depth_anything_{a.encoder}.pth")
if not os.path.isfile(ckpt):
    fatal(f"Missing model file: {ckpt}\nSee INSTALL.md for the download command.")

# Depth from one model must not be mixed with another's
model_id = f"video-depth-anything/{a.encoder}"
model_note = os.path.join(out_dir, "_model.txt")
if os.path.isfile(model_note) and open(model_note).read().strip() != model_id:
    fatal(f"This clip's depth was made with {open(model_note).read().strip()}, but you asked for {model_id}.\n"
          f"Add  --redo depth  to throw the old depth away and start again.")
open(model_note, "w").write(model_id)
if a.encoder != "vits":
    print(f"Note: Video Depth Anything {'Large' if a.encoder == 'vitl' else 'Base'} is non-commercial only. "
          f"For client work use the Small model (vits).")

frames = list_frames(frames_dir)
n = len(frames)
if n == 0:
    fatal(f"No frames found in {frames_dir}")
h, w = cv2.imread(frames[0]).shape[:2]
scale = min(1.0, a.max_res / max(h, w))
wh = (max(2, round(w * scale)), max(2, round(h * scale)))  # working size
print(f"{n} frames at {w}x{h} (depth worked out at {wh[0]}x{wh[1]})")

def png_path(i): return os.path.join(out_dir, out_name(frames[i], "depth"))
match_depth_white(out_dir, [png_path(i) for i in range(n)], a.far_white)
def raw_path(i): return os.path.join(raw_dir, f"raw_{i + 1:06d}.npy")

def save_raw(i, d):
    tmp = raw_path(i) + ".tmp.npy"
    np.save(tmp, d.astype(np.float16))
    os.replace(tmp, raw_path(i))

# Chunk k "owns" frames [k*chunk, (k+1)*chunk) and also reads the `overlap` frames before them
chunks = [(s, min(s + a.chunk, n)) for s in range(0, n, a.chunk)]
def chunk_done(k): return os.path.isfile(os.path.join(raw_dir, f"_chunk_{k:05d}_done"))

if not all(is_done(png_path(i)) for i in range(n)):
    todo = [k for k in range(len(chunks)) if not chunk_done(k)]
    left = sum(chunks[k][1] - chunks[k][0] for k in todo)
    if left < n:
        print(f"Resuming: {n - left} frames of depth already done, {left} to go.")

    if todo:
        import torch
        if not torch.cuda.is_available():
            fatal("No CUDA GPU found. Check this tool's environment has the cu128 PyTorch.")
        os.chdir(vda_dir)
        sys.path.insert(0, vda_dir)
        from video_depth_anything.video_depth import VideoDepthAnything
        from utils.util import compute_scale_and_shift

        configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        print(f"Loading Video Depth Anything ({a.encoder})...")
        model = VideoDepthAnything(**configs[a.encoder])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        model = model.to("cuda").eval()
        beat()

        def read(i):
            img = cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB)
            return cv2.resize(img, wh, interpolation=cv2.INTER_AREA) if scale < 1 else img

        started, done = time.time(), 0
        for k in todo:
            s, e = chunks[k]
            lead = 0 if k == 0 else min(a.overlap, s)       # shared frames from the chunk before
            idx = list(range(s - lead, e))
            batch = np.stack([read(i) for i in idx])
            beat("frame")
            depths, _ = model.infer_video_depth(batch, 24, input_size=518, device="cuda", fp32=False)
            depths = np.asarray(depths, np.float32)
            del batch

            if lead:
                # Match this chunk to the depth already saved for the shared frames
                old = np.stack([np.load(raw_path(i)).astype(np.float32) for i in idx[:lead]])
                new = depths[:lead]
                step = 4  # every 4th pixel is plenty for working out scale and shift
                sc, sh = compute_scale_and_shift(new[:, ::step, ::step].ravel(), old[:, ::step, ::step].ravel(),
                                                 np.ones(new[:, ::step, ::step].size, bool))
                depths = np.maximum(depths * sc + sh, 0)
                # Cross-fade the shared frames from old to new so there's no jump
                for j in range(lead):
                    t = (j + 1) / (lead + 1)
                    save_raw(idx[j], old[j] * (1 - t) + depths[j] * t)

            for j in range(lead, len(idx)):
                save_raw(idx[j], depths[j])
            open(os.path.join(raw_dir, f"_chunk_{k:05d}_done"), "w").write("ok")
            done += e - s
            beat("frame")
            progress("depth", done, left, started)
            del depths
            torch.cuda.empty_cache()
        print()

    # One near/far range for the whole clip, then the PNGs
    print("Working out the depth range across the whole clip...")
    sample = sorted(set(np.linspace(0, n - 1, min(n, 300)).astype(int)))
    vals = np.concatenate([np.load(raw_path(i)).astype(np.float32).ravel()[::7] for i in sample])
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    del vals
    with open(os.path.join(out_dir, "_depth_range.txt"), "w") as f:
        f.write(f"model {model_id}\nraw values used for black/white: {lo} {hi}\n"
                f"(Video Depth Anything gives inverse depth: bigger = nearer)\n")
    beat("frame")
    print("Writing depth PNGs...")
    for i in range(n):
        if is_done(png_path(i)):
            continue
        d = np.load(raw_path(i)).astype(np.float32)
        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
        norm = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)  # VDA: bigger = nearer, so near = white already
        if a.far_white:
            norm = 1.0 - norm
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
