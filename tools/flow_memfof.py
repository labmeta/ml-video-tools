"""
MEMFOF optical flow -> motion vector sequence.  (run by process_clip.py)

MEMFOF looks at THREE frames at a time (before, now, after), which gives steadier motion
over time and cleaner edges than two-frame models like WAFT and RAFT. It goes frame by
frame, so memory stays flat however long the clip is.

Output in --out, depending on --format (default both):
    png/box_optical-flow_0001.png ...   png16 or both: 16-bit motion vectors, coded 0-1:
                                        R = x movement, G = y movement, B = 0
                                        0.5 = still; 0 and 1 = the clip's top speed (see _flow_range.txt)
                                        pixels per frame = (value * 2 - 1) * range
    exr/box_optical-flow_0001.exr ...   exr or both: 32-bit float motion vectors in PIXELS (R = x, G = y, B = 0)
    preview/box_optical-flow_preview_0001.png ...  colour pictures, ONE colour scale for the whole clip
Frame N holds the motion from frame N to frame N+1 (or N to N-1 with --backward).
y is positive moving DOWN the picture, in both formats.
The partner-less end frame is written as zero motion so the count matches the clip.

How png16 works: each frame is worked out and saved as an EXR first (so a crash can resume),
then once the whole clip is done, one range is picked for the whole clip and the 16-bit PNGs
are written. With --format png16 the EXRs are then deleted; otherwise they're kept.
"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import worker_start, beat, fatal, list_frames, is_done, progress, out_name

MODELS = {
    "tskh": "MEMFOF-Tartan-T-TSKH",           # the makers' pick for real-world video
    "spring": "MEMFOF-Tartan-T-TSKH-spring",  # tuned on Spring: high-detail, high-res footage
    "sintel": "MEMFOF-Tartan-T-TSKH-sintel",  # tuned on Sintel: film-like CG
    "kitti": "MEMFOF-Tartan-T-TSKH-kitti",    # tuned on driving footage
}

p = worker_start("MEMFOF optical flow")
p.add_argument("--install", help="the MEMFOF folder (not needed - MEMFOF is installed into its environment)")
p.add_argument("--model", default="tskh", choices=list(MODELS),
               help="tskh = general real-world (default), spring = tuned for fine detail")
p.add_argument("--iters", type=int, default=8, help="refinement steps per frame (more = slower, slightly better)")
p.add_argument("--backward", action="store_true", help="motion from frame N to N-1 instead of N to N+1")
p.add_argument("--format", default="both", choices=["png16", "exr", "both"],
               help="both = 16-bit PNG + 32-bit float EXR (default), png16 = PNG only, exr = EXR only")
p.add_argument("--range", type=float,
               help="png16: fixed top speed in pixels per frame (default: worked out from the clip). "
                    "Use the same number on several clips so their PNGs mean the same thing.")
p.add_argument("--no-preview", action="store_true", help="skip the colour preview PNGs")
a = p.parse_args()

try:
    import OpenEXR  # OpenCV's own EXR writer isn't included on Windows
except ImportError:
    fatal("Missing the EXR writer. In the MEMFOF environment run:  pip install OpenEXR")
import numpy as np
import cv2
import torch

want_png, want_exr = a.format in ("png16", "both"), a.format in ("exr", "both")
frames_dir, out_dir = os.path.abspath(a.frames), os.path.abspath(a.out)
exr_dir, png_dir, prev_dir = (os.path.join(out_dir, d) for d in ("exr", "png", "preview"))
for d in [exr_dir] + ([png_dir] if want_png else []) + ([] if a.no_preview else [prev_dir]):
    os.makedirs(d, exist_ok=True)

# Don't mix frames made with different engines or settings
note = os.path.join(out_dir, "_settings.txt")
mine = f"engine=memfof model={MODELS[a.model]} iters={a.iters} backward={a.backward}"
if os.path.isfile(note) and open(note).read().strip() != mine:
    fatal(f"This clip's flow was made with different settings:\n  {open(note).read().strip()}\n"
          f"Add  --redo flow  to throw the old flow away and start again.")
open(note, "w").write(mine)

frames = list_frames(frames_dir)
n = len(frames)
if n < 2:
    fatal(f"Need at least 2 frames in {frames_dir}")
h, w = cv2.imread(frames[0]).shape[:2]
print(f"{n} frames at {w}x{h}, model {MODELS[a.model]}, output {a.format}")

def exr_path(i): return os.path.join(exr_dir, out_name(frames[i], "optical-flow", ".exr"))
def png_path(i): return os.path.join(png_dir, out_name(frames[i], "optical-flow"))
def prev_path(i): return os.path.join(prev_dir, out_name(frames[i], "optical-flow_preview"))

range_file = os.path.join(out_dir, "_flow_range.txt")

def saved_range():
    if not os.path.isfile(range_file):
        return None
    for line in open(range_file):
        if line.startswith("range_pixels_per_frame"):
            return float(line.split("=")[1])
    return None

def save_exr(i, flow):
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    channels = {"R": np.ascontiguousarray(flow[..., 0], np.float32),
                "G": np.ascontiguousarray(flow[..., 1], np.float32),
                "B": np.zeros((h, w), np.float32)}
    tmp = exr_path(i) + ".tmp"
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, channels) as f:
        f.write(tmp)
    os.replace(tmp, exr_path(i))  # appears only once fully written

def save_png(i, flow, rng):
    coded = np.clip(flow / rng * 0.5 + 0.5, 0.0, 1.0)              # 0.5 = still
    px = np.round(coded * 65535).astype(np.uint16)
    bgr = np.dstack([np.zeros((h, w), np.uint16), px[..., 1], px[..., 0]])  # OpenCV order: B, G, R
    tmp = png_path(i) + ".tmp.png"
    cv2.imwrite(tmp, bgr)
    os.replace(tmp, png_path(i))

def read_flow(i):
    """Motion in pixels for frame i - from the EXR if it's there, otherwise decoded from the 16-bit PNG."""
    if is_done(exr_path(i)):
        c = OpenEXR.File(exr_path(i), separate_channels=True).channels()
        return np.stack([c["R"].pixels, c["G"].pixels], axis=-1).astype(np.float32)
    img = cv2.imread(png_path(i), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
    return (np.stack([img[..., 2], img[..., 1]], axis=-1) * 2 - 1) * saved_range()

def done(i):
    return is_done(exr_path(i)) or (want_png and is_done(png_path(i)) and saved_range() is not None)

# A fixed --range has to match what any finished PNGs were made with
if want_png and a.range and saved_range() not in (None, a.range) and any(is_done(png_path(i)) for i in range(n)):
    fatal(f"This clip's PNGs were made with a range of {saved_range():g} pixels.\n"
          f"Add  --redo flow  to make them again with --range {a.range:g}.")

# Forward: frames 0..n-2 have a partner, frame n-1 is zero.  Backward: frames 1..n-1, frame 0 is zero.
needed = range(1, n) if a.backward else range(0, n - 1)
blank = 0 if a.backward else n - 1
todo = [i for i in needed if not done(i)]
if len(todo) < len(needed):
    print(f"Resuming: {len(needed) - len(todo)} frames already done, {len(todo)} to go.")

if todo:
    from memfof import MEMFOF
    from memfof.utils.flow_viz import flow_to_image  # noqa: F401  (check it's there before the long run)
    if not torch.cuda.is_available():
        fatal("No CUDA GPU found. Check this tool's environment has the cu128 PyTorch.")
    print(f"Loading {MODELS[a.model]} (downloads the first time)...")
    try:
        model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{MODELS[a.model]}").eval().to("cuda")
    except Exception as e:
        if any(x in str(e).lower() for x in ("connection", "resolve", "offline", "timed out")):
            fatal(f"Couldn't download the MEMFOF model - check the internet connection.\n({e})")
        raise
    beat()

    def load(i):
        img = cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float()[None]  # 0-255 RGB, as MEMFOF expects

    started, cache, last = time.time(), [None, None, None], None
    with torch.inference_mode():
        for k, i in enumerate(todo, 1):
            if last != i - 1:  # start, or after a gap: nothing to reuse
                cache = [None, None, None]
            trio = torch.stack([load(max(i - 1, 0)), load(i), load(min(i + 1, n - 1))], dim=1).to("cuda")
            output = model(trio, iters=a.iters, fmap_cache=cache)
            flow = output["flow"][-1][:, 0 if a.backward else 1]  # [1, 2, H, W]
            save_exr(i, flow[0].permute(1, 2, 0).float().cpu().numpy())
            cache = output["fmap_cache"]
            cache.pop(0)
            cache.append(None)
            last = i
            beat("frame")
            progress("flow", k, len(todo), started)
    print()

if not done(blank):
    save_exr(blank, np.zeros((h, w, 2), np.float32))

# 16-bit PNGs, with ONE range for the whole clip so a value means the same speed on every frame
if want_png and not all(is_done(png_path(i)) for i in range(n)):
    rng = saved_range()
    if rng is None:
        if a.range:
            rng, how = a.range, "set with --range"
        else:
            print("Finding the top speed across the clip...")
            peaks = []
            for i in range(n):  # every frame, so a fast moment can't be missed
                peaks.append(float(np.percentile(np.abs(read_flow(i)), 99.99)))
                if i % 20 == 0:
                    beat("frame")
            rng, how = max(max(peaks) * 1.05, 0.5), "worked out from the clip (fastest 0.01% may clip)"
        rng = round(rng, 4)  # exactly what's saved below, so frames written after a crash match
        with open(range_file, "w") as f:
            f.write(f"range_pixels_per_frame = {rng:.4f}\n"
                    f"({how})\n\n"
                    f"16-bit PNGs: R = x movement, G = y movement (positive = DOWN), B = 0\n"
                    f"0.5 = still. To get pixels per frame back:  (value * 2 - 1) * {rng:.4f}\n"
                    f"Frame N holds the motion from frame N to N{'-' if a.backward else '+'}1.\n")
    print(f"Writing 16-bit PNGs (range +/- {rng:.2f} pixels per frame)...")
    for i in range(n):
        if not is_done(png_path(i)):
            save_png(i, read_flow(i), rng)
        if i % 10 == 0:
            beat("frame")

# Previews with ONE colour scale for the whole clip, so they don't flicker or hide detail
if not a.no_preview and not all(os.path.isfile(prev_path(i)) for i in range(n)):
    from memfof.utils.flow_viz import flow_to_image
    print("Writing previews...")
    sample = sorted(set(np.linspace(0, n - 1, min(n, 60)).astype(int)))
    mags = np.concatenate([np.linalg.norm(read_flow(i), axis=-1).ravel()[::9] for i in sample])
    scale = max(float(np.percentile(mags, 99.5)), 1e-3)
    with open(os.path.join(out_dir, "_preview_scale.txt"), "w") as f:
        f.write(f"previews: full colour = {scale:.2f} pixels per frame or faster\n")
    for i in range(n):
        if os.path.isfile(prev_path(i)):
            continue
        flow = read_flow(i)
        mag = np.linalg.norm(flow, axis=-1, keepdims=True)
        flow = flow * np.minimum(1.0, scale / np.maximum(mag, 1e-6))  # cap speed at the scale, keep direction
        cv2.imwrite(prev_path(i), flow_to_image(flow, convert_to_bgr=True, rad_min=scale))
        if i % 10 == 0:
            beat("frame")
    print(f"  colour scale: full colour = {scale:.2f} pixels per frame")

# png16 only: the EXRs were just working files
if not want_exr and all(is_done(png_path(i)) for i in range(n)):
    shutil.rmtree(exr_dir, ignore_errors=True)

print(f"Done -> {out_dir}")
