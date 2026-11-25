"""
WAFT optical flow -> motion vector sequence. (MEMFOF is the default flow engine - this is --flow-engine waft)  (run by process_clip.py)

Output in --out:
    exr/box_optical-flow_0001.exr ...              32-bit float motion vectors in PIXELS
                                                   R = x movement, G = y movement, B = 0
    preview/box_optical-flow_preview_0001.png ...  colour-wheel picture to eyeball it
Frame N holds the motion from frame N to frame N+1 (or N to N-1 with --backward).
The partner-less end frame is written as zero motion so the count matches the clip.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import worker_start, beat, fatal, list_frames, is_done, progress, load_settings, tool_install, out_name

p = worker_start("WAFT optical flow")
p.add_argument("--install", help="the WAFT folder (default: from settings.json)")
p.add_argument("--ckpt", default="ckpts/waft_a1.pth", help="the WAFT checkpoint, inside the WAFT folder")
p.add_argument("--cfg", default="config/a1/tar-c-t.json")
p.add_argument("--half-res", action="store_true", help="work at half size: faster, less GPU memory, softer edges")
p.add_argument("--backward", action="store_true", help="motion from frame N to N-1 instead of N to N+1")
p.add_argument("--no-preview", action="store_true", help="skip the colour preview PNGs")
p.add_argument("--iters", type=int, help="refinement steps per frame (WAFT's own default is 5)")
a = p.parse_args()

try:
    import OpenEXR  # OpenCV's own EXR writer isn't included on Windows
except ImportError:
    fatal("Missing the EXR writer. In the WAFT environment run:  pip install OpenEXR")
import numpy as np
import cv2
import torch

frames_dir, out_dir = os.path.abspath(a.frames), os.path.abspath(a.out)
waft_dir = os.path.abspath(a.install or tool_install(load_settings(), "flow"))

# WAFT's own code expects to be run from its folder
os.chdir(waft_dir)
sys.path.insert(0, waft_dir)
for need in [a.ckpt, "depth-anything-ckpts/depth_anything_v2_vits.pth"]:
    if not os.path.isfile(need):
        fatal(f"Missing file: {os.path.join(waft_dir, need)}\nSee INSTALL.md for where to download it.")
from config.parser import json_to_args
from model import fetch_model
from utils.utils import load_ckpt
from utils.flow_viz import flow_to_image
from inference_tools import InferenceWrapper

exr_dir = os.path.join(out_dir, "exr")
prev_dir = os.path.join(out_dir, "preview")
for d in [exr_dir] + ([] if a.no_preview else [prev_dir]):
    os.makedirs(d, exist_ok=True)

# Don't mix frames made with different settings
note = os.path.join(out_dir, "_settings.txt")
mine = f"half_res={a.half_res} backward={a.backward}" + (f" iters={a.iters}" if a.iters else "")
if os.path.isfile(note) and open(note).read().strip() != mine:
    fatal(f"This clip's flow was started with different settings ({open(note).read().strip()}).\n"
          f"Add  --redo flow  to throw the old flow away and start again.")
open(note, "w").write(mine)

frames = list_frames(frames_dir)
n = len(frames)
if n < 2:
    fatal(f"Need at least 2 frames in {frames_dir}")
h, w = cv2.imread(frames[0]).shape[:2]
print(f"{n} frames at {w}x{h}")

def exr_path(i): return os.path.join(exr_dir, out_name(frames[i], "optical-flow", ".exr"))
def preview_path(i): return os.path.join(prev_dir, out_name(frames[i], "optical-flow_preview"))

step = -1 if a.backward else 1
order = range(n - 1, 0, -1) if a.backward else range(0, n - 1)
todo = [i for i in order if not is_done(exr_path(i))]
if len(todo) < len(order):
    print(f"Resuming: {len(order) - len(todo)} frames already done, {len(todo)} to go.")

def save(i, flow):
    if not a.no_preview:
        cv2.imwrite(preview_path(i), flow_to_image(flow, convert_to_bgr=True))
    channels = {"R": np.ascontiguousarray(flow[..., 0], np.float32),
                "G": np.ascontiguousarray(flow[..., 1], np.float32),
                "B": np.zeros((h, w), np.float32)}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    tmp = exr_path(i) + ".tmp"
    with OpenEXR.File(header, channels) as f:
        f.write(tmp)
    os.replace(tmp, exr_path(i))  # appears only once fully written

if todo:
    if not torch.cuda.is_available():
        fatal("No CUDA GPU found. Check this tool's environment has the cu128 PyTorch.")
    device = torch.device("cuda")
    cfg = json_to_args(a.cfg)
    if a.iters:
        cfg.iters = a.iters
    model = fetch_model(cfg)
    load_ckpt(model, a.ckpt)
    model = model.to(device).eval()
    beat()
    runner = InferenceWrapper(model, scale=(-1 if a.half_res else 0), train_size=cfg.image_size,
                              pad_to_train_size=False, tiling=False)

    def load(path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).float().permute(2, 0, 1)[None].to(device)  # 0-255 RGB

    started = time.time()
    with torch.no_grad():
        for k, i in enumerate(todo, start=1):
            out = runner.calc_flow(load(frames[i]), load(frames[i + step]))
            save(i, out["flow"][-1][0].permute(1, 2, 0).cpu().numpy())
            beat("frame")
            progress("flow", k, len(todo), started)
    print()

save(0 if a.backward else n - 1, np.zeros((h, w, 2), np.float32))
print(f"Done -> {out_dir}")
