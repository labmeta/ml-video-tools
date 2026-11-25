"""
SAM 3.1 -> black-and-white matte sequences from a text prompt.  (run by process_clip.py)

Output in --out, one folder per prompt:
    person/box_segmentation_person_0001.png ...   white = the object, black = everything else
Every match for that word goes into the same matte (all the people in one matte).

Long clips are worked through in chunks so memory use stays flat.

Includes three workarounds for SAM 3.1 on Windows:
  - Linux-only speed-ups (FlashAttention 3, torch.compile) are switched off
  - SAM forces PyTorch's "flash" attention, which Windows PyTorch doesn't have,
    so we let it fall back to the other methods ("No available kernel" error)
  - SAM 3.1's own start_session passes a setting its video model rejects,
    so we start sessions ourselves
"""
import os, sys, time, re, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common import worker_start, beat, fatal, list_frames, is_done, progress, out_name

p = worker_start("SAM 3.1 mattes from a word")
p.add_argument("--prompt", action="append", required=True,
               help='what to find, e.g. --prompt person. Repeat for more: --prompt person --prompt dog')
p.add_argument("--chunk", type=int, default=200, help="frames per chunk. Lower = less memory")
p.add_argument("--max-objects", type=int, default=16, help="most things it will track at once per prompt")
p.add_argument("--threshold", type=float, default=0.5, help="how sure it must be (0-1). Lower finds more")
a = p.parse_args()

import numpy as np
import cv2
import torch

frames_dir, out_dir = os.path.abspath(a.frames), os.path.abspath(a.out)
chunks_dir = os.path.join(out_dir, "_chunks")

def folder_name(prompt):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", prompt.strip()).strip("_") or "prompt"

prompts = [(pr, os.path.join(out_dir, folder_name(pr))) for pr in a.prompt]
mine = f"threshold={a.threshold} max_objects={a.max_objects}"
for pr, folder in prompts:
    os.makedirs(folder, exist_ok=True)
    note = os.path.join(folder, "_settings.txt")
    if os.path.isfile(note) and open(note).read().strip() != mine:
        fatal(f"The '{pr}' mattes were started with different settings ({open(note).read().strip()}).\n"
              f"Add  --redo mattes  to throw the old mattes away and start again.")
    open(note, "w").write(mine)

frames = list_frames(frames_dir)
n = len(frames)
if n == 0:
    fatal(f"No frames found in {frames_dir}")
h, w = cv2.imread(frames[0]).shape[:2]
print(f"{n} frames at {w}x{h}")

def matte_path(folder, i):  # box_0001.png -> person/box_segmentation_person_0001.png
    return os.path.join(folder, out_name(frames[i], f"segmentation_{os.path.basename(folder)}"))

def save(folder, i, mask):
    tmp = matte_path(folder, i) + ".tmp.png"
    cv2.imwrite(tmp, mask)
    os.replace(tmp, matte_path(folder, i))

chunks = [(s, min(s + a.chunk, n)) for s in range(0, n, a.chunk)]
jobs = [(pr, folder, s, e) for pr, folder in prompts for s, e in chunks
        if not all(is_done(matte_path(folder, i)) for i in range(s, e))]
total = sum(e - s for _, _, s, e in jobs)
if total < len(prompts) * n:
    print(f"Resuming: {len(prompts) * n - total} mattes already done, {total} to go.")

def make_chunk_dir(s, e):
    """Give SAM just this chunk's frames, named 0.png, 1.png ... without copying (hard links)."""
    d = os.path.join(chunks_dir, f"{s:06d}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for k, i in enumerate(range(s, e)):
        dst = os.path.join(d, f"{k}{os.path.splitext(frames[i])[1].lower()}")
        try:
            os.link(frames[i], dst)
        except OSError:
            shutil.copy2(frames[i], dst)
    return d

def start_session(pred, folder):
    """SAM 3.1's own start_session crashes (it passes offload_state_to_cpu), so do it ourselves."""
    import uuid, inspect
    kwargs = dict(resource_path=folder, offload_video_to_cpu=True)
    accepted = inspect.signature(pred.model.init_state).parameters
    if "async_loading_frames" in accepted and hasattr(pred, "async_loading_frames"):
        kwargs["async_loading_frames"] = pred.async_loading_frames
    if "offload_state_to_cpu" in accepted:
        kwargs["offload_state_to_cpu"] = False
    with torch.inference_mode():
        state = pred.model.init_state(**kwargs)
    sid = str(uuid.uuid4())
    pred._all_inference_states[sid] = {"state": state, "session_id": sid,
                                       "start_time": time.time(), "last_use_time": time.time()}
    return sid

if jobs:
    if not torch.cuda.is_available():
        fatal("No CUDA GPU found. Check this tool's environment has the cu128 PyTorch.")
    print("Loading SAM 3.1 (downloads about 3GB the first time)...")
    from sam3.model_builder import build_sam3_predictor

    # Windows PyTorch has no "flash" attention - let SAM fall back to the other methods
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import sam3.model.decoder as _sam_decoder
    _backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    _sam_decoder.sdpa_kernel = lambda *args, **kwargs: sdpa_kernel(_backends)

    try:
        predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, compile=False,
                                         max_num_objects=a.max_objects)
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("401", "403", "gated", "access")):
            fatal("Hugging Face won't let me download SAM 3.1.\n"
                  "Check your access was approved at huggingface.co/facebook/sam3.1,\n"
                  "then in the SAM environment run:  hf auth login")
        raise
    beat()

    started, done = time.time(), 0
    for pr, folder, s, e in jobs:
        print(f"\n'{pr}': frames {s + 1}-{e}")
        chunk_dir = make_chunk_dir(s, e)
        sid = start_session(predictor, chunk_dir)
        try:
            predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                      "text": pr, "output_prob_thresh": a.threshold})
            written = set()
            for out in predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": sid, "propagation_direction": "forward",
                    "output_prob_thresh": a.threshold, "evict_cached_frame_outputs": True}):
                k = out["frame_index"]
                masks = out["outputs"]["out_binary_masks"]  # (objects, H, W) true/false
                matte = np.zeros((h, w), np.uint8)
                if len(masks):
                    matte = np.any(np.asarray(masks), axis=0).astype(np.uint8) * 255
                    if matte.shape != (h, w):
                        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_NEAREST)
                save(folder, s + k, matte)
                written.add(k)
                done += 1
                beat("frame")
                progress(f"'{pr}'", done, total, started)
            # Frames SAM didn't report on get an empty matte, so the sequence has no gaps
            for k in range(e - s):
                if k not in written and not is_done(matte_path(folder, s + k)):
                    save(folder, s + k, np.zeros((h, w), np.uint8))
        finally:
            try:
                predictor.handle_request({"type": "close_session", "session_id": sid})
            except Exception:
                pass
            shutil.rmtree(chunk_dir, ignore_errors=True)
            torch.cuda.empty_cache()

shutil.rmtree(chunks_dir, ignore_errors=True)
print(f"\nDone -> {out_dir}")
