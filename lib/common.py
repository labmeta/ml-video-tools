"""
Shared bits used by process_clip.py and all the tool scripts.

Only uses Python's standard library, so it works from any of the tool environments.
"""
import os
import re
import sys
import json
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(REPO_ROOT, "settings.json")
SETTINGS_EXAMPLE = os.path.join(REPO_ROOT, "settings.example.json")


def version():
    try:
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"

# Set by the babysitter so a tool can say "I'm still alive"
HEARTBEAT_ENV = "CLIP_TOOLS_HEARTBEAT"

# Exit code meaning "stopped on purpose, restarting won't help"
EXIT_NO_RETRY = 3

DEFAULT_SETTINGS = {
    "tools": {
        "depth_vda": {"install": r"C:\ML\vda"},
        "depth_da3": {"install": r"C:\ML\da3\Depth-Anything-3"},
        "flow_memfof": {"install": r"C:\ML\memfof"},
        "flow_waft": {"install": r"C:\ML\waft"},
        "mattes": {"install": r"C:\ML\sam"},
    },
    "babysitter": {
        "frame_timeout_sec": 300,
        "setup_timeout_sec": 3600,
        "max_fails_in_a_row": 10,
        "wait_before_retry_sec": 10,
    },
    "defaults": {
        "depth_engine": "vda",                     # vda = Video Depth Anything, da3 = Depth Anything 3
        "vda_encoder": "vits",                     # vits (default, commercial OK), vitb / vitl (non-commercial)
        "depth_model": "depth-anything/DA3-BASE",  # only used when depth_engine is da3
        "depth_white": "far",                      # far = white (default) or near = white
        "flow_engine": "memfof",                   # memfof = MEMFOF, waft = WAFT
        "memfof_model": "tskh",                    # tskh = real-world (default), spring = fine detail
        "flow_format": "both",                     # MEMFOF output: both (PNG + EXR), png16 or exr
    },
}


def fatal(msg):
    """Stop for a reason that retrying won't fix (missing file, wrong setting)."""
    print("\n" + msg, file=sys.stderr, flush=True)
    sys.exit(EXIT_NO_RETRY)


def _merge(base, extra):
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_settings():
    """Read settings.json (falls back to the defaults for anything it doesn't mention)."""
    if not os.path.isfile(SETTINGS_PATH):
        # First run: make settings.json from the example. Updates ship only the example,
        # so unzipping a new version never overwrites your own settings.
        if os.path.isfile(SETTINGS_EXAMPLE):
            import shutil
            shutil.copyfile(SETTINGS_EXAMPLE, SETTINGS_PATH)
            print(f"Made settings.json from settings.example.json - check the folders in it are right.")
        else:
            return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            user = json.load(f)
    except json.JSONDecodeError as e:
        fatal(f"settings.json has a typo on line {e.lineno}: {e.msg}\n"
              f"Tip: in folder paths use forward slashes (C:/ML/waft) or double backslashes (C:\\\\ML\\\\waft).")
    # Settings from before 0.3.0 called the Depth Anything 3 install just "depth"
    tools = user.get("tools", {})
    if "depth" in tools and "depth_da3" not in tools:
        tools["depth_da3"] = tools.pop("depth")
    # ...and before 0.4.0 the WAFT install was just "flow"
    if "flow" in tools and "flow_waft" not in tools:
        tools["flow_waft"] = tools.pop("flow")
    return _merge(DEFAULT_SETTINGS, user)


def engine_key(tool, engine):
    """('depth', 'vda') -> 'depth_vda': the settings / tools key for an engine."""
    return f"{tool}_{engine}"


def depth_engine_key(engine):
    return engine_key("depth", engine)


def tool_install(settings, tool):
    return os.path.normpath(settings["tools"][tool]["install"])


def tool_python(settings, tool):
    """The Python inside that tool's own environment (or an explicit "python" in settings.json)."""
    t = settings["tools"][tool]
    if t.get("python"):
        return os.path.normpath(t["python"])
    install = tool_install(settings, tool)
    if os.name == "nt":
        return os.path.join(install, ".venv", "Scripts", "python.exe")
    return os.path.join(install, ".venv", "bin", "python")


def tool_script(tool):
    return os.path.join(REPO_ROOT, "tools", f"{tool}.py")


def windows_tweaks():
    """No click-to-pause in the Command Prompt window, and no sleeping mid-run."""
    if os.name != "nt":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    try:
        h = k32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)  # QuickEdit off
    except Exception:
        pass
    try:
        k32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keep the PC awake
    except Exception:
        pass


def beat(stage="setup"):
    """Tell the babysitter we're alive. stage is "setup" (slow bits) or "frame" (steady work)."""
    path = os.environ.get(HEARTBEAT_ENV)
    if path:
        try:
            with open(path, "w") as f:
                f.write(stage)
        except OSError:
            pass


def list_frames(folder):
    """Image files in a folder, in frame order (by the last number in each name)."""
    files = [f for f in glob.glob(os.path.join(folder, "*"))
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]

    def key(f):
        nums = re.findall(r"\d+", os.path.basename(f))
        return (int(nums[-1]) if nums else 0, os.path.basename(f))
    return sorted(files, key=key)


def out_name(frame_path, suffix, ext=".png"):
    """Name an output after its source frame, frame number last:
    box_0001.png -> box_depth_0001.png"""
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    m = re.match(r"^(.*?)[_.-]?(\d+)$", stem)
    if m and m.group(1):
        return f"{m.group(1)}_{suffix}_{m.group(2)}{ext}"
    if m:  # name is only a number, e.g. 0001.png
        return f"{suffix}_{m.group(2)}{ext}"
    return f"{stem}_{suffix}{ext}"


def is_done(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def worker_start(description):
    """Common start for every tool script: Windows fixes, heartbeat, and the shared arguments."""
    import argparse
    windows_tweaks()
    beat()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--frames", required=True, help="folder of extracted frames")
    p.add_argument("--out", required=True, help="folder to write the results into")
    return p


def progress(label, done, total, started, extra=""):
    import time
    per = (time.time() - started) / max(done, 1)
    left = (total - done) * per
    print(f"  {label} {done}/{total}  |  {per:.2f}s per frame  |  about {left / 60:.0f} min left {extra}  ",
          end="\r", flush=True)


def match_depth_white(out_dir, png_paths, far_white):
    """Make a shot's finished depth PNGs match the near/far = white setting asked for.

    The choice is recorded in _white.txt. If finished PNGs were made the other way round they're
    flipped in place (65535 - value), so there's no need to work the depth out again. Shots from
    before this note existed were always near = white. Safe to stop part-way: flipped copies are
    written first (.flip.png) and only swapped in once they're all there.
    """
    import cv2
    note = os.path.join(out_dir, "_white.txt")
    wanted = "far" if far_white else "near"
    done = [p for p in png_paths if is_done(p)]
    state = open(note).read().strip() if os.path.isfile(note) else ("near" if done else wanted)
    if state == wanted:
        open(note, "w").write(wanted)
        return
    flip = lambda p: p[:-4] + ".flip.png"
    if state != f"{wanted} (swapping in)":
        print(f"Depth PNGs were made with {state.split()[0]} = white - flipping {len(done)} of them "
              f"to {wanted} = white (no need to work the depth out again)...")
        for k, p in enumerate(done):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(flip(p), (65535 - img.astype("int32")).astype(img.dtype))
            if k % 20 == 0:
                beat("frame")
        open(note, "w").write(f"{wanted} (swapping in)")
    for p in done:
        if os.path.isfile(flip(p)):
            os.replace(flip(p), p)
    open(note, "w").write(wanted)
