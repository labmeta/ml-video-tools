r"""
Process a clip through all three tools in one go:
  1. extracts the frames ONCE
  2. depth   (Video Depth Anything by default, or Depth Anything 3 with --depth-engine da3)
  3. flow    (MEMFOF by default, or WAFT with --flow-engine waft)
  4. mattes  (SAM 3.1 - only if you give it --prompt)

Each tool runs from its own install (set in settings.json), looked after by the
babysitter: if one crashes or freezes it's restarted and carries on where it left off.

Run it with ANY Python - no need to switch on an environment first:
    python process_clip.py "D:\clips\test.mp4" --prompt person
    python process_clip.py --check          (checks your settings and installs)

Everything ends up in one folder next to the clip:
    test_shot\
        frames\     test_0001.png ...  (kept, so you can re-run any tool)
        depth\      test_depth_0001.png ...
        flow\       exr\test_optical-flow_0001.exr ...  and  preview\test_optical-flow_preview_0001.png ...
        mattes\     person\test_segmentation_person_0001.png ...
        log.txt     what happened
"""
import os, sys, time, glob, shutil, argparse, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.common import (load_settings, tool_install, tool_python, tool_script,
                        windows_tweaks, SETTINGS_PATH, version, engine_key)
from lib.babysitter import babysit

TOOL_ORDER = ["depth", "flow", "mattes"]
ENGINES = {
    "depth": {"vda": "Video Depth Anything", "da3": "Depth Anything 3"},
    "flow": {"memfof": "MEMFOF", "waft": "WAFT"},
}
DEFAULT_ENGINE = {"depth": "vda", "flow": "memfof"}

# Folder names inside <clip>_shot. Shots made before 0.7.0 used the lower-case names on the left.
FOLDERS = {"frames": "Color", "depth": "Depth", "flow": "Optical", "mattes": "Segm"}


def rename_old_folders(shot, log):
    """Rename a shot's pre-0.7.0 folders (frames, depth, flow, mattes) to Color, Depth, Optical, Segm.

    Goes via a temporary name, because Windows sees "depth" and "Depth" as the same folder.
    """
    for old, new in FOLDERS.items():
        there = os.listdir(shot)
        if old not in there:
            continue
        if new in there:
            log(f"Note: this shot has both '{old}' and '{new}' folders - leaving '{old}' alone. "
                f"Move anything you need out of it, then delete it.")
            continue
        temp = os.path.join(shot, f"_{new}_renaming")
        os.rename(os.path.join(shot, old), temp)
        os.rename(temp, os.path.join(shot, new))
        log(f"Renamed the '{old}' folder to '{new}'.")
    for new in FOLDERS.values():  # finish a rename that was interrupted half way
        temp = os.path.join(shot, f"_{new}_renaming")
        if os.path.isdir(temp) and new not in os.listdir(shot):
            os.rename(temp, os.path.join(shot, new))


def default_engine(settings, tool):
    return settings["defaults"].get(f"{tool}_engine", DEFAULT_ENGINE[tool])


def tool_key(settings, tool, engine=None):
    """The settings / script name for a tool: depth -> depth_vda, flow -> flow_memfof, mattes -> mattes."""
    if tool in ENGINES:
        return engine_key(tool, engine or default_engine(settings, tool))
    return tool


def engine_label(key):
    """depth_vda -> 'Video Depth Anything'"""
    tool, _, eng = key.partition("_")
    return ENGINES.get(tool, {}).get(eng, "")
START_DIR = os.getcwd()


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(time.strftime("%H:%M:%S  ") + msg + "\n")
        self.f.flush()


def check_tool(settings, tool, encoder=None):
    """Return a list of problems with one tool's install (empty list = fine). tool is a key like depth_vda."""
    problems = []
    install, py = tool_install(settings, tool), tool_python(settings, tool)
    if not os.path.isdir(install):
        problems.append(f"folder not found: {install}")
    elif not os.path.isfile(py):
        problems.append(f"no environment found (expected {py})")
    if tool == "flow_waft" and os.path.isdir(install):
        for f in ["ckpts/waft_a1.pth", "depth-anything-ckpts/depth_anything_v2_vits.pth"]:
            if not os.path.isfile(os.path.join(install, f)):
                problems.append(f"missing model file: {os.path.join(install, f)}")
    if tool == "depth_vda" and os.path.isdir(install):
        enc = encoder or settings["defaults"].get("vda_encoder", "vits")
        f = os.path.join(install, "checkpoints", f"video_depth_anything_{enc}.pth")
        if not os.path.isfile(f):
            problems.append(f"missing model file: {f}")
    return problems


def run_check(settings):
    print(f"ML Video Tools v{version()}")
    print(f"Settings file: {SETTINGS_PATH}" + ("" if os.path.isfile(SETTINGS_PATH) else "  (not found - using defaults)"))
    print(f"ffmpeg: {'found' if shutil.which('ffmpeg') else 'NOT FOUND - install it and add it to your PATH'}")
    all_ok = bool(shutil.which("ffmpeg"))
    imports = {"depth_vda": "video_depth_anything.video_depth", "depth_da3": "depth_anything_3",
               "flow_memfof": "memfof, OpenEXR", "flow_waft": "OpenEXR", "mattes": "sam3"}
    checks = []
    for tool in TOOL_ORDER:
        if tool in ENGINES:
            d = default_engine(settings, tool)
            checks.append((engine_key(tool, d), f"{tool} (default)"))
            checks += [(engine_key(tool, e), f"{tool} (optional)") for e in ENGINES[tool] if e != d]
        else:
            checks.append((tool, tool))
    for t, label in checks:
        problems = check_tool(settings, t)
        optional = "optional" in label
        if optional and not os.path.isdir(tool_install(settings, t)):
            tool, _, eng = t.partition("_")
            print(f"\n{label:17s} not installed (only needed with --{tool}-engine {eng})")
            continue
        if not problems:
            code = (f"import torch, {imports[t]}; "
                    f"print(torch.__version__, 'GPU' if torch.cuda.is_available() else 'NO GPU')")
            r = subprocess.run([tool_python(settings, t), "-c", code], capture_output=True, text=True,
                               cwd=tool_install(settings, t))
            if r.returncode != 0:
                problems.append("environment won't load: " + (r.stderr.strip().splitlines() or ["?"])[-1])
            elif "NO GPU" in r.stdout:
                problems.append(f"PyTorch can't see the GPU ({r.stdout.strip()})")
        status = "OK" if not problems else "PROBLEM"
        name = engine_label(t)
        print(f"\n{label:17s} {status}   {tool_install(settings, t)}" + (f"   ({name})" if name else ""))
        for pr in problems:
            print(f"        - {pr}")
        all_ok = all_ok and not problems
    print("\nAll good." if all_ok else "\nFix the problems above (see INSTALL.md), then run --check again.")
    return all_ok


def main():
    windows_tweaks()
    p = argparse.ArgumentParser(description="Run depth, flow and mattes on a clip, extracting frames once.")
    p.add_argument("video", nargs="?")
    p.add_argument("--check", action="store_true", help="check settings.json and the three installs, then stop")
    p.add_argument("--version", action="version", version=f"ML Video Tools v{version()}")
    p.add_argument("--prompt", action="append", default=[],
                   help="what SAM should matte, e.g. --prompt person. Repeat for more. Leave out to skip mattes")
    p.add_argument("--only", choices=TOOL_ORDER, help="run just one tool")
    p.add_argument("--skip", action="append", choices=TOOL_ORDER, default=[], help="leave a tool out, e.g. --skip flow")
    p.add_argument("--redo", action="append", choices=TOOL_ORDER, default=[],
                   help="throw away one tool's results and run it again, e.g. --redo depth (keeps the frames)")
    p.add_argument("--fresh", action="store_true", help="delete this clip's whole shot folder and start again")
    p.add_argument("--delete-frames", action="store_true", help="delete the extracted frames at the end")
    # options handed on to the tools
    p.add_argument("--depth-engine", choices=list(ENGINES["depth"]),
                   help="depth: vda = Video Depth Anything (default), da3 = Depth Anything 3")
    p.add_argument("--vda-encoder", choices=["vits", "vitb", "vitl"],
                   help="depth with vda: vits = Small (default, commercial OK), vitb / vitl = Base / Large (non-commercial)")
    p.add_argument("--depth-chunk", type=int, help="depth with vda: frames per chunk (default 320, lower = less memory)")
    p.add_argument("--depth-model", help="depth with da3: e.g. depth-anything/DA3MONO-LARGE")
    white = p.add_mutually_exclusive_group()
    white.add_argument("--far-white", action="store_true", help="depth: far = white, near = black (the default)")
    white.add_argument("--near-white", action="store_true", help="depth: near = white, far = black")
    p.add_argument("--flow-engine", choices=list(ENGINES["flow"]),
                   help="flow: memfof = MEMFOF (default), waft = WAFT")
    p.add_argument("--memfof-model", choices=["tskh", "spring", "sintel", "kitti"],
                   help="flow with memfof: tskh = real-world (default), spring = tuned for fine detail")
    p.add_argument("--flow-iters", type=int, help="flow: refinement steps per frame (MEMFOF default 8, WAFT 5)")
    p.add_argument("--half-res", action="store_true", help="flow with waft: work at half size (faster, lighter)")
    p.add_argument("--backward", action="store_true", help="flow: motion to the previous frame instead of the next")
    p.add_argument("--flow-format", choices=["png16", "exr", "both"],
                   help="flow with memfof: both = 16-bit PNG + 32-bit EXR (default), png16 = PNG only, exr = EXR only")
    p.add_argument("--flow-range", type=float,
                   help="flow png16: fixed top speed in pixels per frame (default: worked out per clip)")
    p.add_argument("--threshold", type=float, help="mattes: how sure SAM must be, 0-1 (default 0.5)")
    p.add_argument("--max-objects", type=int, help="mattes: most things tracked per word (default 16)")
    p.add_argument("--sam-chunk", type=int, help="mattes: frames per chunk (default 200, lower = less memory)")
    a = p.parse_args()

    settings = load_settings()
    if a.check:
        sys.exit(0 if run_check(settings) else 1)
    if not a.video:
        p.error("give me a clip, e.g.  python process_clip.py \"D:\\clips\\test.mp4\" --prompt person")

    video = os.path.abspath(os.path.join(START_DIR, a.video))
    if not os.path.isfile(video):
        sys.exit(f"Can't find the clip: {video}")
    if shutil.which("ffmpeg") is None:
        sys.exit("Can't find ffmpeg. It needs to be installed and on your PATH.")

    name = os.path.splitext(os.path.basename(video))[0]
    shot = os.path.join(os.path.dirname(video), f"{name}_shot")
    if a.fresh:
        shutil.rmtree(shot, ignore_errors=True)
    os.makedirs(shot, exist_ok=True)
    log = Log(os.path.join(shot, "log.txt"))
    log(f"===== {name}  ({time.strftime('%Y-%m-%d %H:%M')})  ML Video Tools v{version()} =====")
    rename_old_folders(shot, log)
    folder = lambda t: os.path.join(shot, FOLDERS[t])
    frames_dir = folder("frames")
    os.makedirs(frames_dir, exist_ok=True)

    wanted = [a.only] if a.only else [t for t in TOOL_ORDER if t not in a.skip]
    if "mattes" in wanted and not a.prompt:
        log("No --prompt given, so skipping mattes. (Add e.g. --prompt person to get mattes.)")
        wanted.remove("mattes")
    if not wanted:
        sys.exit("Nothing to do.")
    engines = {"depth": a.depth_engine or default_engine(settings, "depth"),
               "flow": a.flow_engine or default_engine(settings, "flow")}
    encoder = a.vda_encoder or settings["defaults"].get("vda_encoder", "vits")
    far_white = a.far_white or (not a.near_white and settings["defaults"].get("depth_white", "far") == "far")
    keys = {t: tool_key(settings, t, engines.get(t)) for t in wanted}
    if a.half_res and keys.get("flow") == "flow_memfof":
        log("Note: --half-res only applies to WAFT, so it's ignored with MEMFOF.")
    if (a.flow_format or a.flow_range) and keys.get("flow") == "flow_waft":
        log("Note: WAFT always writes EXR, so --flow-format / --flow-range are ignored with it.")
    for t in wanted:
        problems = check_tool(settings, keys[t], encoder)
        if problems:
            log(f"The {t} tool ({keys[t]}) isn't ready:")
            for pr in problems:
                log(f"  - {pr}")
            log(f"Check the {keys[t]} folder in settings.json, or run  python process_clip.py --check")
            sys.exit(1)

    # Shots made by an older version used different file names. Move those results aside
    # (into _old_naming, nothing is deleted) so the new names don't end up mixed with them.
    naming_note = os.path.join(shot, "_naming.txt")
    if not os.path.isfile(naming_note):
        old = [t for t in TOOL_ORDER if os.path.isdir(folder(t))]
        if old:
            aside = os.path.join(shot, "_old_naming")
            os.makedirs(aside, exist_ok=True)
            for t in old:
                shutil.move(folder(t), os.path.join(aside, f"{t}_{time.strftime('%Y%m%d_%H%M%S')}"))
            log(f"This shot was made with the old file names - moved its {', '.join(old)} results "
                f"into _old_naming and doing them again with the new names.")
        old_frames = [f for f in glob.glob(os.path.join(frames_dir, "*.png"))
                      if not os.path.basename(f).startswith(f"{name}_")]
        if old_frames:
            for f in old_frames:
                os.remove(f)
            try:
                os.remove(os.path.join(frames_dir, "_extracted.txt"))
            except OSError:
                pass
        open(naming_note, "w").write("frames: <clip>_0001.png   outputs: <clip>_<process>_0001.<ext>\n")

    # 1. Extract frames once
    marker = os.path.join(frames_dir, "_extracted.txt")
    if os.path.isfile(marker):
        log("Frames already extracted - using them.")
    else:
        log("Extracting frames (once, for all tools)...")
        for f in glob.glob(os.path.join(frames_dir, "*.png")):
            os.remove(f)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                        "-start_number", "1", os.path.join(frames_dir, f"{name}_%04d.png")], check=True)
        open(marker, "w").write("ok")
    log(f"{len(glob.glob(os.path.join(frames_dir, '*.png')))} frames ready.")

    for t in a.redo:
        if t in wanted:
            shutil.rmtree(folder(t), ignore_errors=True)
            log(f"Throwing away the old {t} results so they're done again.")

    # 2. Run each tool with its own environment's Python
    results = {}
    for t in wanted:
        args = ["--frames", frames_dir, "--out", folder(t)]
        key = keys[t]
        if key == "depth_vda":
            args += ["--install", tool_install(settings, key), "--encoder", encoder]
            args += ["--far-white"] if far_white else []
            args += ["--chunk", str(a.depth_chunk)] if a.depth_chunk else []
        elif key == "depth_da3":
            args += ["--model", a.depth_model or settings["defaults"].get("depth_model", "depth-anything/DA3-BASE")]
            args += ["--far-white"] if far_white else []
        elif key == "flow_memfof":
            args += ["--model", a.memfof_model or settings["defaults"].get("memfof_model", "tskh")]
            args += ["--iters", str(a.flow_iters)] if a.flow_iters else []
            args += ["--backward"] if a.backward else []
            args += ["--format", a.flow_format or settings["defaults"].get("flow_format", "both")]
            args += ["--range", str(a.flow_range)] if a.flow_range else []
        elif key == "flow_waft":
            args += ["--install", tool_install(settings, key)]
            args += ["--half-res"] if a.half_res else []
            args += ["--backward"] if a.backward else []
            args += ["--iters", str(a.flow_iters)] if a.flow_iters else []
        elif t == "mattes":
            for pr in a.prompt:
                args += ["--prompt", pr]
            if a.threshold is not None:
                args += ["--threshold", str(a.threshold)]
            if a.max_objects is not None:
                args += ["--max-objects", str(a.max_objects)]
            if a.sam_chunk is not None:
                args += ["--chunk", str(a.sam_chunk)]
        what = f"{engine_label(key)}, " if engine_label(key) else ""
        log(f"\n========== {t.upper()} ({what}{tool_install(settings, key)}) ==========")
        t0 = time.time()
        okay = babysit(t, [tool_python(settings, key), tool_script(key)] + args,
                       cwd=tool_install(settings, key), log=log, settings=settings)
        results[t] = ("done" if okay else "FAILED") + f" ({(time.time() - t0) / 60:.0f} min)"
        log(f"{t}: {results[t]}")

    # 3. Summary
    log("\n========== SUMMARY ==========")
    for t, r in results.items():
        log(f"  {t:7s} {r}")
    log(f"Everything is in: {shot}")
    if a.delete_frames and all(r.startswith("done") for r in results.values()):
        shutil.rmtree(frames_dir, ignore_errors=True)
        log("Deleted the extracted frames.")
    if any(r.startswith("FAILED") for r in results.values()):
        log("Something failed - scroll up (or open log.txt) for the error.\n"
            "If it crashed, run the same command again and it'll pick up where it stopped.\n"
            "If it says you changed a setting, add --redo and the tool's name, e.g. --redo depth")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to carry on.")
