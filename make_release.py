"""
Makes a versioned zip of the toolkit to share:  ml-video-tools_v<VERSION>.zip

    python make_release.py

Leaves out your own settings.json, so people get settings.example.json instead.
The zip is made next to this folder (one level up).
"""
import os, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.common import REPO_ROOT, version

SKIP_FILES = {"settings.json"}
SKIP_DIRS = {"__pycache__", ".git"}

v = version()
if v == "unknown":
    sys.exit("No VERSION file - put a version number like 0.2.0 in a file called VERSION.")

folder = os.path.basename(REPO_ROOT)  # the zip unpacks into a folder of this name
out = os.path.join(os.path.dirname(REPO_ROOT), f"{folder}_v{v}.zip")
if os.path.exists(out):
    sys.exit(f"{os.path.basename(out)} already exists. Bump the number in VERSION first "
             f"(and add a line to CHANGELOG.md), or delete the old zip.")

count = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if f in SKIP_FILES or f.endswith(".pyc"):
                continue
            full = os.path.join(root, f)
            z.write(full, os.path.join(folder, os.path.relpath(full, REPO_ROOT)))
            count += 1
print(f"Made {out}  ({count} files, version {v})")
