r"""
Writes the EXACT package versions from your working installs into requirements\*.txt,
so anyone else (or you, on a new PC) gets exactly the same, known-good set.

Run it on a machine where all three tools work:
    python lock_versions.py

Re-run it whenever you update a tool and it still works.
"""
import os, sys, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.common import load_settings, tool_python, tool_install, REPO_ROOT

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Packages that are installed from the tool's own folder, not from the internet
LOCAL_PACKAGES = {"depth_vda": None, "depth_da3": "depth-anything-3", "flow_memfof": "memfof", "flow_waft": None, "mattes": "sam3"}


def lock(settings, tool):
    py = tool_python(settings, tool)
    if not os.path.isfile(py):
        print(f"{tool:11s} skipped - no environment at {py}")
        return False
    r = subprocess.run([py, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tool:11s} FAILED - {r.stderr.strip()}")
        return False
    keep, skipped = [], []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0].split(" @ ")[0].strip().lower().replace("_", "-")
        if name == (LOCAL_PACKAGES[tool] or "") or " @ file:" in line:
            skipped.append(line)  # installed from a local folder - the installer does that step itself
            continue
        keep.append(line)
    out = os.path.join(REPO_ROOT, "requirements", f"{tool}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Exact versions from a working {tool} install ({tool_install(settings, tool)})\n")
        f.write(f"# Locked {time.strftime('%Y-%m-%d')} by lock_versions.py - re-run it to update.\n")
        f.write(f"# PyTorch comes from its own site (the +cu128 builds):\n")
        f.write(f"--extra-index-url {TORCH_INDEX}\n\n")
        f.write("\n".join(sorted(keep, key=str.lower)) + "\n")
    print(f"{tool:11s} locked {len(keep)} packages -> requirements\\{tool}.txt"
          + (f"  (left out {len(skipped)} local: {', '.join(s.split('==')[0].split(' @ ')[0] for s in skipped)})" if skipped else ""))
    return True


if __name__ == "__main__":
    s = load_settings()
    results = [lock(s, t) for t in ("depth_vda", "depth_da3", "flow_memfof", "flow_waft", "mattes")]
    print("\nDone." if all(results) else "\nSome tools were skipped - see above.")
