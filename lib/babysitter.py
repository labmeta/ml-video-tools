"""
The babysitter: runs one tool and keeps it going.

  - if the tool crashes, it starts it again (the tools resume from where they got to)
  - if the tool freezes (no heartbeat for a while), it kills it and starts it again
  - if the tool stops on purpose (exit code 3: missing file, changed setting), it doesn't retry
  - if it keeps failing without getting any frames done, it gives up

Every tool script calls lib.common.beat() as it works, which is how we know it's alive.
"""
import os
import time
import tempfile
import subprocess

from lib.common import HEARTBEAT_ENV, EXIT_NO_RETRY


def _kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _read_beat(path):
    try:
        with open(path) as f:
            return f.read().strip(), os.path.getmtime(path)
    except OSError:
        return None, None


def babysit(label, cmd, cwd, log=print, settings=None):
    """Run cmd until it finishes. Returns True if it finished, False if it gave up."""
    s = (settings or {}).get("babysitter", {})
    frame_timeout = s.get("frame_timeout_sec", 300)
    setup_timeout = s.get("setup_timeout_sec", 3600)
    max_fails = s.get("max_fails_in_a_row", 10)
    wait = s.get("wait_before_retry_sec", 10)

    heartbeat = os.path.join(tempfile.gettempdir(), f"clip_tools_heartbeat_{os.getpid()}_{label}.txt")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env[HEARTBEAT_ENV] = heartbeat

    attempt, fails, last_mark = 0, 0, None
    try:
        while True:
            attempt += 1
            if attempt > 1:
                log(f"--- {label}: restarting (attempt {attempt}) ---")
            try:
                os.remove(heartbeat)
            except OSError:
                pass

            started = time.time()
            proc = subprocess.Popen(cmd, cwd=cwd, env=env)
            frozen = False
            while proc.poll() is None:
                time.sleep(5)
                stage, stamp = _read_beat(heartbeat)
                limit = frame_timeout if stage == "frame" else setup_timeout
                if time.time() - (stamp or started) > limit:
                    log(f"\n--- {label}: no progress for {limit // 60} min, looks frozen ---")
                    _kill_tree(proc)
                    frozen = True
                    break

            if not frozen and proc.returncode == 0:
                return True
            if not frozen and proc.returncode == EXIT_NO_RETRY:
                log(f"--- {label}: stopped - see the message above. Restarting won't fix this one. ---")
                return False

            # Count failures in a row that got no frames done
            stage, stamp = _read_beat(heartbeat)
            if stage == "frame" and stamp != last_mark:
                last_mark, fails = stamp, 0
            else:
                fails += 1
            if not frozen:
                log(f"\n--- {label}: stopped with an error (code {proc.returncode}) ---")
            if fails >= max_fails:
                log(f"--- {label}: gave up after {fails} attempts in a row with no progress ---")
                return False
            log(f"Trying {label} again in {wait} seconds... (Ctrl+C to stop)")
            time.sleep(wait)
    finally:
        try:
            os.remove(heartbeat)
        except OSError:
            pass
