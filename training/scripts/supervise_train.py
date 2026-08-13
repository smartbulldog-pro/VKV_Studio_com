"""Overnight training supervisor for E4B then E2B on synapse_v2.jsonl (v2.4, ~5K).
Survives: (a) intermittent native segfaults / hangs (crash-resume from checkpoints),
(b) Windows VRAM->sysmem fragmentation that slows step time (proactive restart every
RESTART_EVERY steps → fresh process resets the allocator), and (c) INCOMPLETE checkpoints
left by a kill that landed mid-save (auto-deleted before each attempt so resume always
picks a complete checkpoint; proactive kill also waits a few seconds to let a save flush)."""
import sys, time, subprocess, glob, re, shutil
from pathlib import Path

TRAINING = Path(r"C:\projects\VKVstudio\training")
PY = str(TRAINING / ".venv-cuda" / "Scripts" / "python.exe")
TRAIN = str(TRAINING / "scripts" / "04_train.py")
VARIANTS = ["E4B", "E2B"]
HANG_SECS = 300
RESTART_EVERY = 50
MAX_FAST_FAILS = 12
POLL = 20
SAVE_FLUSH_WAIT = 15   # sec to let an in-progress checkpoint save finish before a proactive kill


def ok_path(v):
    return TRAINING / "output" / f"synapse-v2-{v.lower()}-lora" / "TRAIN_OK"


def log_path(v):
    return TRAINING / f"train_{v.lower()}.log"


def ckpt_dir(v):
    return TRAINING / "checkpoints" / v.lower()


def latest_step(v):
    steps = []
    for c in glob.glob(str(ckpt_dir(v) / "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)", c.replace("\\", "/"))
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else 0


def clean_incomplete(v):
    """Delete any checkpoint dir missing trainer_state.json/optimizer.pt — a partial
    save (e.g. killed mid-write) would make resume crash-loop."""
    for c in glob.glob(str(ckpt_dir(v) / "checkpoint-*")):
        p = Path(c)
        if not (p / "trainer_state.json").exists() or not (p / "optimizer.pt").exists():
            print(f"[{v}] removing INCOMPLETE checkpoint {p.name}", flush=True)
            shutil.rmtree(p, ignore_errors=True)


def run_once(v):
    clean_incomplete(v)
    LOG = log_path(v)
    start = latest_step(v)
    t0 = time.time()
    logf = open(LOG, "a", encoding="utf-8", buffering=1)
    logf.write(f"\n===== [{v}] attempt @ {time.strftime('%Y-%m-%d %H:%M:%S')} start_step={start} =====\n")
    p = subprocess.Popen([PY, TRAIN, "--model", v], cwd=str(TRAINING), stdout=logf, stderr=subprocess.STDOUT)
    reason = "exit"
    while p.poll() is None:
        time.sleep(POLL)
        try:
            idle = time.time() - LOG.stat().st_mtime
        except OSError:
            idle = 0
        cur = latest_step(v)
        if idle > HANG_SECS:
            logf.write(f"\n[supervisor] HANG {int(idle)}s idle — killing child\n")
            logf.flush()
            p.kill()
            reason = "hang"
            break
        if cur - start >= RESTART_EVERY:
            # let any in-progress checkpoint save flush so we don't corrupt it
            logf.write(f"\n[supervisor] proactive defrag restart at step {cur}; waiting {SAVE_FLUSH_WAIT}s for save to flush\n")
            logf.flush()
            time.sleep(SAVE_FLUSH_WAIT)
            p.kill()
            reason = "restart"
            break
    try:
        p.wait(timeout=60)
    except Exception:
        pass
    logf.close()
    ran = time.time() - t0
    progressed = latest_step(v) > start
    return reason, ran, progressed


for v in VARIANTS:
    OK = ok_path(v)
    fast_fails = 0
    att = 0
    while not OK.exists():
        att += 1
        print(f"[{v}] attempt {att} @ {time.strftime('%H:%M:%S')} step={latest_step(v)}", flush=True)
        reason, ran, progressed = run_once(v)
        if OK.exists():
            break
        if reason == "restart" or progressed:
            fast_fails = 0
            print(f"[{v}] {reason} after {int(ran)}s (step={latest_step(v)}) — resuming", flush=True)
        elif ran < 90:
            fast_fails += 1
            print(f"[{v}] FAST-FAIL {fast_fails}/{MAX_FAST_FAILS} (ran {int(ran)}s, no progress)", flush=True)
            if fast_fails >= MAX_FAST_FAILS:
                print(f"[{v}] GIVING UP — {MAX_FAST_FAILS} consecutive fast-fails", flush=True)
                sys.exit(1)
        else:
            # a >90s attempt that made no progress = likely a corrupt-latest-checkpoint loop;
            # clean_incomplete() at next run_once will drop it and resume from an earlier one.
            fast_fails = 0
            print(f"[{v}] no-progress attempt ({int(ran)}s) — will clean incomplete checkpoints and resume", flush=True)
        time.sleep(10)
    print(f"[{v}] TRAIN_OK (step={latest_step(v)})", flush=True)

print("ALL VARIANTS TRAINED OK", flush=True)
