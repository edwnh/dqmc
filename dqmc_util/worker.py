"""Worker process for Lustre-friendly sharded directory queue.

Consumes tasks from the queue created by dqmc_util.queue.
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Must match dqmc_util.queue.NSHARDS
NSHARDS = 128
MAX_SLEEP_INTERVAL = 4


def claim_one(queue_root: Path, pid: int):
    """Atomically claim one task. Returns path to claimed entry or None."""
    arr = [f"{i:02x}" for i in range(NSHARDS)]
    random.shuffle(arr)
    for sh in arr:
        todo = queue_root / "todo" / sh
        running = queue_root / "running" / sh
        try:
            entries = list(todo.iterdir())
        except FileNotFoundError:
            continue
        random.shuffle(entries)
        for ent in entries:
            if not ent.is_symlink():
                continue
            dst = running / f"{ent.name}.__pid{pid}_{int(time.time())}"
            try:
                os.replace(ent, dst)
                return dst
            except (FileNotFoundError, OSError):
                continue
    return None


def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 3:
        print(f"usage: {argv[0]} <queue_dir> <dqmc_bin> [-s save_interval] [-t max_time]",
              file=sys.stderr)
        sys.exit(2)

    queue_root = Path(argv[1])
    dqmc_bin = argv[2]

    # Parse optional -s and -t flags for the worker
    save_interval = 0
    max_time = 0
    i = 3
    while i < len(argv):
        if argv[i] == "-s" and i + 1 < len(argv):
            save_interval = int(argv[i + 1])
            i += 2
        elif argv[i] == "-t" and i + 1 < len(argv):
            max_time = int(argv[i + 1])
            i += 2
        else:
            i += 1

    pid = os.getpid()
    hostname = os.uname().nodename
    t_start = time.time()

    def log(msg):
        print(f"{hostname:>16} {pid:6d}: {msg}", flush=True)

    # Initial jitter to desynchronize workers
    time.sleep(random.uniform(0, MAX_SLEEP_INTERVAL))

    while True:
        # Check time limit before claiming new work
        remaining = int(t_start + max_time - time.time()) if max_time > 0 else None
        if remaining is not None and remaining <= 0:
            log("time limit reached; exiting")
            break

        claimed = claim_one(queue_root, pid)
        if claimed is None:
            log("queue empty; exiting")
            break

        shard = claimed.parent.name
        h5_path = os.path.realpath(claimed)
        log_path = h5_path + ".log"
        log(f"starting: {h5_path}")

        cmd = [dqmc_bin, "-l", log_path]
        if save_interval > 0:
            cmd += ["-s", str(save_interval)]
        if remaining is not None:
            cmd += ["-t", str(remaining)]
        cmd.append(h5_path)

        rc = subprocess.call(cmd)

        done_dir = queue_root / "done" / shard
        todo_dir = queue_root / "todo" / shard
        original_entry = claimed.name.split(".__", 1)[0]

        if rc == 1: # Checkpointed: return to todo queue and exit worker
            log(f"checkpointed: {h5_path}")
            os.replace(claimed, todo_dir / original_entry)
            break
        else:
            if rc == 0:
                log(f"completed: {h5_path}")
            else:
                log(f"error ({rc}): {h5_path}")
            os.replace(claimed, done_dir / claimed.name)


if __name__ == "__main__":
    main()
