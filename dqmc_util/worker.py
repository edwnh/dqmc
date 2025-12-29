"""Worker process for Lustre-friendly sharded directory queue.

Consumes tasks from the queue created by dqmc_util.queue.
"""
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

# Must match dqmc_util.queue.NSHARDS
NSHARDS = 128
MAX_SLEEP_INTERVAL = 4


def claim_one(queue_root: Path, unique_id: str):
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
            dst = running / f"{ent.name}.__{unique_id}_{int(time.time())}"
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
        print(f"usage: {argv[0]} <queue_dir> <dqmc_bin> [-n num_workers] [-s save_interval] [-t max_time]",
              file=sys.stderr)
        sys.exit(2)

    queue_root = Path(argv[1])
    dqmc_bin = argv[2]

    # Parse optional flags
    save_interval = 0
    max_time = 0
    num_workers = 1
    i = 3
    while i < len(argv):
        if argv[i] == "-s" and i + 1 < len(argv):
            save_interval = int(argv[i + 1])
            i += 2
        elif argv[i] == "-t" and i + 1 < len(argv):
            max_time = int(argv[i + 1])
            i += 2
        elif argv[i] == "-n" and i + 1 < len(argv):
            num_workers = int(argv[i + 1])
            i += 2
        else:
            i += 1

    if num_workers > 1:
        run_parallel(queue_root, dqmc_bin, num_workers, save_interval, max_time)
    else:
        _worker_loop(queue_root, dqmc_bin, save_interval, max_time)


def run_parallel(queue_root: Path, dqmc_bin: str, num_workers: int,
                 save_interval: int = 0, max_time: int = 0):
    """Run multiple workers in parallel with signal forwarding."""
    # Use 'spawn' to avoid fork issues with MPI and ensure clean child state
    ctx = multiprocessing.get_context("spawn")
    workers: list[multiprocessing.Process] = []
    
    def spawn_worker():
        p = ctx.Process(target=_worker_loop,
                        args=(queue_root, dqmc_bin, save_interval, max_time))
        p.start()
        return p
    
    # Spawn initial workers
    for _ in range(num_workers):
        workers.append(spawn_worker())
    
    # Forward signals to all workers
    received_signal = [None]
    
    def forward_signal(signum, frame):
        received_signal[0] = signum
        for w in workers:
            if w.is_alive():
                try:
                    os.kill(w.pid, signum)
                except (ProcessLookupError, OSError):
                    pass
    
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        old_handlers[sig] = signal.signal(sig, forward_signal)
    
    try:
        # Wait for all workers to finish
        for w in workers:
            w.join()
    finally:
        # Restore original handlers
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    
    # If we received a signal, exit with appropriate code
    if received_signal[0] is not None:
        sys.exit(128 + received_signal[0])


def _worker_loop(queue_root: Path, dqmc_bin: str, save_interval: int, max_time: int):
    """Worker loop for multiprocessing.Process - wraps main() logic."""
    # Seed random uniquely for this process (important for MPI + multiprocessing)
    # Combine PID, time with nanoseconds, and hostname hash for uniqueness
    seed = os.getpid() ^ int(time.time() * 1e9) ^ hash(os.uname().nodename)
    random.seed(seed)
    
    unique_id = f"{os.uname().nodename}.{os.getpid()}"
    hostname = os.uname().nodename
    pid = os.getpid()
    t_start = time.time()

    def log(msg):
        print(f"{hostname:>16} {pid:6d}: {msg}", flush=True)

    # Initial jitter to desynchronize workers (scaled by number of potential workers)
    time.sleep(random.uniform(0, MAX_SLEEP_INTERVAL))

    while True:
        # Check time limit before claiming new work
        remaining = int(t_start + max_time - time.time()) if max_time > 0 else None
        if remaining is not None and remaining <= 0:
            log("time limit reached; exiting")
            break

        claimed = claim_one(queue_root, unique_id)
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

        # Run in a new process group so we can forward signals
        proc = subprocess.Popen(cmd, start_new_session=True)
        
        # Set up signal forwarding to child
        def forward_signal(signum, frame):
            try:
                os.killpg(proc.pid, signum)
            except (ProcessLookupError, OSError):
                pass
        
        old_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            old_handlers[sig] = signal.signal(sig, forward_signal)
        
        rc = proc.wait()
        
        # Restore original handlers
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

        done_dir = queue_root / "done" / shard
        todo_dir = queue_root / "todo" / shard
        original_entry = claimed.name.split(".__", 1)[0]

        if rc == 1:  # Checkpointed: return to todo queue and exit worker
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
