"""Lustre-friendly sharded directory queue for DQMC jobs.

Uses atomic rename (os.replace) instead of lock files for robustness
on distributed filesystems like Lustre.
"""
import glob
import hashlib
import os
import sys
from pathlib import Path

NSHARDS = 128


def get_shard_entry(fullpath: str) -> tuple[str, str]:
    """Compute shard and entry name using SHA1 of the path."""
    h = hashlib.sha1(fullpath.encode())
    shard_val = h.digest()[0] % NSHARDS
    shard = f"{shard_val:02x}"
    entry = f"{os.path.basename(fullpath)}__{h.hexdigest()[:16]}"
    return shard, entry


# --- Enqueue ---

def enqueue_main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 3:
        print(f"usage: {argv[0]} <queue_dir> <glob1> [glob2 ...]", file=sys.stderr)
        sys.exit(2)

    queue_root = Path(argv[1])
    patterns = argv[2:]

    for state in ("todo", "running", "done"):
        for i in range(NSHARDS):
            (queue_root / state / f"{i:02x}").mkdir(parents=True, exist_ok=True)

    # 1. Collect and deduplicate paths
    paths = set()
    for pat in patterns:
        paths.update(glob.glob(pat))

    # 2. Group by shard to minimize directory listings
    by_shard = {}
    for p in paths:
        if not p.endswith(".h5"):
            continue
        full = os.path.realpath(p)
        if not os.path.isfile(full):
            continue

        shard, entry = get_shard_entry(full)

        if shard not in by_shard:
            by_shard[shard] = []
        by_shard[shard].append((full, entry))

    enqueued = 0
    skipped = 0

    # 3. Process each shard
    for shard, items in by_shard.items():
        # Check running/done to avoid duplicates
        # We need listdir to find running jobs with suffixes
        exclude = set()
        for state in ("running", "done"):
            d = queue_root / state / shard
            if d.exists():
                exclude.update(p.name.split(".__")[0] for p in d.iterdir())

        for full, entry in items:
            if entry in exclude:
                skipped += 1
                continue

            try:
                os.symlink(full, queue_root / "todo" / shard / entry)
                enqueued += 1
            except FileExistsError:
                skipped += 1

    print(f"enqueued={enqueued} skipped={skipped}")


# --- Status ---

def status_main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 2:
        print(f"usage: {argv[0]} <queue_dir>", file=sys.stderr)
        sys.exit(2)

    queue_root = Path(argv[1])
    counts = {"todo": 0, "running": 0, "done": 0}

    for state in counts:
        for i in range(NSHARDS):
            d = queue_root / state / f"{i:02x}"
            try:
                counts[state] += sum(1 for _ in d.iterdir())
            except FileNotFoundError:
                pass

    print(f"todo={counts['todo']} running={counts['running']} done={counts['done']}")


# --- Dequeue (remove from queue) ---

def dequeue_main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 3:
        print(f"usage: {argv[0]} <queue_dir> <glob1> [glob2 ...]", file=sys.stderr)
        sys.exit(2)

    queue_root = Path(argv[1])
    patterns = argv[2:]

    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    removed = notfound = 0
    for p in sorted(set(paths)):
        if not p.endswith(".h5"):
            continue
        full = os.path.realpath(p)
        shard, entry = get_shard_entry(full)

        link = queue_root / "todo" / shard / entry
        try:
            link.unlink()
            removed += 1
        except FileNotFoundError:
            notfound += 1

    print(f"removed={removed} not_in_todo={notfound}")


if __name__ == "__main__":
    enqueue_main()
