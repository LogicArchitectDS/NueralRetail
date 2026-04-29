#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/seshu/NueralRetail_Solo")
SYNC_SCRIPT = ROOT / "scripts" / "sync_to_windows.sh"
LOCK_FILE = Path("/tmp/neuralretail_sync_watcher.lock")
LOG_PREFIX = "[sync-watcher]"
POLL_SECONDS = 2.0

EXCLUDED_DIRS = {
    ".git",
    ".planning",
    ".venv",
    "node_modules",
    "data",
    "__pycache__",
    ".pytest_cache",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}

EXCLUDED_FILES = {
    ".env",
}

WATCH_ROOTS = [
    ROOT / "src",
    ROOT / "config",
    ROOT / "docker",
    ROOT / "tests",
    ROOT / "scripts",
    ROOT / "Makefile",
    ROOT / "docker-compose.yml",
    ROOT / "pyproject.toml",
    ROOT / "poetry.lock",
    ROOT / "README.md",
    ROOT / ".gitignore",
    ROOT / ".env.example",
]

running = True


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def handle_signal(signum: int, _frame) -> None:
    global running
    log(f"received signal {signum}, shutting down")
    running = False


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDED_FILES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return any(part in EXCLUDED_DIRS for part in path.parts)


def snapshot() -> str:
    hasher = hashlib.sha256()

    for root in WATCH_ROOTS:
        if not root.exists():
            continue

        if root.is_file():
            if should_skip(root):
                continue
            stat = root.stat()
            hasher.update(str(root.relative_to(ROOT)).encode())
            hasher.update(str(stat.st_mtime_ns).encode())
            hasher.update(str(stat.st_size).encode())
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)
            dirnames[:] = [
                name for name in dirnames if name not in EXCLUDED_DIRS
            ]
            dirnames.sort()

            for filename in sorted(filenames):
                file_path = current_dir / filename
                if should_skip(file_path):
                    continue
                stat = file_path.stat()
                hasher.update(str(file_path.relative_to(ROOT)).encode())
                hasher.update(str(stat.st_mtime_ns).encode())
                hasher.update(str(stat.st_size).encode())

    return hasher.hexdigest()


def run_sync() -> bool:
    log("changes detected, syncing to Windows mirror")
    result = subprocess.run([str(SYNC_SCRIPT)], cwd=ROOT)
    if result.returncode == 0:
        log("sync completed")
        return True
    log(f"sync failed with exit code {result.returncode}")
    return False


def main() -> int:
    if not SYNC_SCRIPT.exists():
        log(f"sync script not found: {SYNC_SCRIPT}")
        return 1

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another watcher instance is already running")
            return 0

        lock_handle.write(str(os.getpid()))
        lock_handle.flush()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        log("starting initial sync")
        if not run_sync():
            return 1

        previous = snapshot()
        log("watching for Ubuntu repo changes")

        while running:
            time.sleep(POLL_SECONDS)
            current = snapshot()
            if current == previous:
                continue

            if run_sync():
                previous = snapshot()
            else:
                previous = current

        log("watcher stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
