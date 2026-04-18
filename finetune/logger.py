"""
Shared logging utility: tee stdout to a log file.

Usage:
    from finetune.logger import setup_logger
    setup_logger("finetune/logs/run.txt")   # call once at start of main()
    print("...")   # goes to both console and file
"""
import sys
from pathlib import Path


class _Tee:
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return False


def setup_logger(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(f)
    print(f"[logger] Output saved to: {log_path}")
