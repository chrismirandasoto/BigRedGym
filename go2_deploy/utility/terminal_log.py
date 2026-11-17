"""Copy go2_deploy's terminal output into the run directory; record how it ended.

terminal.log: everything printed to stdout/stderr, each line time-stamped.
exit.json: reason, end time, state and uptime. Written for a damped shutdown
    (deploy.py), Ctrl+C, uncaught exceptions and normal exit. A run directory
    without one was SIGKILLed or lost power.
"""

import atexit
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime


class _Tee:
    """Writes to the terminal and to terminal.log, stamping each new line."""

    def __init__(self, stream, file, lock):
        self._stream, self._file, self._lock = stream, file, lock

    def write(self, text):
        try:
            self._stream.write(text)
        except (OSError, ValueError):  # terminal gone (tmux kill-session): keep logging
            pass
        with self._lock:
            for piece in text.splitlines(keepends=True):
                if self._file.at_line_start:
                    self._file.write(f"[{datetime.now():%H:%M:%S.%f}"[:-3] + "] ")
                self._file.write(piece)
                self._file.at_line_start = piece.endswith("\n")
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass
        self._file.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding, ...
        return getattr(self._stream, name)


class TerminalLog:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.start_wall = datetime.now()
        self.start_monotonic = time.monotonic()
        self.state = lambda: None  # deploy.py points this at the controller
        self._exit_recorded = False

        file = open(os.path.join(run_dir, "terminal.log"), "a", buffering=1)
        file.at_line_start = True
        lock = threading.Lock()
        sys.stdout = _Tee(sys.stdout, file, lock)
        sys.stderr = _Tee(sys.stderr, file, lock)

        previous_hook = sys.excepthook

        def excepthook(exc_type, exc, tb):
            if issubclass(exc_type, KeyboardInterrupt):
                self.record_exit("Ctrl+C")
            else:
                trace = "".join(traceback.format_exception(exc_type, exc, tb))
                self.record_exit(f"uncaught {exc_type.__name__}", trace)
            previous_hook(exc_type, exc, tb)

        sys.excepthook = excepthook
        atexit.register(self.record_exit, "exited")

    def record_exit(self, reason, detail=None):
        """Write exit.json; only the first call counts."""
        if self._exit_recorded:
            return
        self._exit_recorded = True
        try:
            state = self.state()
        except Exception:  # never let bookkeeping block an exit
            state = None
        end = datetime.now()
        info = {
            "reason": reason,
            "start_wall_clock": self.start_wall.isoformat(),
            "end_wall_clock": end.isoformat(),
            "uptime_s": round(time.monotonic() - self.start_monotonic, 3),
            "state": state,
            "detail": detail,
        }
        with open(os.path.join(self.run_dir, "exit.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"=== go2_deploy exiting at {end:%H:%M:%S}: {reason} (state {state}) ===")
        sys.stdout.flush()
