import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


class Benchmarker:
    """Collects named timing measurements (in milliseconds) and peak memory.

    `time(...)` measures a code block. By default it uses CUDA events, which give real GPU time:
    recording an event is asynchronous, so timing many blocks does not stall the GPU — the single
    `torch.cuda.synchronize()` needed to read the elapsed times is deferred until the results are
    read (`dump`/`summarize`/`execution_times`). It falls back to wall-clock when CUDA is
    unavailable. `record(...)` stores a pre-measured millisecond value (e.g. the optimizer's own
    per-iteration CUDA-event timings). All stored values are milliseconds.
    """

    def __init__(self):
        self._execution_times = defaultdict(list)
        # CUDA-event measurements awaiting a synchronize(): (tag, start_event, end_event, num_calls).
        self._pending = []

    @property
    def execution_times(self):
        """The recorded times per tag (ms). Resolves any pending CUDA-event measurements first."""
        self._flush()
        return self._execution_times

    def _flush(self) -> None:
        """Read all pending CUDA-event timings with a single synchronize()."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for tag, start_event, end_event, num_calls in self._pending:
            ms = start_event.elapsed_time(end_event)  # CUDA events measure milliseconds
            for _ in range(num_calls):
                self._execution_times[tag].append(ms / num_calls)
        self._pending = []

    @contextmanager
    def time(self, tag: str, num_calls: int = 1, disable: bool = False, cuda: bool = True):
        if disable:
            yield
            return
        if cuda and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                # record() is async; the elapsed time is read later in _flush() after one sync.
                end_event.record()
                self._pending.append((tag, start_event, end_event, num_calls))
        else:
            start = perf_counter()
            try:
                yield
            finally:
                ms = (perf_counter() - start) * 1000.0
                for _ in range(num_calls):
                    self._execution_times[tag].append(ms / num_calls)

    def record(self, tag: str, value: float | int) -> None:
        """Record a pre-measured value under the given tag: time, memory, per-iteration count (e.g. number of gaussians)."""
        self._execution_times[tag].append(value)

    def merge(self, other: "Benchmarker") -> None:
        """Merge another benchmarker's recorded times into this one."""
        for tag, times in other.execution_times.items():  # property flushes `other`'s pending events
            self._execution_times[tag].extend(times)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg {np.mean(times):.1f} ms/call, total {sum(times)/1000:.1f} s")

    def clear_history(self) -> None:
        self._execution_times = defaultdict(list)
        self._pending = []
