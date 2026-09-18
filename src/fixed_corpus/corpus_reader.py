"""Random access into an immutable JSONL corpus via its row-offset sidecar."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence


# Random corpus lookups are latency-bound rather than bandwidth-bound on network
# storage, so these defaults trade a little read amplification for far fewer
# round trips. Measured on the 21M-row wiki18 corpus over JuiceFS with 1000
# random ordinals per request: 324ms unsorted and serial, 271ms once sorted and
# coalesced, 41ms once the coalesced reads are also issued concurrently.
DEFAULT_READ_WORKERS = 32
DEFAULT_COALESCE_GAP_BYTES = 64 << 10


def _pread_exact(fd: int, length: int, offset: int) -> bytes:
    """Read ``length`` bytes at ``offset``; ``os.pread`` may return a short read."""
    chunk = os.pread(fd, length, offset)
    if len(chunk) >= length:
        return chunk
    chunks = [chunk]
    remaining = length - len(chunk)
    offset += len(chunk)
    while remaining > 0 and chunk:
        chunk = os.pread(fd, remaining, offset)
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class CorpusTextReader:
    """Fetch corpus rows by ordinal from an immutable JSONL file.

    A request sorts the wanted ordinals, coalesces rows that sit close together in
    the file into one read, and issues those reads concurrently. Reads go through
    ``os.pread`` on a single descriptor, which carries no seek position and is
    therefore safe to share across threads and to inherit across a fork.
    """

    def __init__(
        self,
        corpus_path: str | os.PathLike[str],
        offsets_path: str | os.PathLike[str],
        *,
        read_workers: int = DEFAULT_READ_WORKERS,
        coalesce_gap_bytes: int = DEFAULT_COALESCE_GAP_BYTES,
    ) -> None:
        import numpy as np

        self.corpus_path = str(corpus_path)
        self.offsets = np.load(offsets_path, mmap_mode="r")
        if self.offsets.ndim != 1 or not np.issubdtype(self.offsets.dtype, np.integer):
            raise ValueError("Corpus offsets must be a one-dimensional integer array")
        self.count = int(self.offsets.shape[0])
        self.size = os.path.getsize(self.corpus_path)
        self.read_workers = max(1, int(read_workers))
        self.coalesce_gap_bytes = max(0, int(coalesce_gap_bytes))
        self._fd = os.open(self.corpus_path, os.O_RDONLY)
        self._executor: ThreadPoolExecutor | None = None
        self._executor_pid = os.getpid()

    def _pool(self) -> ThreadPoolExecutor:
        # A pool inherited across a fork has no live threads, so rebuild it when the
        # owning pid changes. This keeps DataLoader and multiprocessing workers safe.
        if self._executor is None or self._executor_pid != os.getpid():
            self._executor = ThreadPoolExecutor(
                max_workers=self.read_workers, thread_name_prefix="corpus-read"
            )
            self._executor_pid = os.getpid()
        return self._executor

    def _row_end(self, ordinal: int) -> int:
        return int(self.offsets[ordinal + 1]) if ordinal + 1 < self.count else self.size

    def _spans(self, ordinals: list[int]) -> list[tuple[int, int, int, int]]:
        """Group sorted ordinals into (first, last, slice_start, slice_stop) reads."""
        spans: list[tuple[int, int, int, int]] = []
        first = previous = ordinals[0]
        slice_start = 0
        for position, ordinal in enumerate(ordinals[1:], start=1):
            if int(self.offsets[ordinal]) - self._row_end(previous) <= self.coalesce_gap_bytes:
                previous = ordinal
                continue
            spans.append((first, previous, slice_start, position))
            first = previous = ordinal
            slice_start = position
        spans.append((first, previous, slice_start, len(ordinals)))
        return spans

    def _read_span(self, span: tuple[int, int, int, int]) -> bytes:
        start = int(self.offsets[span[0]])
        return _pread_exact(self._fd, self._row_end(span[1]) - start, start)

    def read_records(self, ordinals: Sequence[int]) -> list[dict[str, Any]]:
        """Return one parsed corpus record per requested ordinal, in request order."""
        values = [int(ordinal) for ordinal in ordinals]
        if not values:
            return []
        unique = sorted(set(values))
        if unique[0] < 0 or unique[-1] >= self.count:
            out_of_range = unique[0] if unique[0] < 0 else unique[-1]
            raise IndexError(f"Corpus ordinal out of range: {out_of_range}")
        spans = self._spans(unique)
        if len(spans) == 1:
            buffers = [self._read_span(spans[0])]
        else:
            buffers = list(self._pool().map(self._read_span, spans))
        by_ordinal: dict[int, dict[str, Any]] = {}
        for span, buffer in zip(spans, buffers):
            base = int(self.offsets[span[0]])
            for ordinal in unique[span[2] : span[3]]:
                start = int(self.offsets[ordinal]) - base
                by_ordinal[ordinal] = json.loads(buffer[start : self._row_end(ordinal) - base])
        return [by_ordinal[value] for value in values]

    def read_contents(self, ordinals: Sequence[int]) -> list[str]:
        return [record["contents"] for record in self.read_records(ordinals)]

    def close(self) -> None:
        if self._executor is not None and self._executor_pid == os.getpid():
            self._executor.shutdown(wait=False)
        self._executor = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
