"""Logging and clock helpers for offline/supported CAN latency diagnostics."""
import csv
import json
import queue
import threading
import time
from pathlib import Path

FIELDS = ('kind', 'mono_s', 'seq', 'motor', 'opcode', 'can_id', 'dlc', 'payload',
          'scheduled_s', 'send_start_s', 'send_end_s', 'kernel_wall_s', 'kernel_mono_s',
          'dequeue_s', 'processed_s', 'clock_uncertainty_ms', 'clock_offset_jump_ms',
          'clock_valid', 'local_tx', 'rx_dropped', 'rtt_ms', 'queue_ms', 'pair_valid',
          'publish_s', 'callback_s', 'tick_s', 'source_age_ms', 'detail')


class CsvLog:
    """Bounded asynchronous disk writer; overflow/errors invalidate rather than hide data."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('x', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.queue = queue.Queue(maxsize=100000)
        self.error = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def emit(self, kind, **fields):
        if self.error:
            raise RuntimeError(f'CSV writer failed: {self.error}')
        self.queue.put_nowait(dict(kind=kind, mono_s=time.monotonic(), **fields))

    def _work(self):
        try:
            while True:
                row = self.queue.get()
                if row is None:
                    break
                self.writer.writerow(row)
                if self.queue.empty():
                    self.file.flush()
        except BaseException as error:
            self.error = error

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        self.file.close()
        if self.error:
            raise RuntimeError(f'CSV writer failed: {self.error}')


class ClockMap:
    """Keep uncertain/stepped clocks visible; never fabricate a fresh CAN timestamp."""
    def __init__(self):
        self.offset = None
        self.width = 0.

    def sample(self):
        before = time.monotonic()
        wall = time.time()
        after = time.monotonic()
        return self.observe(before, wall, after)

    def observe(self, before, wall, after):
        width = after - before
        offset = wall - .5 * (before + after)
        jump = 0. if self.offset is None else offset - self.offset
        valid = width <= .001 and (self.offset is None or abs(jump) <= .001 + width + self.width)
        # Rebaseline after a clock step, invalidating this sample; timestamps
        # spanning a step are not used for request/reply latency.
        self.offset, self.width = offset, width
        return offset, width * 1000., jump * 1000., valid


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
