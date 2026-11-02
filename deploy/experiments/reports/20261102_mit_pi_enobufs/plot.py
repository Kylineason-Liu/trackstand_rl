"""Display host request/reply backlog; this is not kernel queue instrumentation."""

from collections import defaultdict, deque
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
stats = json.loads((HERE/"statistics.json").read_text())
end = stats["estimated_failure_monotonic_s"]
with (HERE/"last_second.csv").open() as stream:
    rows = list(csv.DictReader(stream))
changes = [1 if r["record_type"] in ("tx", "command") else -1 if r["record_type"] == "rx" else 0
           for r in rows]
initial = sum(stats["pending_logged_queries"].values())-sum(changes)
assert initial >= 0
t = np.array([(float(r["host_monotonic_s"])-end)*1000 for r in rows])
n = initial+np.cumsum(changes)
pending, delays = defaultdict(deque), defaultdict(list)
for r in rows:
    stamp, kind, op = float(r["host_monotonic_s"]), r["record_type"], r["opcode"]
    if kind in ("tx", "command"):
        pending["F1" if op == "MIT" else op].append(stamp)
    elif kind == "rx" and pending[op]:
        delays[op].append(((stamp-end)*1000, (stamp-pending[op].popleft())*1000))
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
axes[0].step(t, n, where="post", color="#333333", label="Requests without a logged reply")
axes[0].set(ylabel="Outstanding requests", title="Application backlog rises before SocketCAN ENOBUFS")
axes[0].set_ylim(-.3, max(n[t >= -100])+1)
for op, color in (("A1", "#dd7c20"), ("A2", "#2166ac"), ("F1", "#358b52")):
    values = np.array(delays[op])
    axes[1].plot(values[:, 0], values[:, 1], ".-", color=color, markersize=3, linewidth=.7, label=op)
axes[1].set(xlabel="Host time relative to estimated failure [ms]", ylabel="FIFO reply delay [ms]",
            xlim=(-100, 0), ylim=(-.2, 11))
for ax in axes:
    ax.axvline(0, color="red", linewidth=1.5, label="Estimated send failure")
    ax.grid(alpha=.25)
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("Rear MIT PI: P=0.30, I=3.3, 200-Hz motion; final 100 ms")
fig.text(.5, .015, "Logged requests are not measured kernel TX-queue occupancy. "
         "FIFO host pairing is not motor-loop latency. Failed sends are absent from the CSV.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, .04, 1, .95))
for suffix in ("png", "svg", "pdf"):
    fig.savefig(HERE/f"queue_backlog.{suffix}", dpi=180)
plt.close(fig)
