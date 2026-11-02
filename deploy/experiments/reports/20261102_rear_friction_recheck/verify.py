"""Independent raw-reply means, normal-equation estimates and input-preservation checks."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    stats = json.loads((HERE / "statistics.json").read_text())
    stem = ROOT / stats["source_stem"]
    meta = json.loads(stem.with_suffix(".json").read_text())
    kt = meta["torque_constant_nm_per_a"]
    starts = {p["segment"]["name"]: p["start_monotonic_s"] for p in meta["phases"]
              if p["segment"]["kind"] == "steady"}
    raw = {name: {op: [] for op in ("A1", "A2")} for name in starts}
    with stem.with_suffix(".csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            name, op = row["method"], row["opcode"]
            if row["record_type"] != "rx" or name not in raw or op not in raw[name]:
                continue
            time = float(row["host_monotonic_s"]) - starts[name]
            if not 2 <= time < 5:
                continue
            b = bytes.fromhex(row["payload_hex"])
            assert b[0] == int(op, 16)
            number = int.from_bytes(b[1:], byteorder="little", signed=True)
            raw[name][op].append(-number * (.001 * kt if op == "A1" else np.pi / 3000))
    rows = list(csv.DictReader((HERE / "steady_holds.csv").open()))
    worst = {"torque_nm": 0., "velocity_rad_s": 0.}
    for r in rows:
        n = r["name"]
        assert len(raw[n]["A1"]) == int(r["torque_samples"])
        assert len(raw[n]["A2"]) == int(r["velocity_samples"])
        worst["torque_nm"] = max(worst["torque_nm"], abs(np.mean(raw[n]["A1"]) - float(r["torque_mean_nm"])))
        worst["velocity_rad_s"] = max(worst["velocity_rad_s"],
                                     abs(np.mean(raw[n]["A2"]) - float(r["velocity_mean_rad_s"])))
    assert max(worst.values()) < 1e-12
    high = [r for r in rows if abs(float(r["target_rad_s"])) >= 1]
    v = np.array([np.mean(raw[r["name"]]["A2"]) for r in high])
    t = np.array([np.mean(raw[r["name"]]["A1"]) for r in high])
    x = np.column_stack((v, np.sign(v), np.ones(len(v))))
    beta = np.linalg.solve(x.T @ x, x.T @ t)
    expected = stats["models"]["moving_ols"]
    reference = np.array([expected[k] for k in ("B_nm_s_per_rad", "Tc_nm", "bias_nm")])
    assert np.max(abs(beta - reference)) < 1e-12
    holds = list(csv.DictReader((HERE / "steady_holds.csv").open()))
    levels = sorted({abs(float(r["target_rad_s"])) for r in holds})
    combined = []
    for speed in levels:
        group = [r for r in holds if abs(float(r["target_rad_s"])) == speed]
        torque = [np.sign(float(r["target_rad_s"])) * float(r["torque_mean_nm"]) for r in group]
        combined.append(dict(target_magnitude_rad_s=speed, n_holds=len(group),
                             mean_aligned_torque_nm=float(np.mean(torque)),
                             between_hold_sd_nm=float(np.std(torque, ddof=1))))
    with (HERE / "torque_by_speed_both_directions.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(combined[0]))
        w.writeheader()
        w.writerows(combined)
    preservation = json.loads((HERE / "preservation_before.json").read_text())
    changed = [p for p, digest in preservation.items() if sha(ROOT / p) != digest]
    assert not changed, changed
    result = dict(passed=True, independent_raw_means_maximum_error=worst,
                  independent_normal_equation_coefficients=beta.tolist(),
                  checked_fresh_a1=sum(len(v["A1"]) for v in raw.values()),
                  checked_fresh_a2=sum(len(v["A2"]) for v in raw.values()),
                  protected_existing_files_unchanged=len(preservation),
                  source_sha256=sha(stem.with_suffix(".csv")))
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
