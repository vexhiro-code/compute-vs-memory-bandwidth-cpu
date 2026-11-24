"""Summarize raw benchmark CSVs using medians and interquartile ranges."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "analysis" / "summary.csv"


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main():
    groups = []
    for path in sorted(RAW.glob("*.csv")):
        if path.name == ".gitkeep":
            continue
        rows = read_rows(path)
        if not rows:
            continue
        first = rows[0]
        kernel = first.get("kernel", path.stem.split("_n")[0])
        size = first.get("elements") or first.get("size") or first.get("dimension") or first.get("blocks") or ""
        threads = first.get("threads") or first.get("threads_requested") or ""
        metrics = [
            col for col in ("nominal_gb_per_s", "nominal_gflop_per_s", "estimated_gflop_per_s")
            if col in first
        ]
        rec = {
            "file": path.name,
            "kernel": kernel,
            "size": size,
            "threads": threads,
            "repetitions": len(rows),
        }
        for col in metrics:
            values = [float(row[col]) for row in rows]
            rec[col + "_median"] = statistics.median(values)
            rec[col + "_q1"] = quantile(values, 0.25)
            rec[col + "_q3"] = quantile(values, 0.75)
        groups.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in groups for k in row))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups)
    print(f"Summarized {len(groups)} measurement files into {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
