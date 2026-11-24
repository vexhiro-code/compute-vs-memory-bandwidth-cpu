import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def main():
    parser = argparse.ArgumentParser(description="NumPy/OpenBLAS reference benchmarks")
    parser.add_argument("kernel", choices=["vector_add", "dot_product", "matrix_multiply"])
    parser.add_argument("size", type=int, help="element count for vectors; matrix dimension for matrix_multiply")
    parser.add_argument("threads", type=int)
    parser.add_argument("repetitions", type=int, nargs="?", default=20)
    args = parser.parse_args()
    if args.size <= 0 or args.threads <= 0 or args.repetitions <= 0:
        parser.error("size, threads, and repetitions must be positive")

    n = args.size
    rng = np.random.default_rng(20261124)
    if args.kernel == "vector_add":
        a = rng.random(n, dtype=np.float64)
        b = rng.random(n, dtype=np.float64)
        c = np.empty_like(a)
        operation = lambda: np.add(a, b, out=c)
        expected = np.sum(a + b, dtype=np.float64)
        check = lambda: np.isclose(np.sum(c), expected, rtol=1e-10, atol=1e-8)
        nominal_bytes, flops = 24*n, n
        label = "size"
    elif args.kernel == "dot_product":
        a = rng.random(n, dtype=np.float64)
        b = rng.random(n, dtype=np.float64)
        operation = lambda: np.dot(a, b)
        expected = float(np.dot(a, b))
        check = lambda value: np.isclose(value, expected, rtol=1e-10, atol=1e-8)
        nominal_bytes, flops = 16*n, 2*n
        label = "size"
    else:
        a = rng.random((n, n), dtype=np.float64)
        b = rng.random((n, n), dtype=np.float64)
        c = np.empty((n, n), dtype=np.float64)
        operation = lambda: np.matmul(a, b, out=c)
        expected_checksum = float(np.sum(a @ b, dtype=np.float64))
        check = lambda: np.isclose(np.sum(c), expected_checksum, rtol=1e-10, atol=1e-7)
        nominal_bytes, flops = 24*n*n, 2*n*n*n
        label = "dimension"

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"numpy_{args.kernel}_{label}{n}_t{args.threads}.csv"
    with threadpool_limits(limits=args.threads, user_api="blas"):
        operation()
        info = threadpool_info()
        print("NumPy:", np.__version__)
        print("Thread pools:", info)
        rows = []
        for rep in range(1, args.repetitions + 1):
            start = time.perf_counter()
            result = operation()
            elapsed = time.perf_counter() - start
            if args.kernel == "vector_add":
                correct = bool(check())
                checksum = float(np.sum(c, dtype=np.float64))
            elif args.kernel == "dot_product":
                correct = bool(check(result))
                checksum = float(result)
            else:
                correct = bool(check())
                checksum = float(np.sum(c, dtype=np.float64))
            if not correct:
                raise RuntimeError(f"Correctness check failed at repetition {rep}")
            rows.append({
                "kernel": args.kernel, "size": n, "threads_requested": args.threads,
                "repetition": rep, "seconds": elapsed,
                "nominal_bytes": nominal_bytes, "nominal_flops": flops,
                "nominal_gb_per_s": nominal_bytes / elapsed / 1e9,
                "nominal_gflop_per_s": flops / elapsed / 1e9,
                "checksum": checksum,
            })
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("Correctness: PASS")
    print(f"Wrote {len(rows)} repetitions to {out}")
    print("Nominal traffic is an algorithmic estimate, not measured DRAM traffic.")


if __name__ == "__main__":
    main()
