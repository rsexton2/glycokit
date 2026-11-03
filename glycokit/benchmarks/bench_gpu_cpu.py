#!/usr/bin/env python3
"""
CPU vs GPU timing of the GlycoAnalysis blocking test.

Runs the analysis once on the CPU (results + surface residues), then for each
analyzed frame times

  CPU: GlycoAnalysis._visibility (KD-tree cutoff pairs + NumPy blocking test)
  GPU: the same cutoff pairs + glycokit._gpu.blocked_pairs_kernel
       + CuPy reduction (positions uploaded every frame)

and checks that both give identical keys, blocked-pair counts and mean
blocked distances. Modes: default/vdw, default/cylinder, v2/cylinder.

Example:
  python benchmarks/bench_gpu_cpu.py -a protein.pdb -b traj.xtc \
      --gs "resname NAG BMA MAN" --max-frames 20 --repeats 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
import time

import MDAnalysis as mda
import numpy as np

from glycokit.config import _KD_PAD
from glycokit.glyco import GlycoAnalysis

try:
    import cupy as cp

    from glycokit import _gpu

except ImportError as exc:  # CPU-only environment
    cp = _gpu = None
    GPU_IMPORT_ERROR = exc
else:
    GPU_IMPORT_ERROR = None

# ---------------------------------------------------------------- timing


def _sync():
    """Wait for all queued GPU work (kernel launches are asynchronous)."""
    if cp is not None:
        cp.cuda.Device().synchronize()


@contextlib.contextmanager
def _stage(stages, name, gpu):
    """Add the wall time of the block to ``stages[name]``."""
    if gpu:
        _sync()
    t0 = time.perf_counter()
    yield
    if gpu:
        _sync()
    stages[name] = stages.get(name, 0.0) + time.perf_counter() - t0


# ---------------------------------------------------------------- CPU path


def cpu_frame(g, start_idx):
    """
    Current CPU implementation for one frame.

    ``prep`` (context + cutoff pairs) is timed separately so the blocking
    share can be estimated as ``total - prep``.
    """
    stages = {}
    with _stage(stages, "total", False):
        cand, vis, n_blocked, mean_d, _, _ = g._visibility(
            start_idx, g._mode, g._blocking)
    with _stage(stages, "prep", False):
        ctx = g._blocking_context(g._mode, g._blocking)
        g._cutoff_pairs(start_idx, ctx)
    return (cand, vis, int(n_blocked), float(mean_d)), stages


# ---------------------------------------------------------------- GPU path


class GpuBlocker:
    """GPU blocking test for a finished GlycoAnalysis run."""

    def __init__(self, g, cell_size=None, threads=128):
        self.g = g
        self.cyl = g._blocking == "cylinder"
        self.threads = int(threads)
        self.cell_size = cell_size
        self.n_gly = len(g.gly_atoms)

        # frame-independent arrays, uploaded once (as _prepare would)
        self.p_ix = cp.asarray(g.start_atoms.ix, dtype=cp.int64)
        self.p_res = cp.asarray(g._start_resindex, dtype=cp.int64)
        self.p_row = cp.asarray(g._start_res_row, dtype=cp.int64)
        self.b_ix = cp.asarray(g.blocker_atoms.ix, dtype=cp.int64)
        self.b_res = cp.asarray(g._blocker_resindex, dtype=cp.int64)
        if self.cyl:
            self.R2 = cp.zeros(
                1, dtype=cp.float64)  # unused by the cylinder test
        else:
            # same arithmetic as GlycoAnalysis._blocking_context
            R = float(g.vdw_scale) * np.asarray(
                g._blocker_radii, dtype=np.float64) + float(g.vdw_probe)
            self.R2 = cp.asarray(R * R)

    def frame(self, start_idx):
        """Return ``(result, stages, n_pairs)`` for the current frame."""
        g = self.g
        st = {}
        empty = np.empty(0, dtype=np.int64)

        with _stage(st, "prep", False):
            ctx = g._blocking_context(g._mode, g._blocking)
            i, gl, dist, _, _ = g._cutoff_pairs(start_idx, ctx)
        n_pairs = len(i)
        if n_pairs == 0:
            st["total"] = sum(st.values())
            return (empty, empty, 0, math.nan), st, 0

        # exact-test dtype: pair dtype for the cylinder (float32 in v2),
        # float64 for the sphere test, as in _blocking_hits
        dtype = ctx["dtype"] if self.cyl else np.float64
        with _stage(st, "upload", True):
            P = cp.asarray(ctx["P_all"].astype(dtype, copy=False))
            G = cp.asarray(ctx["G"] if self.cyl else ctx["G64"])
            pi = cp.asarray(np.asarray(start_idx, dtype=np.int64)[i])  # global
            gi = cp.asarray(gl.astype(np.int64, copy=False))
            d = cp.asarray(dist.astype(np.float64))

        out = cp.zeros(n_pairs, dtype=cp.int8)
        if ctx["btree"] is not None:  # at least one blocker
            reach = float(ctx["search_radius"] + _KD_PAD)
            cell = self.cell_size or max(reach, 1.0)
            with _stage(st, "cells", True):
                Bd, origin, inv, dims, order, cs, ce = _gpu.build_cells(
                    ctx["B"], cell)
            radius = (
                ctx["dtype"](ctx["radius"]) if self.cyl else np.float64(0.0))
            blocks = (n_pairs + self.threads - 1) // self.threads
            with _stage(st, "kernel", True):
                _gpu.blocked_pairs_kernel[blocks, self.threads](
                    P, G, pi, gi, self.p_ix, self.p_res, Bd,
                    self.R2, self.b_ix, self.b_res, order, cs, ce, origin,
                    float(inv), dims, reach, radius, self.cyl,
                    bool(ctx["exclude_own_residue"]),
                    bool(ctx["exclude_start_overlap"]), out,
                )

        with _stage(st, "reduce", True):
            blocked = out.astype(cp.bool_)
            keys = self.p_row[pi] * self.n_gly + gi
            cand = cp.unique(keys).get()
            vis = cp.unique(keys[~blocked]).get()
            n_blocked = int(blocked.sum())
            mean_d = float(d[blocked].mean()) if n_blocked else math.nan

        st["total"] = sum(st.values())
        return (cand, vis, n_blocked, mean_d), st, n_pairs


# ---------------------------------------------------------------- checks


def compare(g, k, cpu_res, gpu_res):
    """Parity of the CPU and GPU results for analyzed frame ``k``."""
    c_cand, c_vis, c_nb, c_md = cpu_res
    g_cand, g_vis, g_nb, g_md = gpu_res
    both_nan = math.isnan(c_md) and math.isnan(g_md)
    md_diff = 0.0 if both_nan else abs(c_md - g_md)
    counts_ok = np.array_equal(
        g._counts_from_visible_keys(g_vis), g.results.glycan_counts[k])
    report = {
        "cand_equal": bool(np.array_equal(c_cand, g_cand)),
        "vis_equal": bool(np.array_equal(c_vis, g_vis)),
        "n_blocked_equal": c_nb == g_nb,
        "counts_equal_run": bool(counts_ok),
        # summation order differs (sorted blockers vs pair order), so the
        # mean can differ in the last bits; keys and counts must be exact
        "mean_dist_abs_diff": md_diff,
    }
    report["ok"] = (
        report["cand_equal"] and report["vis_equal"]
        and report["n_blocked_equal"] and report["counts_equal_run"]
        and md_diff <= 1e-9 * max(1.0,
                                  abs(c_md) if not both_nan else 1.0))
    return report


def _median(runs, key):
    values = [r[key] for r in runs if key in r]
    return statistics.median(values) if values else 0.0


# ---------------------------------------------------------------- main


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-a", "--top", required=True, help="topology / structure")
    p.add_argument(
        "-b", "--traj", nargs="*", default=[], help="trajectory file(s)")
    p.add_argument("--gs", default="", help="glycan selection (empty: guesser)")
    p.add_argument("--mode", default="default", choices=("default", "v2"))
    p.add_argument("--blocking", default=None, choices=("vdw", "cylinder"))
    p.add_argument("--start", type=int)
    p.add_argument("--stop", type=int)
    p.add_argument("--step", type=int)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--n-workers", type=int, default=-1,
        help="CPU KD-tree threads (-1 = all)")
    p.add_argument(
        "--skip-sasa", action="store_true",
        help="all start residues are surface residues")
    p.add_argument(
        "--cell-size", type=float, default=None,
        help="GPU cell edge in A (default: search radius)")
    p.add_argument(
        "--threads", type=int, default=128, help="CUDA threads per block")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--json", help="write per-frame results to this file")
    p.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default="auto",
        help="auto: time the GPU path if available; gpu: require it; "
        "cpu: CPU timings only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    use_gpu = args.device != "cpu"
    if use_gpu and (GPU_IMPORT_ERROR is not None or not _gpu.is_available()):
        reason = GPU_IMPORT_ERROR or "no CUDA device"
        if args.device == "gpu":
            sys.exit(f"--device gpu: {reason}")
        print(f"GPU unavailable ({reason}); running CPU only", file=sys.stderr)
        use_gpu = False

    traj = args.traj if len(args.traj) != 1 else args.traj[0]
    u = mda.Universe(args.top, traj) if args.traj else mda.Universe(args.top)
    frames = list(range(u.trajectory.n_frames))[slice(
        args.start, args.stop, args.step)][:args.max_frames]

    g = GlycoAnalysis(
        u, gly_sel=args.gs, mode=args.mode, blocking=args.blocking,
        skip_sasa=args.skip_sasa, n_workers=args.n_workers,
    )
    t0 = time.perf_counter()
    g.run(frames=frames, write=False)
    t_run = time.perf_counter() - t0
    print(
        f"mode={g._mode} blocking={g._blocking} frames={g.n_frames} "
        f"start_atoms={len(g.start_atoms)} glycan_atoms={len(g.gly_atoms)} "
        f"blocker_atoms={len(g.blocker_atoms)}")
    print(f"full CPU run (incl. FreeSASA, all frames): {t_run:.2f} s\n")

    gpu = GpuBlocker(g, args.cell_size, args.threads) if use_gpu else None
    compile_s = None
    records, all_ok = [], True

    header = (
        f"{'frame':>5} {'starts':>7} {'pairs':>9} "
        f"{'CPU ms':>9} {'GPU ms':>9} {'speedup':>8}  parity")
    print(header)
    print("-" * len(header))

    for k in range(g.n_frames):
        with g._at_analyzed_frame(k):
            start_idx = g.surface_start_indices(k)

            # warm-up; the very first GPU call includes context + JIT
            for w in range(max(args.warmup, 1 if use_gpu else 0)):
                cpu_frame(g, start_idx)
                if use_gpu:
                    _, st0, _ = gpu.frame(start_idx)
                    if compile_s is None:
                        compile_s = st0.get("kernel", 0.0)

            cpu_runs, gpu_runs = [], []
            cpu_res = gpu_res = None
            n_pairs = 0
            for _ in range(args.repeats):
                cpu_res, cst = cpu_frame(g, start_idx)
                cpu_runs.append(cst)
                if use_gpu:
                    gpu_res, gst, n_pairs = gpu.frame(start_idx)
                    gpu_runs.append(gst)

        rec = {
            "analyzed_frame": k,
            "trajectory_frame": int(g._analyzed_frames()[k]),
            "surface_start_atoms": int(len(start_idx)),
            "n_pairs": int(n_pairs),
            "cpu_s": {
                key: _median(cpu_runs, key)
                for key in ("total", "prep")
            },
        }
        cpu_ms = rec["cpu_s"]["total"] * 1e3
        if use_gpu:
            rec["gpu_s"] = {
                key: _median(gpu_runs, key)
                for key in (
                    "prep", "upload", "cells", "kernel", "reduce", "total")
            }
            rec["parity"] = compare(g, k, cpu_res, gpu_res)
            all_ok &= rec["parity"]["ok"]
            gpu_ms = rec["gpu_s"]["total"] * 1e3
            speed = cpu_ms / gpu_ms if gpu_ms else float("inf")
            flag = "ok" if rec["parity"]["ok"] else "MISMATCH"
            print(
                f"{k:5d} {len(start_idx):7d} {n_pairs:9d} "
                f"{cpu_ms:9.2f} {gpu_ms:9.2f} {speed:7.1f}x  {flag}")
        else:
            print(f"{k:5d} {len(start_idx):7d} {'-':>9} {cpu_ms:9.2f}")
        records.append(rec)

    # ---------------- summary
    print("\nMedian per frame (ms):")
    cpu_tot = statistics.median(r["cpu_s"]["total"] for r in records) * 1e3
    cpu_prep = statistics.median(r["cpu_s"]["prep"] for r in records) * 1e3
    print(
        f"  CPU  total {cpu_tot:9.2f}   prep (ctx+cutoff) {cpu_prep:8.2f}   "
        f"blocking ~{cpu_tot - cpu_prep:8.2f}")
    if use_gpu:
        parts = {
            key: statistics.median(r["gpu_s"][key] for r in records) * 1e3
            for key in ("prep", "upload", "cells", "kernel", "reduce", "total")
        }
        print(
            "  GPU  total {total:9.2f}   prep {prep:8.2f}   upload {upload:7.2f}"
            "   cells {cells:7.2f}   kernel {kernel:8.2f}   reduce {reduce:7.2f}"
            .format(**parts))
        print(
            f"  speedup (end to end, excl. FreeSASA): {cpu_tot / parts['total']:.1f}x"
        )
        print(
            f"  speedup (blocking only): "
            f"{(cpu_tot - cpu_prep) / max(parts['total'] - parts['prep'], 1e-12):.1f}x"
        )
        print(f"  first-call kernel time (JIT + context): {compile_s:.2f} s")
        print(
            f"\nparity: {'all frames identical' if all_ok else 'MISMATCHES FOUND'}"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "run_s": t_run,
                    "compile_s": compile_s,
                    "frames": records
                }, fh, indent=2, default=float,
            )
        print(f"wrote {args.json}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
