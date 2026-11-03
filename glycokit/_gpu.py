"""
Optional CUDA kernels for the default-mode blocking test.

Import only when the CUDA backend is requested::

    from glycokit import _gpu   # raises ImportError without cupy/numba-cuda
"""

import math

import numpy as np

from glycokit.config import _KD_PAD

try:
    import cupy as cp
    from numba import cuda
except ImportError as exc:  # CPU-only install
    raise ImportError(
        "glycokit._gpu needs cupy and numba-cuda "
        "(conda install -c conda-forge cupy numba-cuda)") from exc


def is_available():
    """True if a usable CUDA device is present."""
    return cuda.is_available()


@cuda.jit(device=True, inline=True)
def _sphere_hit(px, py, pz, gx, gy, gz, bx, by, bz, r2, excl_overlap):
    # mirrors utils._sphere_blocked (float64)
    dx, dy, dz = gx - px, gy - py, gz - pz  # start -> glycan
    cx, cy, cz = bx - px, by - py, bz - pz  # start -> blocker
    seg2 = dx * dx + dy * dy + dz * dz
    if seg2 < 1e-12:
        seg2 = 1e-12
    t = (cx * dx + cy * dy + cz * dz) / seg2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex, ey, ez = cx - t * dx, cy - t * dy, cz - t * dz
    if ex * ex + ey * ey + ez * ez > r2:  # hit is "<= R2"
        return False
    if excl_overlap and cx * cx + cy * cy + cz * cz <= r2:  # hit needs "> R2"
        return False
    return True


@cuda.jit(device=True, inline=True)
def _cylinder_hit(px, py, pz, gx, gy, gz, bx, by, bz, r):
    # mirrors utils._cylinder_blocked. In v2: px..gz and r are float32,
    # bx..bz float64, so vec/const stay float32 and q terms promote to float64.
    vx, vy, vz = px - gx, py - gy, pz - gz  # pt2 - pt1 (glycan -> start)
    const = r * math.sqrt(vx * vx + vy * vy + vz * vz)
    ax, ay, az = bx - gx, by - gy, bz - gz  # q_pt1 = B - G
    if ax * vx + ay * vy + az * vz <= 0.0:  # strict end cap
        return False
    qx, qy, qz = bx - px, by - py, bz - pz  # q_pt2 = B - P
    if qx * vx + qy * vy + qz * vz >= 0.0:  # strict end cap
        return False
    kx = ay * vz - az * vy  # q_pt1 x vec
    ky = az * vx - ax * vz
    kz = ax * vy - ay * vx
    return math.sqrt(kx * kx + ky * ky + kz * kz) <= const  # inclusive radius


@cuda.jit(device=True, inline=True)
def _axis_range(a, b, o, inv, n, reach):
    lo = int(math.floor((min(a, b) - reach - o) * inv))
    hi = int(math.floor((max(a, b) + reach - o) * inv))
    return max(lo, 0), min(hi, n - 1)


@cuda.jit
def blocked_pairs_kernel(
    P, G, pi, gi, p_ix, p_res, B, R2, b_ix, b_res, order, cell_start, cell_end,
    origin, inv_cs, dims, reach, radius, use_cylinder, excl_own, excl_overlap,
    out,
):
    """One thread per cutoff pair; ``out[k] = 1`` on the first blocking hit."""
    k = cuda.grid(1)
    if k >= pi.size:
        return
    i, g = pi[k], gi[k]
    px, py, pz = P[i, 0], P[i, 1], P[i, 2]
    gx, gy, gz = G[g, 0], G[g, 1], G[g, 2]
    own_ix, own_res = p_ix[i], p_res[i]

    x0, x1 = _axis_range(px, gx, origin[0], inv_cs, dims[0], reach)
    y0, y1 = _axis_range(py, gy, origin[1], inv_cs, dims[1], reach)
    z0, z1 = _axis_range(pz, gz, origin[2], inv_cs, dims[2], reach)

    for cx in range(x0, x1 + 1):
        for cy in range(y0, y1 + 1):
            for cz in range(z0, z1 + 1):
                c = (cx * dims[1] + cy) * dims[2] + cz
                for s in range(cell_start[c], cell_end[c]):
                    j = order[s]
                    if b_ix[j] == own_ix:
                        continue  # start atom never blocks
                    if excl_own and b_res[j] == own_res:
                        continue
                    if use_cylinder:
                        hit = _cylinder_hit(
                            px, py, pz, gx, gy, gz, B[j, 0], B[j, 1], B[j, 2],
                            radius)
                    else:
                        hit = _sphere_hit(
                            px, py, pz, gx, gy, gz, B[j, 0], B[j, 1], B[j, 2],
                            R2[j], excl_overlap,
                        )
                    if hit:
                        out[k] = 1
                        return


def build_cells(B, cell_size):
    """
    Device cell list of blockers (sort by cell id, then searchsorted).

    :param B: Blocker coordinates, shape (k, 3).
    :param float cell_size: Cell edge in Å (about the search radius).
    :return: ``(B_device, origin, inv_cell, dims, order, cell_start, cell_end)``
    """
    Bd = cp.asarray(B, dtype=cp.float64)
    origin = Bd.min(axis=0)
    inv = 1.0 / cell_size
    ijk = cp.floor((Bd - origin) * inv).astype(cp.int64)
    dims = ijk.max(axis=0) + 1
    cid = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = cp.argsort(cid)
    cid_sorted = cid[order]
    cells = cp.arange(int(dims.prod()), dtype=cp.int64)
    start = cp.searchsorted(cid_sorted, cells, side="left")
    end = cp.searchsorted(cid_sorted, cells, side="right")
    return Bd, origin, inv, dims, order, start, end


def is_available():
    """True if both CuPy and Numba-CUDA can use a device."""
    try:
        return bool(cp.is_available()) and bool(cuda.is_available())
    except Exception:  # pylint: disable=broad-except  # driver/runtime errors
        return False


class GpuBlocking:
    """Default-mode (float64) blocking test on the GPU for a prepared GlycoAnalysis."""

    def __init__(self, g, threads=128):
        self.g = g
        self.threads = int(threads)
        self.n_gly = len(g.gly_atoms)
        # frame-independent arrays, uploaded once
        self.p_ix = cp.asarray(g.start_atoms.ix, dtype=cp.int64)
        self.p_res = cp.asarray(g._start_resindex, dtype=cp.int64)
        self.p_row = cp.asarray(g._start_res_row, dtype=cp.int64)
        self.b_ix = cp.asarray(g.blocker_atoms.ix, dtype=cp.int64)
        self.b_res = cp.asarray(g._blocker_resindex, dtype=cp.int64)
        if g._blocking == "vdw":
            R = (
                float(g.vdw_scale) * np.asarray(g._blocker_radii, np.float64) +
                float(g.vdw_probe))  # same arithmetic as _blocking_context
            self.R2 = cp.asarray(R * R)
        else:
            self.R2 = cp.zeros(1, dtype=cp.float64)  # unused by the cylinder

    def visibility(self, start_idx, mode, blocking):
        """``(cand_keys, vis_keys, n_blocked, mean_blocked_distance)``, like _visibility."""
        g = self.g
        empty = np.empty(0, dtype=np.int64)
        start_idx = np.asarray(start_idx, dtype=np.intp)
        if len(start_idx) == 0 or self.n_gly == 0:
            return empty, empty, 0, np.nan

        # cutoff pairs stay on the CPU (KD-tree), so pair selection is identical
        ctx = g._blocking_context(mode, blocking)
        i, gl, dist, _, _ = g._cutoff_pairs(start_idx, ctx)
        if len(i) == 0:
            return empty, empty, 0, np.nan

        P = cp.asarray(ctx["P_all"])  # float64 in default mode
        G = cp.asarray(ctx["G64"])
        pi = cp.asarray(start_idx[i].astype(np.int64))
        gi = cp.asarray(gl.astype(np.int64))
        out = cp.zeros(len(i), dtype=cp.int8)

        if ctx["btree"] is not None:
            reach = float(ctx["search_radius"] + _KD_PAD)
            Bd, origin, inv, dims, order, cs, ce = build_cells(
                ctx["B"], max(reach, 1.0))
            cyl = ctx["cylinder"]
            radius = np.float64(ctx["radius"]) if cyl else np.float64(0.0)
            blocks = (len(i) + self.threads - 1) // self.threads
            blocked_pairs_kernel[blocks, self.threads](
                P, G, pi, gi, self.p_ix, self.p_res,
                Bd, self.R2, self.b_ix, self.b_res, order, cs, ce, origin,
                float(inv), dims, reach, radius, cyl,
                bool(ctx["exclude_own_residue"]),
                bool(ctx["exclude_start_overlap"]), out,
            )

        blocked = out.astype(cp.bool_)
        keys = self.p_row[pi] * self.n_gly + gi
        cand = cp.unique(keys).get()
        vis = cp.unique(keys[~blocked]).get()
        blocked_h = blocked.get()
        n_blocked = int(blocked_h.sum())
        mean_d = float(dist[blocked_h].mean()) if n_blocked else np.nan
        return cand, vis, n_blocked, mean_d
