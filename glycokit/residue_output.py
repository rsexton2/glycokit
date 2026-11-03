"""
Per-residue output writers shared by GlycoAnalysis and FreesasaAnalysis.

:class:`ResidueOutputMixin` adds frame bookkeeping, a B-factor PDB writer and
a CSV writer to an AnalysisBase subclass. The host class must provide:

* ``u`` (Universe), ``outdir`` (str or None)
* ``n_frames`` / ``frames`` (set by ``AnalysisBase.run``)
* :meth:`_residue_column_atoms`: one atom per value column
* :meth:`_residue_values`: values, shape (n_frames, n_columns)
"""

import contextlib
import csv
import operator
import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from glycokit.console import written
from glycokit.utils import _fmt_float, _resid_labels, pdb_atom_ids


class ResidueOutputMixin:
    """Shared writers; mix in before AnalysisBase. Defines no state or __init__."""

    u: mda.Universe
    outdir: str | None
    n_frames: int

    # -- hooks the host implements ---------------------------------------

    def _residue_column_atoms(self):
        """AtomGroup with one atom per value column (e.g. one CA per residue)."""
        raise NotImplementedError

    def _residue_values(self):
        """Per-residue values, shape (n_frames, n_columns)."""
        raise NotImplementedError

    # -- overridable defaults --------------------------------------------

    def _error(self, message, exc=ValueError):
        """Library default: raise. GlycoAnalysis overrides it (parser.error)."""
        raise exc(message)

    def _require_run(self):
        """:raises RuntimeError: Before run()."""
        try:
            values = self._residue_values()
        except AttributeError:
            values = None
        if values is None:
            raise RuntimeError(
                f"{type(self).__name__}.run() must be called first")

    # -- frame bookkeeping -----------------------------------------------

    def _analyzed_frames(self):
        """Trajectory frame index of each analyzed frame."""
        frames = getattr(self, "frames", None)
        if frames is not None and len(frames) == self.n_frames:
            return np.asarray(frames, dtype=int)
        return np.arange(self.n_frames)

    @contextlib.contextmanager
    def _at_analyzed_frame(self, k):
        """Temporarily move the trajectory to analyzed frame ``k``."""
        old = int(self.u.trajectory.frame)
        try:
            self.u.trajectory[int(self._analyzed_frames()[k])]  # pylint: disable=expression-not-assigned
            yield
        finally:
            self.u.trajectory[old]  # pylint: disable=pointless-statement

    def _resolve_frame_index(self, frame_index=None, n=None):
        """Normalize an index into the analyzed frames (None: the last)."""
        if n is None:
            n = getattr(self, "n_frames", None)
        if n is None:
            raise RuntimeError(
                "frame indices are only defined after run() started")
        n = int(n)
        if n <= 0:
            self._error(
                "no frames to analyze (check --frames/--start/--stop/--step)")
        if frame_index is None:
            return n - 1
        msg = f"frame index (--bfactor-frame) must be an integer, got {frame_index!r}"
        if isinstance(frame_index, bool):
            self._error(msg)
        try:
            k = operator.index(frame_index)
        except TypeError:
            self._error(msg)
        if not -n <= k < n:
            self._error(
                f"frame index (--bfactor-frame) {k} is out of range for {n} "
                f"analyzed frame(s); valid values are -{n} to {n - 1}",
                exc=IndexError,
            )
        return k + n if k < 0 else k

    def _output_frame(self, frame_index):
        """``(k, trajectory_frame)`` an output is written for."""
        self._require_run()
        k = self._resolve_frame_index(frame_index)
        return k, int(self._analyzed_frames()[k])

    # -- paths, identities, CSV ------------------------------------------

    def _output_path(self, output, default_name):
        path = Path(output) if output is not None else Path(
            self.outdir or ".") / default_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _rows_for(residue_resindices, atom_resindices):
        """Column (position in ``residue_resindices``) of each atom's residue."""
        residue_resindices = np.asarray(residue_resindices)
        order = np.argsort(residue_resindices)
        pos = np.searchsorted(residue_resindices, atom_resindices, sorter=order)
        return order[pos]

    @staticmethod
    def _atom_attr(atom_group, attribute, default=""):
        try:
            values = getattr(atom_group, attribute)
        except (AttributeError, mda.exceptions.NoDataError):
            return np.full(len(atom_group), default, dtype=object)
        return np.asarray([str(v).strip() for v in values], dtype=object)

    def _identity_columns(self, atom_group, prefix="", atoms=True):
        """Columns chain, resid, resname [, atom_name, atom_serial]."""
        p = f"{prefix}_" if prefix else ""
        names = [f"{p}chain", f"{p}resid", f"{p}resname"]
        cols = [
            self._atom_attr(atom_group, "chainIDs"),
            _resid_labels(atom_group),
            self._atom_attr(atom_group, "resnames"),
        ]
        if atoms:
            names += [f"{p}atom_name", f"{p}atom_serial"]
            cols += [
                self._atom_attr(atom_group, "names"),
                pdb_atom_ids(atom_group)
            ]
        return names, cols

    def _write_csv(self, output, default_name, header, batches, label):
        """Write ``header`` and every batch of rows; returns the path."""
        output = self._output_path(output, default_name)
        n = 0
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for batch in batches:
                batch = list(batch)
                writer.writerows(batch)
                n += len(batch)
        written(label, output, n, base=self.outdir or ".")
        return output

    # -- B-factor PDB ----------------------------------------------------

    def _ensure_topology_attr(self, name):
        try:
            getattr(self.u.atoms, name)
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr(name)

    def _ensure_tempfactors(self):
        self._ensure_topology_attr("tempfactors")

    def _ensure_occupancies(self):
        self._ensure_topology_attr("occupancies")

    @staticmethod
    def _strip_placeholder_cryst1(path, dims):
        """Remove the 1 A CRYST1 MDAnalysis writes for a Universe without a box."""
        if dims is None or np.allclose(np.asarray(dims[:3]), 1.0):
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text(
                "".join(l for l in lines if not l.startswith("CRYST1")),
                encoding="utf-8")

    def _residue_atom_rows(self):
        """All atoms of the column residues (sorted by index) and their column."""
        col_atoms = self._residue_column_atoms()
        atoms = col_atoms.residues.atoms
        cols = self._rows_for(col_atoms.resindices, atoms.resindices)
        order = np.argsort(atoms.indices, kind="stable")
        return atoms[order], cols[order]

    def write_bfactors(
            self, frame_index=None, output=None, *, values=None,
            default_name="b_factors.pdb", label="B-factors"):
        """
        PDB of one frame with each residue's value as B-factor, plus a .npy
        of all frames next to it.

        Every atom of a column residue (hydrogens included) gets that
        residue's value; all other atoms get 0. The values stay set as
        ``tempfactors`` on the Universe.

        :param values: (n_frames, n_columns) array; None uses the hook.
        """
        k, _ = self._output_frame(frame_index)
        matrix = np.asarray(
            self._residue_values() if values is None else values)
        output = self._output_path(output, default_name)
        self._ensure_tempfactors()

        row = matrix[k]
        finite = row[np.isfinite(row)]
        if finite.size and finite.max() > 999.99:
            warnings.warn(
                f"max B-factor value {finite.max():g} exceeds the PDB B-factor "
                f"column width (999.99); use {output.with_suffix('.npy').name} "
                "for the raw values", stacklevel=2,
            )
        atoms, cols = self._residue_atom_rows()
        self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
        atoms.tempfactors = row[cols]

        with self._at_analyzed_frame(k):
            self.u.atoms.write(str(output))
            dims = self.u.dimensions
        self._strip_placeholder_cryst1(output, dims)

        np.save(output.with_suffix(".npy"), matrix)
        written(label, output, base=self.outdir or ".")
        return output

    # -- statistics over analyzed frames ---------------------------------

    def _stats_bin_width(self):
        """Histogram bin width; None: one bin per integer value 0..max.

        Hosts with continuous values (e.g. SASA) override this.
        """
        return None

    def _stats_matrix(self, values):
        self._require_run()
        m = np.asarray(self._residue_values() if values is None else values)
        if m.ndim != 2:
            raise ValueError(
                f"statistics need a 2-D (frames, columns) array, got {m.shape}")
        return m

    def _bin_index(self, m):
        """Bin of every value, a finite mask, and the value of each bin."""
        finite = np.isfinite(m)
        if (m[finite] < 0).any():
            raise ValueError("residue statistics need non-negative values")
        width = self._stats_bin_width()  #pylint: disable=assignment-from-none
        clean = np.where(finite, m, 0)
        if width is None:
            if not np.array_equal(m[finite], np.round(m[finite])):
                raise ValueError(
                    f"{type(self).__name__}: non-integer values need "
                    "a bin width (override _stats_bin_width)")
            idx = clean.astype(np.int64)
            n_bins = int(idx[finite].max(initial=0)) + 1
            centres = np.arange(n_bins)  # 0..max, as before
        else:
            width = float(width)
            idx = np.floor(clean / width).astype(np.int64)
            n_bins = int(idx[finite].max(initial=0)) + 1
            centres = (np.arange(n_bins) + 0.5) * width  # bin centres
        return idx, finite, centres

    def residue_histograms(self, values=None):
        """
        ``(values, hist)``: ``hist[i, b]`` is the number of analyzed frames in
        which column ``i`` falls into bin ``b`` (bin value ``values[b]``).
        Non-finite values are ignored. Both arrays are read-only.
        """
        m = self._stats_matrix(values)
        idx, finite, centres = self._bin_index(m)
        n_cols, nb = m.shape[1], len(centres)
        cols = np.broadcast_to(np.arange(n_cols), m.shape)
        hist = np.bincount((cols * nb + idx)[finite],
                           minlength=n_cols * nb).reshape(n_cols,
                                                          nb).astype(np.int64)
        centres.flags.writeable = False
        hist.flags.writeable = False
        return centres, hist

    def residue_statistics(self, values=None, level=0.95):
        """
        Per column: ``mode`` (bin value with the most frames), ``lower`` /
        ``upper`` (equal-tailed ``level`` interval of the bin values),
        ``width`` (upper - lower) and ``n_frames`` (finite frames).
        """
        centres, hist = self.residue_histograms(values)
        n = hist.sum(axis=1)
        cum = np.cumsum(hist, axis=1)
        tail = (1.0 - level) / 2.0
        eps = 1e-9  # avoid 0.975*n rounding just above an integer cumulative count
        lo = np.argmax(cum >= tail * n[:, None] - eps, axis=1)
        hi = np.argmax(cum >= (1.0 - tail) * n[:, None] - eps, axis=1)
        c = centres.astype(float)
        empty = n == 0

        def pick(ix):
            return np.where(empty, np.nan, c[ix])

        lower, upper = pick(lo), pick(hi)
        return {
            "mode": pick(np.argmax(hist, axis=1)),
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "n_frames": n
        }

    def _bin_labels(self, centres):
        if self._stats_bin_width() is None:
            return [str(int(v)) for v in centres]
        return [f"{v:g}" for v in centres]

    def _column_identities(self):
        names, cols = self._identity_columns(
            self._residue_column_atoms(), atoms=False)
        return names, list(zip(*(c.tolist() for c in cols)))

    def write_histograms_csv(
            self, output=None, *, values=None,
            default_name="residue_histograms.csv", label="Residue histograms"):
        """One row per column residue, one column per bin value."""
        centres, hist = self.residue_histograms(values)
        names, idents = self._column_identities()
        rows = ([*ident, *h] for ident, h in zip(idents, hist.tolist()))
        return self._write_csv(
            output, default_name, [*names, *self._bin_labels(centres)], [rows],
            label)

    def write_statistics_csv(
            self, output=None, *, values=None,
            default_name="residue_statistics.csv", label="Residue statistics"):
        stats = self.residue_statistics(values)
        names, idents = self._column_identities()
        keys = ("mode", "lower", "upper", "width")
        rows = (
            [
                *ident, *(_fmt_float(stats[k][i]) for k in keys),
                int(stats["n_frames"][i])
            ] for i, ident in enumerate(idents))
        return self._write_csv(
            output, default_name, [*names, *keys, "n_frames"], [rows], label)

    def write_statistics_pdb(
            self, frame_index=None, output=None, *, values=None,
            default_name="b_factors_stats.pdb", label="Statistics B-factors"):
        """PDB with B-factor = mode and occupancy = interval width."""
        k, _ = self._output_frame(frame_index)
        stats = self.residue_statistics(values)
        output = self._output_path(output, default_name)
        self._ensure_tempfactors()
        self._ensure_occupancies()
        for name in ("mode", "width"):
            finite = stats[name][np.isfinite(stats[name])]
            if finite.size and finite.max() > 999.99:
                warnings.warn(
                    f"max {name} {finite.max():g} exceeds the PDB "
                    "column width (999.99)", stacklevel=2)
        atoms, cols = self._residue_atom_rows()
        old_bf = self.u.atoms.tempfactors.copy()
        old_occ = self.u.atoms.occupancies.copy()
        try:
            self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
            self.u.atoms.occupancies = np.zeros(self.u.atoms.n_atoms)
            atoms.tempfactors = np.nan_to_num(stats["mode"][cols])
            atoms.occupancies = np.nan_to_num(stats["width"][cols])
            with self._at_analyzed_frame(k):
                self.u.atoms.write(str(output))
                dims = self.u.dimensions
        finally:
            self.u.atoms.tempfactors = old_bf
            self.u.atoms.occupancies = old_occ
        self._strip_placeholder_cryst1(output, dims)
        written(label, output, base=self.outdir or ".")
        return output
