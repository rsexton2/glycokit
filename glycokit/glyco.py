#!/usr/bin/env python3
"""
GLYCO (GLYcan COverage) of protein surface residues (glycokit).

Counting modes (``--mode`` / ``--blocking``):

================  =========================================================
mode / blocking   description
================  =========================================================
default/vdw       heavy-atom starts (per-residue union), float64 pair test
                  (norm < ``--dc``), van der Waals sphere blocking test
                  (``--vdw-scale``, ``--vdw-probe``), own residue excluded
default/cylinder  as default, with the cylinder test (``--cr``, inclusive)
v1/box            reproduces the original GLYCO script
v2/cylinder       reproduces GLYCO-2
================  =========================================================
"""

from __future__ import annotations

import contextlib
import csv
import difflib
import json
import multiprocessing as mp
import operator
import os
import re
import sys
import warnings
from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.guesser.default_guesser import DefaultGuesser
from MDAnalysis.guesser.tables import vdwradii as VDW_RADII
from scipy.spatial import cKDTree
from scipy.stats import t as t_res

from glycokit.config import (
    _BLOCKING_TESTS,
    _BOX_STATE,
    _JOB_ERRORS,
    _KD_PAD,
    _LOAD_ERRORS,
    _SELECTION_ERRORS,
    _V1_RESNAMES,
    _V2_RESNAMES,
    VMD_SCRIPT_NAME,
)
from glycokit.console import (
    diagnostic,
    error,
    heading,
    set_color_mode,
    style,
    written,
)
from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.parsers import base_parser
from glycokit.utils import (
    _box_chunk,
    _build_x_buckets,
    _cylinder_blocked,
    _first_altloc,
    _fmt_float,
    _init_box_worker,
    _legacy_coords,
    _package_versions,
    _parser_defaults,
    _resid_labels,
    _sphere_blocked,
    confirm_overwrite,
    heavy_atoms,
    load_universe,
    pdb_atom_ids,
    validate_arguments,
)
from glycokit.warnlog import warning_log


class GlycoAnalysis(AnalysisBase):
    """
    GLYCO (Glycan Coverage): glycan coverage of protein surface residues.

    Counting modes (``mode`` / ``blocking``):

    default / vdw
        Heavy-atom starts with a per-residue union, float64 pair selection
        (norm < cutoff); a blocker blocks if the closed start->glycan
        segment touches its sphere of radius ``vdw_scale * r_vdW +
        vdw_probe``. Own residue and blockers whose sphere contains the
        start atom are excluded.
    default / cylinder
        As default, with the GLYCO-2 cylinder test in float64 (radius
        ``cylinder_radius``, inclusive).
    v1 / box
        Original GLYCO: integer cutoff (strict '<'), legacy hydrogen filter,
        ATOM-record blockers, legacy FreeSASA set, int-truncated +-1 A box.
    v2 / cylinder
        GLYCO-2: float32 pair and cylinder arithmetic, start residue CAN
        block; legacy hydrogen filter, resname filter and FreeSASA set.

    Methods taking ``mode`` / ``blocking`` follow the run when they are
    None. Per-call overrides only change the visibility test, not the atom
    groups or SASA built in :meth:`_prepare`.

    Results (after :meth:`run`) are in ``results``: ``glycan_counts`` and
    ``sasa`` (shape (n_frames, n_residues), columns follow ``prot_atoms``),
    ``n_surface_residues``, ``n_blocked_pairs``, ``mean_blocked_distance``,
    ``trajectory_frames``, ``count_values`` (shape (n_values,)) and
    ``residue_histograms`` (shape (n_residues, n_values)).

    :param u: Universe to analyze.
    :type u: MDAnalysis.Universe
    :param bool verbose: Diagnostic output and progress bars.
    :param kwargs: Parameters (any :func:`base_parser` destination, e.g.
        ``gly_sel``, ``mode``, ``vdw_scale``).
    :raises TypeError: On unknown parameters.
    :raises ValueError: On invalid parameters or selections.
    """

    # region init

    _analysis_algorithm_is_parallelizable = False

    # Parameters set by _apply_params from base_parser() destinations.
    # Declared here for linters and readers; values come from the parser.
    mode: str
    blocking: str | None
    n_workers: int
    top: str | None
    traj: list[str] | None
    indir: str | None
    outdir: str
    prot_sel: str
    gly_sel: str
    block_sel: str
    sasa_env: str | None
    sasa_hetatm: bool
    sasa_probe: float
    area_cutoff: float
    distance_cutoff: float
    cylinder_radius: float
    vdw_scale: float
    vdw_probe: float
    frame_start: int | None
    frame_stop: int | None
    frame_step: int | None
    frame_list: list[int] | None
    bfactor_frame: int | None
    write_excluded: bool
    write_blockers: bool
    write_vmd: bool
    write_stats: bool
    write_chimerax: bool
    color: str
    force: bool
    in_types: str
    verbose: bool

    @classmethod
    def get_supported_backends(cls):
        """
        Backends usable with :meth:`run` (serial only).

        :rtype: tuple[str]
        """
        return ("serial",)

    def __init__(self, u, verbose=False, *, _parser=None, _args=None, **kwargs):
        super().__init__(u.trajectory, verbose=verbose)
        self.u = u
        self.parser = _parser
        self.progress_file = os.environ.get("GLYCO_PROGRESS_FILE")
        self._run_slice = {}
        self._write_outputs = True

        # internal caches (reset again in _prepare)
        self._frame_cache = None
        self._blocker_radii = None
        self._start_residue_keys = None
        self._blocker_residue_keys = None

        # state built in _prepare
        self.frame_count = 0
        self._output_k = None
        self._mode = self._blocking = None
        self._legacy = False
        self.prot_atoms = self.blocker_atoms = self.gly_atoms = self.start_atoms = None
        self.residue_areas = (
            None  # _require_run() relies on this being None before run()
        )
        self._start_residue_atoms = None
        self._residue_resindices = None
        self._n_res_rows = None
        self._blocker_resindex = None
        self._prot_atom_res_row = None
        self._start_res_row = None
        self._start_resindex = None
        self._progress_every = None

        # Single source of truth for defaults: base_parser(), parsed once.
        defaults = _parser_defaults()
        self._param_names = sorted(defaults)
        self._from_cli = _args is not None
        self._apply_params(defaults, _args, kwargs)

        self._validate_args()
        self._validate_atom_selections()

    @classmethod
    def from_args(cls, u, args, parser=None, verbose=False, **overrides):
        """
        Build from parsed CLI arguments.

        :param u: Universe to analyze.
        :type u: MDAnalysis.Universe
        :param args: Parsed arguments.
        :type args: argparse.Namespace
        :param parser: Errors go through ``parser.error`` if given.
        :type parser: argparse.ArgumentParser or None
        :param bool verbose: Passed to :class:`AnalysisBase`.
        :param overrides: Parameters overriding ``args``.
        :rtype: GlycoAnalysis
        """
        return cls(u, verbose=verbose, _parser=parser, _args=args, **overrides)

    # endregion

    # -------------------------------------------------------------
    # region Mode Resolution
    # ------------------------------------------------------------

    def _resolve_test(self, mode=None, blocking=None):
        """
        Resolve the counting mode and blocking test.


        :param mode: ``'default'``, ``'v1'`` or ``'v2'``; None uses ``mode``.
        :type mode: str or None
        :param blocking: ``'vdw'``, ``'cylinder'`` or ``'box'``; None uses
            ``blocking`` if the mode allows it, else the mode's default test.
        :type blocking: str or None
        :return: ``(mode, blocking)``
        :rtype: tuple[str, str]
        :raises ValueError: On an unknown mode or a blocking test the mode does
            not allow.
        """
        mode = self.mode if mode is None else mode
        if mode not in _BLOCKING_TESTS:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {sorted(_BLOCKING_TESTS)}"
            )

        allowed = _BLOCKING_TESTS[mode]
        if blocking is None:
            blocking = self.blocking if self.blocking in allowed else allowed[0]
        if blocking not in allowed:
            raise ValueError(
                f"mode {mode!r} does not support blocking={blocking!r} "
                f"(allowed: {', '.join(allowed)})"
            )
        return mode, blocking

    def _n_workers(self, processes=False):
        """
        Resolve ``n_workers``.

        :param bool processes: For the box-test process pool, -1 becomes the
            number of available CPUs; KD-tree queries take -1 directly.
        :rtype: int
        """
        n = int(self.n_workers)
        if n == -1 and processes:
            try:
                return max(1, len(os.sched_getaffinity(0)))  # honours CPU limits
            except AttributeError:
                return max(1, os.cpu_count() or 1)
        return n

    def _kd_query_kwargs(self):
        """
        Keyword arguments for every KD-tree ``query_ball_point`` call.

        :rtype: dict
        """
        return {"workers": self._n_workers(), "return_sorted": False}

    # endregion

    # -------------------------------------------------------------
    # region Configuration
    # --------------------------------------------------------------

    def _apply_params(self, defaults, args, kwargs):
        """
        Set ``defaults < args < kwargs`` as attributes; unknown kwargs fail.

        :raises TypeError: On unknown keyword arguments.
        """
        values = dict(defaults)
        if args is not None:
            values.update(vars(args))

        known = sorted(values)
        unknown = sorted(set(kwargs) - set(known))
        if unknown:
            parts = []
            for name in unknown:
                close = difflib.get_close_matches(name, known, n=1)
                parts.append(
                    f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
                )
            raise TypeError(
                f"{type(self).__name__} got unknown parameter(s): " + ", ".join(parts)
            )

        values.update(kwargs)
        for name, value in values.items():
            setattr(self, name, value)

    def _describe_residues(self, resindices, limit=10):
        """
        Short ``RESNAME+resid`` list of residues for error messages.

        :rtype: str
        """
        resindices = [int(r) for r in resindices]
        labels = [f"{r.resname}{r.resid}" for r in self.u.residues[resindices[:limit]]]
        more = len(resindices) - limit
        return ", ".join(labels) + (f" (+{more} more)" if more > 0 else "")

    def _error(self, message, exc=ValueError):
        """
        Report a problem with the user's input or data.

        CLI (``parser`` given): ``parser.error()`` prints usage and message
        and exits with status 2. Library: raises ``exc(message)``.

        :param str message: Error message.
        :param type exc: Exception class raised in library use.
        """
        if self.parser is not None:
            self.parser.error(message)
        raise exc(message)

    def _require_run(self):
        """
        :raises RuntimeError: If :meth:`run` has not been called yet.
        """
        if getattr(self, "residue_areas", None) is None:
            raise RuntimeError(f"{type(self).__name__}.run() must be called first")

    def _validate_args(self):
        """
        Validate parameters via :func:`validate_arguments`.

        Library use without input files skips the file checks.
        """
        library_mode = (
            self.parser is None
            and self.top is None
            and self.indir is None
            and not self.traj
        )
        validate_arguments(
            self, self._error, check_inputs=not library_mode, warn=not self._from_cli
        )

    def run(
        self,
        start=None,
        stop=None,
        step=None,
        frames=None,
        verbose=None,
        n_workers=None,
        n_parts=None,
        backend=None,
        *,
        unsupported_backend=False,
        progressbar_kwargs=None,
        write=True,
    ):
        """
        Run the analysis; see :meth:`MDAnalysis.analysis.base.AnalysisBase.run`.

        Differences from the base class:

        * If ``start``, ``stop``, ``step`` and ``frames`` are all None, the
        frame selection given at construction (``frame_start``,
        ``frame_stop``, ``frame_step``, ``frame_list``) is used.
        * The same frame selection is used for the FreeSASA pass.

        :param bool write: Write the output files to ``outdir``; with False
            only ``results`` is filled (the writers stay callable).
        :rtype: GlycoAnalysis
        """
        if all(v is None for v in (start, stop, step, frames)):
            start, stop, step = self.frame_start, self.frame_stop, self.frame_step
            frames = self.frame_list
        self._run_slice = {"start": start, "stop": stop, "step": step, "frames": frames}
        self._write_outputs = bool(write)
        return super().run(
            **self._run_slice,
            verbose=verbose,
            n_workers=n_workers,
            n_parts=n_parts,
            backend=backend,
            unsupported_backend=unsupported_backend,
            progressbar_kwargs=progressbar_kwargs,
        )

    def _compat_selection(self, sel, blockers=False, mode=None):
        """
        Restrict a selection to the legacy residue names (v1 / v2 only).

        v1 uses ``_V1_RESNAMES`` and, for blockers, ATOM records only; v2
        uses ``_V2_RESNAMES``. Glycan selections are never passed here.

        :param str sel: Selection string.
        :param bool blockers: The selection is the blocker selection.
        :param mode: None follows the run.
        :type mode: str or None
        :rtype: str
        """
        mode = self._resolve_test(mode)[0]
        if not sel or mode == "default":
            return sel
        names = _V1_RESNAMES if mode == "v1" else _V2_RESNAMES
        sel = f"({sel}) and (resname {names})"
        if blockers and mode == "v1":
            candidate = f"({sel}) and record_type ATOM"
            try:
                self.u.select_atoms(candidate)
                return candidate
            except _SELECTION_ERRORS:
                warnings.warn(
                    "record_type is unavailable for this topology; "
                    "v1 blockers are NOT restricted to ATOM records",
                    stacklevel=2,
                )
        return sel

    def _validate_atom_selections(self):
        """
        Validate protein, glycan and blocker selections against the Universe.

        The user's ``prot_sel`` / ``gly_sel`` / ``block_sel`` are left
        unchanged. The effective selections (legacy filters applied in v1/v2)
        are stored in ``_validated_*_sel``. Idempotent, so it is called
        again in :meth:`_prepare` in case the mode was changed.

        :raises ValueError: On invalid selections (library use).
        """
        prot_sel = self._compat_selection((self.prot_sel or "").strip())
        gly_sel = (self.gly_sel or "").strip()
        block_sel = self._compat_selection(
            (self.block_sel or "").strip(), blockers=True
        )

        results = {}
        for name, selection in (
            ("protein", prot_sel),
            ("glycan", gly_sel),
            ("blockers", block_sel),
        ):
            if not selection:
                results[name] = {
                    "valid": False,
                    "message": "Selection cannot be empty.",
                }
                continue
            try:
                atoms = self.u.select_atoms(f"({selection})")
            except _SELECTION_ERRORS as e:
                results[name] = {
                    "valid": False,
                    "message": f"Invalid selection ('{selection}'): {e}",
                }
                continue
            if len(atoms) == 0:
                results[name] = {
                    "valid": False,
                    "message": f"0 atoms selected ('{selection}')",
                }
            else:
                results[name] = {"valid": True, "message": f"{len(atoms):,} atoms"}

        if (
            results["protein"]["valid"]
            and len(self.u.select_atoms(f"({prot_sel}) and name CA")) == 0
        ):
            results["protein"] = {
                "valid": False,
                "message": f"no CA atoms in protein selection ('{prot_sel}')",
            }

        self.selection_validation_results = results
        if not all(r["valid"] for r in results.values()):
            problems = "; ".join(
                f"{name}: {r['message']}"
                for name, r in results.items()
                if not r["valid"]
            )
            self._error(f"invalid atom selection(s): {problems}")

        self._validated_prot_sel = prot_sel
        self._validated_gly_sel = gly_sel
        self._validated_block_sel = block_sel

        if self.sasa_env:
            try:
                self.u.select_atoms(f"({self.sasa_env})")
            except _SELECTION_ERRORS as e:
                self._error(f"invalid --sasa-env selection ('{self.sasa_env}'): {e}")

    def selection_report(self):
        """
        Human-readable summary of the validated selections.

        :rtype: str
        """
        lines = ["Atom selections:"]
        for name, r in self.selection_validation_results.items():
            flag = "ok" if r["valid"] else "INVALID"
            lines.append(f"  {name}: {flag} - {r['message']}")
        return "\n".join(lines)

    # endregion

    # -------------------------------------------------------------
    # region FreeSASA setup
    # -------------------------------------------------------------

    def _sasa_atoms(self):
        """
        FreeSASA atom set.

        Legacy (v1/v2, without ``--sasa-env``): the freesasa executable's set
        on the whole PDB (heavy atoms of ATOM records, first altloc;
        ``--sasa-hetatm`` adds HETATM). Otherwise: start residues + blockers
        (+ ``--sasa-env``).

        :rtype: MDAnalysis.core.groups.AtomGroup
        """

        if self._legacy and not self.sasa_env:
            sel = "all" if self.sasa_hetatm else "record_type ATOM"
            try:
                return _first_altloc(heavy_atoms(self.u.select_atoms(sel)))
            except _SELECTION_ERRORS as e:
                self._error(
                    "could not build the legacy FreeSASA atom set "
                    f"(needs PDB record types; use --sasa-env): {e}"
                )
        sel = f"({self._validated_prot_sel}) or ({self._validated_block_sel})"
        if self.sasa_env:
            sel += f" or ({self.sasa_env})"
        return heavy_atoms(self.u.select_atoms(sel))

    def _check_sasa_structure(self, fsa):
        """FreeSASA structure mismatch: error in legacy modes, else warning."""
        if fsa.structure_valid:
            return
        msg = (
            "FreeSASA structure did not match the MDAnalysis residues; SASA "
            "used MDAnalysis vdW radii instead of ProtOr, so areas will not "
            "match the freesasa executable"
        )
        if self._legacy:
            self._error(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    def _check_sasa_coverage(self, missing):
        """Start residues without SASA: buried in legacy modes, else error."""
        if not missing:
            return
        if self._legacy:
            warnings.warn(
                f"{len(missing)} start residue(s) are not in the FreeSASA "
                "set and are treated as buried (SASA 0): "
                f"{self._describe_residues(missing)}",
                stacklevel=2,
            )
        else:
            self._error(
                f"FreeSASA has no area for {len(missing)} start residue(s) "
                f"of the protein selection: {self._describe_residues(missing)}. "
                "Add their atoms to the SASA set (e.g. with --sasa-env)"
            )

    # endregion

    # -------------------------------------------------------------
    # region Run Lifecycle
    # -------------------------------------------------------------

    def _prepare(self):
        """Build atom groups, run FreeSASA and allocate results (AnalysisBase hook)."""
        # pylint: disable=too-many-locals,too-many-statements

        self.frame_count = 0
        # fail on a bad --bfactor-frame before the (slow) FreeSASA pass
        self._output_k = self._resolve_frame_index(self.bfactor_frame)

        self._mode, self._blocking = self._resolve_test()
        self._legacy = self._mode != "default"
        self._validate_atom_selections()  # mode may have changed

        self._write_progress(0, self.n_frames)
        if self._verbose:
            print(self.selection_report())

        prot_sel = self._validated_prot_sel
        gly_sel = self._validated_gly_sel
        block_sel = self._validated_block_sel

        # start residues (heavy atoms) and their CA output columns
        self._start_residue_atoms = heavy_atoms(self.u.select_atoms(f"({prot_sel})"))
        self.prot_atoms = self.u.select_atoms(f"({prot_sel}) and name CA")
        self.blocker_atoms = heavy_atoms(
            self.u.select_atoms(f"({block_sel})"), legacy=self._legacy
        )
        self.gly_atoms = heavy_atoms(
            self.u.select_atoms(f"({gly_sel})"), legacy=self._legacy
        )

        # vdW radii up front, so missing-radius warnings appear before the loop
        self._blocker_radii = (
            self._blocker_vdw_radii() if self._blocking == "vdw" else None
        )

        self._residue_resindices = np.unique(self._start_residue_atoms.resindices)
        self._n_res_rows = len(self._residue_resindices)

        # ---------------- FreeSASA ----------------
        sasa_ag = self._sasa_atoms()
        if len(sasa_ag) == 0:
            self._error("FreeSASA atom set is empty")
        sasa_res = np.unique(sasa_ag.resindices)

        fsa = FreesasaAnalysis(sasa_ag, self.sasa_probe).run(
            verbose=self._verbose,
            progressbar_kwargs={"desc": "Running FreeSASA Analysis"},
            **self._run_slice,
        )
        self._diag(
            "[SASA diagnostic]",
            f"mode={self._mode}",
            f"universe_atoms={len(self.u.atoms)}",
            f"sasa_atoms={len(sasa_ag)}",
            f"sasa_residues={len(sasa_res)}",
            f"start_residues={self._n_res_rows}",
            f"structure_valid={fsa.structure_valid}",
        )
        self._check_sasa_structure(fsa)

        areas = np.asarray(fsa.results.residue_areas, dtype=float)
        if areas.shape != (self.n_frames, len(sasa_res)):
            raise RuntimeError(
                "Unexpected FreeSASA residue-area shape: got "
                f"{areas.shape}, expected {(self.n_frames, len(sasa_res))}"
            )

        column_of = {int(r): i for i, r in enumerate(sasa_res)}
        missing = [int(r) for r in self._residue_resindices if int(r) not in column_of]
        self._check_sasa_coverage(missing)

        self.residue_areas = np.zeros((self.n_frames, self._n_res_rows))
        for row, resindex in enumerate(self._residue_resindices):
            col = column_of.get(int(resindex))
            if col is not None:
                self.residue_areas[:, row] = areas[:, col]

        self._diag(
            "[SASA diagnostic]",
            f"surface_residues={int(np.count_nonzero(self.residue_areas[0] >= self.area_cutoff))}",
            f"area_cutoff={self.area_cutoff}",
        )

        # ---------------- residue bookkeeping ----------------
        self._blocker_resindex = np.asarray(self.blocker_atoms.resindices)

        if not np.isin(self.prot_atoms.resindices, self._residue_resindices).all():
            self._error(
                "some atoms named CA in the protein selection were "
                "classified as hydrogens; check their element assignments"
            )
        ca_res, ca_n = np.unique(self.prot_atoms.resindices, return_counts=True)
        if (ca_n > 1).any():
            self._error(
                "more than one atom named CA in residue(s) "
                f"{self._describe_residues(ca_res[ca_n > 1])}; results are "
                "per residue and need exactly one (alternate locations?)"
            )
        self._prot_atom_res_row = np.searchsorted(
            self._residue_resindices, self.prot_atoms.resindices
        )

        # line-of-sight starts: every heavy atom of each start residue
        self.start_atoms = (
            heavy_atoms(self._start_residue_atoms, legacy=True)
            if self._legacy
            else self._start_residue_atoms
        )
        self._start_res_row = np.searchsorted(
            self._residue_resindices, self.start_atoms.resindices
        )
        self._start_resindex = np.asarray(self.start_atoms.resindices)

        # residue keys for the box test, built on first use
        self._start_residue_keys = None
        self._blocker_residue_keys = None
        self._frame_cache = None

        n_ca = self.prot_atoms.n_atoms
        self.results = Results()
        self.results.glycan_counts = np.zeros((self.n_frames, n_ca), dtype=np.int64)
        self.results.sasa = self.residue_areas[:, self._prot_atom_res_row]
        self.results.n_surface_residues = np.zeros(self.n_frames, dtype=np.int64)
        self.results.n_blocked_pairs = np.zeros(self.n_frames, dtype=np.int64)
        self.results.mean_blocked_distance = np.full(self.n_frames, np.nan)
        self.results.trajectory_frames = self._analyzed_frames()

        self._progress_every = max(1, self.n_frames // 100)

        geometry = {
            "box": f"box_cutoff={int(self.distance_cutoff)}",
            "cylinder": f"cylinder_radius={self.cylinder_radius}",
            "vdw": f"vdw_scale={self.vdw_scale} vdw_probe={self.vdw_probe}",
        }[self._blocking]
        self._diag(
            "[diagnostic]",
            f"mode={self._mode}",
            f"blocking={self._blocking}",
            geometry,
            f"n_workers={self._n_workers()}",
            f"start_atoms={len(self.start_atoms)}",
            f"blocker_atoms={len(self.blocker_atoms)}",
            f"glycan_atoms={len(self.gly_atoms)}",
            f"protein_CA={len(self.prot_atoms)}",
        )

    @property
    def all_counts(self):
        """Per-CA glycan counts, shape (n_frames, n_residues); see ``results``."""
        return self.results.glycan_counts

    def _single_frame(self):
        """
        Count visible glycan atoms for the current frame (AnalysisBase hook).

        The ``--bfactor-frame`` frame is cached for the CSV writers.
        """
        k = self._frame_index
        mode, blocking = self._mode, self._blocking

        start_idx = self.surface_start_indices(k)
        self.results.n_surface_residues[k] = len(
            np.unique(self._start_res_row[start_idx])
        )

        cand, vis, n_blocked, mean_d, blockers, dists = self._visibility(
            start_idx, mode, blocking
        )
        self.all_counts[k] = self._counts_from_visible_keys(vis)
        self.results.n_blocked_pairs[k] = n_blocked
        self.results.mean_blocked_distance[k] = mean_d

        if k == self._output_k:
            self._frame_cache = (
                k,
                mode,
                blocking,
                start_idx,
                cand,
                vis,
                blockers,
                dists,
            )

        self._diag(
            "[diagnostic]",
            f"frame={k}",
            f"surface_start_atoms={len(start_idx)}",
            f"candidate_keys={len(cand)}",
            f"visible_keys={len(vis)}",
            f"blocker_rows={'n/a' if blockers is None else len(blockers)}",
            f"nonzero={np.count_nonzero(self.all_counts[k])}",
            f"total={int(self.all_counts[k].sum())}",
        )

        self.frame_count += 1
        if (
            self.frame_count % self._progress_every == 0
            or self.frame_count == self.n_frames
        ):
            self._write_progress(self.frame_count, self.n_frames)

    def _conclude(self):
        """Store the residue histograms and write the requested output files."""
        (
            self.results.count_values,
            self.results.residue_histograms,
        ) = self._compute_residue_histograms()

        if self._write_outputs:
            os.makedirs(self.outdir, exist_ok=True)
            heading(f"Writing results to {style(self.outdir + '/', 'cyan')}")

            self.write_bfactors(frame_index=self.bfactor_frame)
            self.write_frame_summary()
            self.write_residue_csv(frame_index=self.bfactor_frame)

            if self.write_excluded:
                self.write_excluded_glycan_atoms_csv(frame_index=self.bfactor_frame)

            if self.write_blockers and self._blocking != "box":
                self.write_blockers_csv(frame_index=self.bfactor_frame)

            if self.write_vmd:
                self.write_vmd_user_script()

            if self.write_stats or self.write_chimerax:
                self.write_histograms_csv()
                self.write_statistics_csv()
                stats_pdb = self.write_statistics_pdb(frame_index=self.bfactor_frame)
                if self.write_chimerax:
                    self.write_chimerax_script(structure=stats_pdb)

            self.write_run_config()
        self._frame_cache = None

    # endregion

    # -------------------------------------------------------------
    # region Geometry
    # ------------------------------------------------------------

    def surface_residue_rows(self, frame_index=None):
        """
        Residue rows that are surface residues (SASA >= area_cutoff).

        Only SASA is read; the trajectory position does not matter.

        :param frame_index: Analyzed-frame index. None means the frame being
            analyzed during :meth:`run`, and the last analyzed frame afterwards.
        :type frame_index: int or None
        :rtype: numpy.ndarray of int
        :raises RuntimeError: Before :meth:`run`.
        """
        self._require_run()
        if frame_index is None:
            frame_index = self._frame_index
        return np.flatnonzero(self.residue_areas[frame_index] >= self.area_cutoff)

    def surface_start_indices(self, frame_index=None):
        """
        Indices (into ``start_atoms``) of start atoms in surface residues.

        Only SASA is read; the trajectory position does not matter.

        :param frame_index: Analyzed-frame index. None means the frame being
            analyzed during :meth:`run`, and the last analyzed frame afterwards.
        :type frame_index: int or None
        :rtype: numpy.ndarray of int
        :raises RuntimeError: Before :meth:`run`.
        """
        rows = self.surface_residue_rows(frame_index)
        return np.flatnonzero(np.isin(self._start_res_row, rows))

    def _cutoff_pairs(self, start_idx, ctx):
        """
        (start, glycan) pairs with ``norm < cutoff`` in the pair dtype of ``ctx``.

        float64, or float32 in v2 (GLYCO-2's ``c_worker``). A padded,
        inclusive float64 KD-tree prefilter precedes the exact test.

        :param numpy.ndarray start_idx: Indices into ``start_atoms``.
        :param dict ctx: Output of :meth:`_blocking_context`; its positions and
            glycan KD-tree are reused.
        :return: ``(i, g, dist, P, G)``: ``i`` indexes ``start_idx``, ``g``
            indexes ``gly_atoms``; ``P`` / ``G`` are in the pair dtype.
        :rtype: tuple[numpy.ndarray, ...]
        """
        dtype, G, gtree = ctx["dtype"], ctx["G"], ctx["gtree"]
        P = ctx["P_all"][np.asarray(start_idx, dtype=np.intp)]
        empty = np.empty(0, dtype=np.intp)
        no_dist = np.empty(0, dtype=dtype)
        if len(P) == 0 or gtree is None:
            return empty, empty, no_dist, P, G

        cutoff = float(self.distance_cutoff)
        neigh = gtree.query_ball_point(
            P.astype(np.float64), cutoff + _KD_PAD, **self._kd_query_kwargs()
        )
        sizes = np.fromiter(map(len, neigh), dtype=np.intp, count=len(neigh))
        if sizes.sum() == 0:
            return empty, empty, no_dist, P, G
        i = np.repeat(np.arange(len(P), dtype=np.intp), sizes)
        g = np.concatenate(neigh).astype(np.intp, copy=False)
        d = np.linalg.norm(G[g] - P[i], axis=1)  # in the pair dtype
        keep = d < dtype(cutoff)
        return i[keep], g[keep], d[keep], P, G

    def _blocker_vdw_radii(self, default=1.7, overrides=None):
        """
        Per-atom van der Waals radii (Å) for ``blocker_atoms``.

        Looked up by element in MDAnalysis' ``vdwradii`` table; missing
        elements are guessed from atom names (names are never looked up
        directly: 'CA' there would be calcium). Unknown elements get
        ``default``.

        :param float default: Radius for elements missing from the table.
        :param overrides: Extra or replacement ``{element: radius}`` entries.
        :type overrides: dict or None
        :rtype: numpy.ndarray of float64
        """
        table = {str(k).upper(): float(v) for k, v in VDW_RADII.items()}
        if overrides:
            table.update({str(k).upper(): float(v) for k, v in overrides.items()})

        atoms = self.blocker_atoms
        try:
            elements = np.asarray(atoms.elements, dtype=object)
        except (AttributeError, mda.exceptions.NoDataError):
            elements = np.full(len(atoms), "", dtype=object)

        missing = np.array([not str(e).strip() for e in elements], dtype=bool)
        if missing.any():
            guess = DefaultGuesser(self.u).guess_atom_element
            names = atoms.names
            for k in np.flatnonzero(missing):
                elements[k] = guess(names[k])

        radii = np.array(
            [table.get(str(e).strip().upper(), np.nan) for e in elements],
            dtype=np.float64,
        )
        unknown = np.isnan(radii)
        if unknown.any():
            bad = sorted({str(e) for e in elements[unknown]})
            warnings.warn(
                f"No vdW radius for element(s) {bad}; using {default} Å", stacklevel=2
            )
            radii[unknown] = default
        return radii

    def _blocking_context(
        self,
        mode,
        blocking,
        exclude_own_residue=None,
        exclude_start_overlap=True,
        radii=None,
        scale=None,
        probe=None,
    ):
        """
        Per-frame arrays and KD-trees shared by every start chunk.

        :param str mode: Resolved counting mode (v2 -> float32).
        :param str blocking: ``'vdw'`` or ``'cylinder'``.
        :param exclude_own_residue: None means True, except False in v2.
        :type exclude_own_residue: bool or None
        :param bool exclude_start_overlap: vdW only; see
          :func:`_sphere_blocked`.
        :param radii: vdW radii per blocker; None uses the cached radii.
        :type radii: numpy.ndarray or None
        :param scale: Radius scale (> 0); None uses ``vdw_scale``.
        :type scale: float or None
        :param probe: Added radius in Å (>= 0); None uses ``vdw_probe``.
        :type probe: float or None
        :rtype: dict
        :raises ValueError: On invalid ``radii``, ``scale`` or ``probe``.
        """
        v2 = mode == "v2"
        cyl = blocking == "cylinder"
        if exclude_own_residue is None:
            exclude_own_residue = not v2
        dtype = np.float32 if v2 else np.float64

        G = self.gly_atoms.positions.astype(dtype)
        G64 = G.astype(np.float64, copy=False)
        B = np.ascontiguousarray(self.blocker_atoms.positions, dtype=np.float64)
        ctx = {
            "v2": v2,
            "cylinder": cyl,
            "dtype": dtype,
            "exclude_own_residue": bool(exclude_own_residue),
            "exclude_start_overlap": bool(exclude_start_overlap),
            "P_all": self.start_atoms.positions.astype(dtype),
            "G": G,
            "G64": G64,
            "gtree": cKDTree(G64) if len(G64) else None,
            "B": B,
            "btree": cKDTree(B) if len(B) else None,
        }

        if cyl:
            ctx["radius"] = float(self.cylinder_radius)
            ctx["search_radius"] = ctx["radius"]
            return ctx

        if radii is None:
            if self._blocker_radii is None:
                self._blocker_radii = self._blocker_vdw_radii()
            radii = self._blocker_radii
        scale = float(self.vdw_scale if scale is None else scale)
        probe = float(self.vdw_probe if probe is None else probe)
        if scale <= 0:
            raise ValueError(f"vdW scale must be positive, got {scale}")
        if probe < 0:
            raise ValueError(f"vdW probe must be non-negative, got {probe}")
        R = scale * np.asarray(radii, dtype=np.float64) + probe
        if R.shape != (len(B),):
            raise ValueError(f"radii must have shape ({len(B)},), got {R.shape}")
        ctx["R2"] = R * R
        ctx["r_max"] = float(R.max()) if len(R) else 0.0
        ctx["search_radius"] = ctx["r_max"]
        return ctx

    def _blocking_hits(self, sub, i, g, dist, P, ctx, chunk=20000):
        """
        Blocking test on precomputed cutoff pairs (vdW or cylinder, from
        ``ctx['cylinder']``).

        Broad phase: every point that can block a segment lies within
        ``half length + search_radius`` of its midpoint. Exact test:
        :func:`_cylinder_blocked` (radius ``cylinder_radius``, inclusive, in
        the pair dtype, so float32 in v2) or :func:`_sphere_blocked` (float64,
        radius ``vdw_scale * r_vdW + vdw_probe``). The start atom itself never
        blocks; its residue is ignored if ``ctx['exclude_own_residue']``.

        :param numpy.ndarray sub: Start-atom indices of this chunk.
        :param numpy.ndarray i: Pair start indices (local to ``sub``).
        :param numpy.ndarray g: Pair glycan indices.
        :param numpy.ndarray dist: Pair distances.
        :param numpy.ndarray P: Start coordinates of ``sub`` (pair dtype).
        :param dict ctx: Output of :meth:`_blocking_context`.
        :param int chunk: Pairs per batch.
        :return: Lists of ``(n, 3)`` blocker rows and float64 distances.
        :rtype: tuple[list, list]
        """
        cyl = ctx["cylinder"]
        B, btree = ctx["B"], ctx["btree"]
        P64 = P.astype(np.float64, copy=False)
        G64 = ctx["G64"]
        # exact-test coordinates: pair dtype for the cylinder (GLYCO-2 float32),
        # float64 for the sphere test
        P_test, G_test = (P, ctx["G"]) if cyl else (P64, G64)
        reach = ctx["search_radius"] + _KD_PAD
        kw = self._kd_query_kwargs()
        blocker_ix = self.blocker_atoms.ix
        sub_res = self._start_resindex[sub]
        sub_ix = self.start_atoms.ix[sub]

        out, dout = [], []
        for s in range(0, len(i), chunk):
            ic, gc, dc = i[s : s + chunk], g[s : s + chunk], dist[s : s + chunk]

            # broad phase around each segment midpoint
            mid = 0.5 * (P64[ic] + G64[gc])
            half = 0.5 * np.linalg.norm(P64[ic] - G64[gc], axis=1)
            cand = btree.query_ball_point(mid, half + reach, **kw)
            counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
            if counts.sum() == 0:
                continue
            jj = np.concatenate(cand).astype(np.intp, copy=False)
            rep = np.repeat(np.arange(len(ic)), counts)

            # the start atom itself never blocks; optionally its residue
            mask = blocker_ix[jj] != sub_ix[ic[rep]]
            if ctx["exclude_own_residue"]:
                mask &= self._blocker_resindex[jj] != sub_res[ic[rep]]
            jj, rep = jj[mask], rep[mask]
            if len(jj) == 0:
                continue

            i_r, g_r = ic[rep], gc[rep]
            if cyl:
                hit = _cylinder_blocked(P_test, G_test, B, i_r, g_r, jj, ctx["radius"])
            else:
                hit = _sphere_blocked(
                    P_test,
                    G_test,
                    B,
                    i_r,
                    g_r,
                    jj,
                    ctx["R2"][jj],
                    ctx["exclude_start_overlap"],
                )
            if hit.any():
                out.append(np.stack([sub[i_r[hit]], jj[hit], g_r[hit]], axis=1))
                dout.append(dc[rep][hit].astype(np.float64))
        return out, dout

    def _iter_blocked_chunks(self, start_idx, ctx, start_chunk=2000, chunk=20000):
        """
        Yield cutoff pairs and their blockers per start chunk.

        :return: Generator of ``(start_ids, gly_ids, rows, dists)``.
        :rtype: collections.abc.Iterator[tuple]
        """
        no_rows = np.empty((0, 3), dtype=np.intp)
        no_dist = np.empty(0, dtype=np.float64)
        for s0 in range(0, len(start_idx), start_chunk):
            sub = start_idx[s0 : s0 + start_chunk]
            i, g, dist, P, _ = self._cutoff_pairs(sub, ctx=ctx)
            if len(i) == 0:
                continue
            rows, dists = no_rows, no_dist
            if ctx["btree"] is not None:
                out, dout = self._blocking_hits(sub, i, g, dist, P, ctx, chunk)
                if out:
                    rows, dists = np.concatenate(out), np.concatenate(dout)
            yield sub[i].astype(np.int64), g.astype(np.int64), rows, dists

    @staticmethod
    def _sort_blockers(row_parts, dist_parts):
        """
        Concatenate blockers and lexsort them by (start, blocker, glycan).

        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        if not row_parts:
            return np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.float64)
        res, dres = np.concatenate(row_parts), np.concatenate(dist_parts)
        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))
        return res[order], dres[order]

    def get_blockers(
        self,
        start_idx=None,
        mode=None,
        blocking=None,
        exclude_own_residue=None,
        exclude_start_overlap=True,
        *,
        frame_index=None,
    ):
        """
        Blocking atoms of every (start atom, glycan atom) cutoff pair of one
        analyzed frame.

        Surface residues and coordinates always come from the same frame:
        the trajectory is moved to ``frame_index`` for the computation and
        restored afterwards.

        :param start_idx: Indices into ``start_atoms``; None uses the surface
            start atoms of the frame.
        :type start_idx: numpy.ndarray or None
        :param mode: None follows the run.
        :type mode: str or None
        :param blocking: ``'vdw'`` or ``'cylinder'``; None follows the run.
        :type blocking: str or None
        :param exclude_own_residue: None means True, except False in v2.
        :type exclude_own_residue: bool or None
        :param bool exclude_start_overlap: vdW only.
        :param frame_index: Analyzed-frame index (negative counts from the
            end); None means the last analyzed frame.
        :type frame_index: int or None
        :return: ``(blockers, distances)``: (n, 3) rows ``[start idx,
            blocker idx, glycan idx]`` and float64 pair distances.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises RuntimeError: Before :meth:`run`.
        :raises ValueError: For the box test (no blocking atoms), or an
            invalid mode / blocking combination.
        :raises IndexError: ``frame_index`` out of range (library use).
        """
        k, _ = self._output_frame(frame_index)
        mode, blocking = self._resolve_test(mode, blocking)
        self._require_blocking_atoms(blocking)

        # default options: the run cache of _frame_result is valid
        if start_idx is None and exclude_own_residue is None and exclude_start_overlap:
            *_, blockers, dists = self._frame_result(
                k, mode, blocking, need_blockers=True
            )
            return blockers, dists

        with self._at_analyzed_frame(k):
            if start_idx is None:
                start_idx = self.surface_start_indices(k)
            *_, blockers, dists = self._visibility(
                start_idx,
                mode,
                blocking,
                exclude_own_residue=exclude_own_residue,
                exclude_start_overlap=exclude_start_overlap,
            )
        return blockers, dists

    @staticmethod
    def _require_blocking_atoms(blocking):
        """:raises ValueError: For the box test, which identifies no blocking atoms."""
        if blocking == "box":
            raise ValueError(
                "the v1 box test does not identify blocking atoms; "
                "pass mode='default' or mode='v2'"
            )

    @staticmethod
    def _residue_keys(atom_group):
        """
        (resname, chain, resid+icode) per atom, as the legacy scripts key residues.

        :rtype: list[tuple[str, str, str]]
        """
        try:
            chains = [str(c).strip() for c in atom_group.chainIDs]
        except (AttributeError, mda.exceptions.NoDataError):
            chains = [""] * len(atom_group)
        return [
            (str(resname).strip(), chain_id, str(label))
            for resname, chain_id, label in zip(
                atom_group.resnames, chains, _resid_labels(atom_group)
            )
        ]

    # endregion

    # -------------------------------------------------------------
    # region Visibility
    # -------------------------------------------------------------

    def _visibility(self, start_idx, mode, blocking, **ctx_kwargs):
        """
        Residue-level visibility for the current trajectory frame.

        :param numpy.ndarray start_idx: Indices into ``start_atoms``.
        :param str mode: Resolved counting mode.
        :param str blocking: Resolved blocking test.
        :param ctx_kwargs: exclude_own_residue / exclude_start_overlap for
        :meth:`_blocking_context` (vdW / cylinder only).
        :return: ``(cand_keys, vis_keys, n_blocked, mean_blocked_distance,
            blockers, distances)``; keys are ``residue_row * n_gly + glycan``;
            blockers / distances are None for the box test.
        :rtype: tuple
        """
        if blocking == "box":
            cand, vis, n_blocked, mean_d = self._box_visibility(start_idx)
            return cand, vis, n_blocked, mean_d, None, None

        start_idx = np.asarray(start_idx, dtype=np.intp)
        n_gly = len(self.gly_atoms)
        no_keys = np.empty(0, dtype=np.int64)
        if len(start_idx) == 0 or n_gly == 0:
            blockers, dists = self._sort_blockers([], [])
            return no_keys, no_keys, 0, np.nan, blockers, dists

        ctx = self._blocking_context(mode, blocking, **ctx_kwargs)
        cand_parts, vis_parts, row_parts, dist_parts = [], [], [], []
        for start_ids, gly_ids, rows, d in self._iter_blocked_chunks(start_idx, ctx):
            pair_key = start_ids * n_gly + gly_ids
            res_rows = self._start_res_row[start_ids].astype(np.int64)
            cand_parts.append(np.unique(res_rows * n_gly + gly_ids))
            if len(rows):
                blocked = np.unique(
                    rows[:, 0].astype(np.int64) * n_gly + rows[:, 2].astype(np.int64)
                )
                keep = ~np.isin(pair_key, blocked)
                res_rows, vis_gly = res_rows[keep], gly_ids[keep]
                row_parts.append(rows)
                dist_parts.append(d)
            else:
                vis_gly = gly_ids
            vis_parts.append(np.unique(res_rows * n_gly + vis_gly))

        cand = np.unique(np.concatenate(cand_parts)) if cand_parts else no_keys
        vis = np.unique(np.concatenate(vis_parts)) if vis_parts else no_keys
        blockers, dists = self._sort_blockers(row_parts, dist_parts)
        n_blocked, mean_d = self._blocked_stats(blockers, dists)
        return cand, vis, n_blocked, mean_d, blockers, dists

    def _blocked_stats(self, blockers, dists):
        """
        Unique blocked (start, glycan) pairs and their mean distance.

        :rtype: tuple[int, float]
        """
        if len(blockers) == 0:
            return 0, np.nan
        n_gly = len(self.gly_atoms)
        key = blockers[:, 0].astype(np.int64) * n_gly + blockers[:, 2].astype(np.int64)
        _, first = np.unique(key, return_index=True)
        return len(first), float(dists[first].mean())

    def _box_visibility(self, start_idx=None, workers=None, starts_per_task=32):
        """
        Original GLYCO v1 box test (cutoff ``int(distance_cutoff)``, strict '<').

        :param start_idx: Indices into ``start_atoms``; None uses the surface
            start atoms of the current frame.
        :type start_idx: numpy.ndarray or None
        :param workers: Worker processes; None uses ``n_workers``.
        :type workers: int or None
        :param int starts_per_task: Start atoms per worker task.
        :return: ``(cand_keys, vis_keys, n_blocked, mean_blocked_distance)``
        :rtype: tuple[numpy.ndarray, numpy.ndarray, int, float]
        """
        if start_idx is None:
            start_idx = self.surface_start_indices()
        empty = np.empty(0, dtype=np.int64)
        n_gly = len(self.gly_atoms)
        if len(start_idx) == 0 or n_gly == 0:
            return empty, empty, 0, np.nan

        if self._start_residue_keys is None:
            self._start_residue_keys = self._residue_keys(self.start_atoms)
            self._blocker_residue_keys = self._residue_keys(self.blocker_atoms)

        start_idx = np.asarray(start_idx, dtype=np.int64)
        blocker_positions = _legacy_coords(self.blocker_atoms.positions)
        initargs = (
            _legacy_coords(self.start_atoms.positions[start_idx]),
            np.asarray(self._start_res_row[start_idx], dtype=np.int64),
            [self._start_residue_keys[i] for i in start_idx],
            _legacy_coords(self.gly_atoms.positions),
            blocker_positions,
            self._blocker_residue_keys,
            _build_x_buckets(blocker_positions),  # rebuilt for THIS frame
            float(int(self.distance_cutoff)),  # v1: int(cutoff), strict '<'
            n_gly,
        )

        n_starts = len(start_idx)
        ranges = [
            (lo, min(lo + starts_per_task, n_starts))
            for lo in range(0, n_starts, starts_per_task)
        ]
        if workers is None:
            workers = self._n_workers(processes=True)
        workers = max(1, min(int(workers), len(ranges)))

        if workers == 1:
            _init_box_worker(*initargs)
            try:
                results = [_box_chunk(r) for r in ranges]
            finally:
                _BOX_STATE.clear()
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_box_worker,
                initargs=initargs,
            ) as executor:
                results = list(executor.map(_box_chunk, ranges))

        cand = [r[0] for r in results if len(r[0])]
        vis = [r[1] for r in results if len(r[1])]
        cand_keys = np.unique(np.concatenate(cand)) if cand else empty
        vis_keys = np.unique(np.concatenate(vis)) if vis else empty
        n_blocked = sum(int(r[2]) for r in results)
        dist_sum = sum(float(r[3]) for r in results)
        mean_d = dist_sum / n_blocked if n_blocked else np.nan
        return cand_keys, vis_keys, n_blocked, mean_d

    def _counts_from_visible_keys(self, vis_keys):
        """
        Per-CA counts from visible ``residue_row * n_gly + glycan`` keys.

        :rtype: numpy.ndarray of int
        """
        counts = np.zeros(len(self.prot_atoms), dtype=np.int64)
        if len(vis_keys):
            n_gly = len(self.gly_atoms)
            per_res = np.bincount(vis_keys // n_gly, minlength=self._n_res_rows)
            counts = per_res[self._prot_atom_res_row].astype(np.int64)
        return counts

    # endregion

    # -------------------------------------------------------------
    # region Frame Access
    # -------------------------------------------------------------

    @contextlib.contextmanager
    def _at_analyzed_frame(self, k):
        """Temporarily move the trajectory to analyzed frame ``k``."""
        old = int(self.u.trajectory.frame)
        try:
            self.u.trajectory[int(self._analyzed_frames()[k])]  # pylint: disable=expression-not-assigned

            yield
        finally:
            self.u.trajectory[old]  # pylint: disable=pointless-statement

    def _analyzed_frames(self):
        """
        Trajectory frame index of each analyzed frame.

        :rtype: numpy.ndarray of int
        """
        frames = getattr(self, "frames", None)
        if frames is not None and len(frames) == self.n_frames:
            return np.asarray(frames, dtype=int)
        return np.arange(self.n_frames)

    def _resolve_frame_index(self, frame_index=None, n=None):
        """
        Normalize an index into the analyzed frames.

        :param frame_index: Index (negative counts from the end); None means
            the last analyzed frame.
        :type frame_index: int or None
        :param n: Number of analyzed frames; None uses ``n_frames``.
        :type n: int or None
        :rtype: int
        :raises RuntimeError: Before :meth:`run`.
        :raises ValueError: No frames or a non-integer index (library use).
        :raises IndexError: Out of range (library use).
        """
        if n is None:
            n = getattr(self, "n_frames", None)
            if n is None:
                raise RuntimeError("frame indices are only defined after run() started")
        n = int(n)
        if n <= 0:
            self._error("no frames to analyze (check --frames/--start/--stop/--step)")
        if frame_index is None:
            return n - 1
        if isinstance(frame_index, bool):
            self._error(
                f"frame index (--bfactor-frame) must be an integer, got {frame_index!r}"
            )
        try:
            k = operator.index(frame_index)
        except TypeError:
            self._error(
                f"frame index (--bfactor-frame) must be an integer, got {frame_index!r}"
            )
        if not -n <= k < n:
            self._error(
                f"frame index (--bfactor-frame) {k} is out of range for {n} "
                f"analyzed frame(s); valid values are {-n} to {n - 1}",
                exc=IndexError,
            )
        return k + n if k < 0 else k

    def _frame_result(self, k, mode=None, blocking=None, need_blockers=False):
        """
        ``(start_idx, cand_keys, vis_keys, blockers, distances)`` of analyzed
        frame ``k``, from the run cache when possible.

        :param int k: Resolved analyzed-frame index.
        :param mode: None follows the run.
        :type mode: str or None
        :param blocking: None follows the run.
        :type blocking: str or None
        :param bool need_blockers: The caller needs per-atom blockers; raises
            for the box test.
        :return: ``(start_idx, cand_keys, vis_keys, blockers, distances)``
        :rtype: tuple
        :raises ValueError: On an unknown mode or a blocking test the mode does
            not allow or if ``need_blockers`` is True with box blocking test.
        """
        mode, blocking = self._resolve_test(mode, blocking)
        if need_blockers:
            self._require_blocking_atoms(blocking)

        cache = self._frame_cache
        if cache is not None and cache[:3] == (k, mode, blocking):
            return cache[3:]
        with self._at_analyzed_frame(k):
            start_idx = self.surface_start_indices(k)
            cand, vis, _, _, blockers, dists = self._visibility(
                start_idx, mode, blocking
            )
        return start_idx, cand, vis, blockers, dists

    # endregion

    # ---------------------------------------------------------
    # region Output Helpers
    # ---------------------------------------------------------

    def residue_identity(self):
        """
        (chain, resid+icode, resname) of every ``prot_atoms`` column, exactly as
        written in the CSV outputs.

        :rtype: list[tuple[str, str, str]]
        """
        _, (chains, resids, resnames) = self._identity_columns(
            self.prot_atoms, atoms=False
        )
        return [(str(c), str(r), str(n)) for c, r, n in zip(chains, resids, resnames)]

    def _compute_residue_histograms(self):
        """
        Histograms of the per-frame glycan counts, computed from ``all_counts``.

        :return: ``(values, hist)``; see :meth:`residue_histograms`.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        counts = np.asarray(self.all_counts, dtype=np.int64)  # (n_frames, n_res)
        _, n_res = counts.shape
        values = np.arange(int(counts.max(initial=0)) + 1)
        hist = np.zeros((n_res, len(values)), dtype=np.int64)
        rows = np.broadcast_to(np.arange(n_res), counts.shape)
        np.add.at(hist, (rows.ravel(), counts.ravel()), 1)
        # read-only: the stored arrays are shared by every caller
        values.flags.writeable = False
        hist.flags.writeable = False
        return values, hist

    def residue_histograms(self):
        """
        Histogram of every residue's glycan count over the analyzed frames.

        Computed once at the end of :meth:`run` and stored in
        ``results.count_values`` / ``results.residue_histograms``. Counts are
        integers, so there is one bin per count value.

        :return: ``(values, hist)``: ``values`` are the count values
            ``0 .. max``; ``hist[i, v]`` is the number of frames in which
            residue ``i`` (``prot_atoms`` order) had count ``values[v]``.
            Both arrays are read-only.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises RuntimeError: Before :meth:`run` has finished.
        """
        self._require_run()
        try:
            return self.results.count_values, self.results.residue_histograms
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__}.residue_histograms() is available "
                "after run() has finished"
            ) from None

    def residue_statistics(self, confidence=0.95, method="percentile"):
        """
        Per-residue summary of the glycan counts over the analyzed frames.

        :param float confidence: Interval coverage, between 0 and 1.
        :param str method: ``'percentile'``: central interval of the per-frame
            counts (how much coverage varies over the trajectory);
            ``'mean'``: t-based confidence interval of the mean count.
        :return: Arrays in ``prot_atoms`` order: ``mode`` (histogram maximum;
            ties go to the lower count), ``mean``, ``std``, ``max``,
            ``lower``, ``upper``, ``width`` and ``n_frames``.
        :rtype: dict[str, numpy.ndarray]
        :raises ValueError: On an invalid ``confidence`` or ``method``.
        :raises RuntimeError: Before :meth:`run`.
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        values, hist = self.residue_histograms()
        counts = np.asarray(self.all_counts, dtype=float)
        n = counts.shape[0]
        mean = counts.mean(axis=0)
        std = counts.std(axis=0, ddof=1) if n > 1 else np.zeros(counts.shape[1])
        alpha = (1.0 - confidence) / 2.0

        if method == "percentile":
            lower, upper = np.percentile(
                counts, [100 * alpha, 100 * (1 - alpha)], axis=0
            )
        elif method == "mean":
            if n > 1:
                half = t_res.ppf(1 - alpha, n - 1) * std / np.sqrt(n)
            else:
                half = np.zeros_like(mean)
            lower, upper = mean - half, mean + half
        else:
            raise ValueError(f"method must be 'percentile' or 'mean', got {method!r}")

        return {
            "mode": values[np.argmax(hist, axis=1)].astype(float),
            "mean": mean,
            "std": std,
            "max": counts.max(axis=0),
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "n_frames": np.full(counts.shape[1], n),
        }

    def write_histograms_csv(self, output=None):
        """
        residue_histograms.csv: one row per (residue, count value) that occurs.

        Columns: chain, resid, resname, glycan_count, n_frames, fraction.

        :param output: Default ``<outdir>/residue_histograms.csv``.
        :type output: str or pathlib.Path or None
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`.
        """
        values, hist = self.residue_histograms()
        names, cols = self._identity_columns(self.prot_atoms, atoms=False)
        n_frames = self.all_counts.shape[0]

        def batches():
            for i in range(hist.shape[0]):
                nz = np.flatnonzero(hist[i])
                ident = [c[i] for c in cols]
                yield (
                    [
                        *ident,
                        int(values[v]),
                        int(hist[i, v]),
                        _fmt_float(hist[i, v] / n_frames),
                    ]
                    for v in nz
                )

        return self._write_csv(
            output,
            "residue_histograms.csv",
            [*names, "glycan_count", "n_frames", "fraction"],
            batches(),
            "Residue histograms",
        )

    def write_statistics_csv(self, output=None, confidence=0.95, method="percentile"):
        """
        residue_statistics.csv: mode, mean, std, max and interval per residue.

        :param output: Default ``<outdir>/residue_statistics.csv``.
        :type output: str or pathlib.Path or None
        :param float confidence: See :meth:`residue_statistics`.
        :param str method: See :meth:`residue_statistics`.
        :rtype: pathlib.Path
        """
        stats = self.residue_statistics(confidence, method)
        names, cols = self._identity_columns(self.prot_atoms, atoms=False)
        keys = ("mode", "mean", "std", "max", "lower", "upper", "width")
        rows = (
            [
                *(c[i] for c in cols),
                int(stats["n_frames"][i]),
                *(_fmt_float(stats[k][i]) for k in keys),
            ]
            for i in range(len(self.prot_atoms))
        )
        return self._write_csv(
            output,
            "residue_statistics.csv",
            [*names, "n_frames", *keys],
            [rows],
            "Residue statistics",
        )

    def _ensure_occupancies(self):
        """Add an ``occupancies`` topology attribute if the Universe lacks one."""
        try:
            self.u.atoms.occupancies  # noqa: B018
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr("occupancies")

    @staticmethod
    def _strip_placeholder_cryst1(path, dims):
        """Remove the 1 A CRYST1 MDAnalysis writes for a Universe without a box."""
        if dims is None or np.allclose(np.asarray(dims[:3]), 1.0):
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text(
                "".join(line for line in lines if not line.startswith("CRYST1")),
                encoding="utf-8",
            )

    def write_statistics_pdb(
        self, frame_index=None, output=None, confidence=0.95, method="percentile"
    ):
        """
        b_factors_stats.pdb: trajectory statistics in the B-factor and occupancy
        columns.

        B-factor: the histogram maximum (mode) of each residue's glycan count
        over all analyzed frames. Occupancy: the width of the ``confidence``
        interval (see :meth:`residue_statistics`). Every atom of a residue in
        ``prot_atoms`` gets that residue's values; all other atoms get 0.
        Coordinates come from ``frame_index``; the trajectory is restored
        afterwards. The original tempfactors and occupancies of the Universe
        are restored as well.

        :param frame_index: Analyzed frame for the coordinates; None means the last.
        :type frame_index: int or None
        :param output: Default ``<outdir>/b_factors_stats.pdb``.
        :type output: str or pathlib.Path or None
        :param float confidence: See :meth:`residue_statistics`.
        :param str method: See :meth:`residue_statistics`.
        :rtype: pathlib.Path
        """
        k, _ = self._output_frame(frame_index)
        stats = self.residue_statistics(confidence, method)
        output = self._output_path(output, "b_factors_stats.pdb")
        self._ensure_tempfactors()
        self._ensure_occupancies()

        for name, values in (
            ("B-factor", stats["mode"]),
            ("occupancy", stats["width"]),
        ):
            if values.size and values.max() > 999.99:
                warnings.warn(
                    f"max {name} value {values.max():g} exceeds the PDB "
                    "column width (999.99); see residue_statistics.csv",
                    stacklevel=2,
                )

        atoms, cols = self._residue_atom_rows()
        old_bf = self.u.atoms.tempfactors.copy()
        old_occ = self.u.atoms.occupancies.copy()
        try:
            self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
            self.u.atoms.occupancies = np.zeros(self.u.atoms.n_atoms)
            atoms.tempfactors = stats["mode"][cols]
            atoms.occupancies = stats["width"][cols]
        finally:
            self.u.atoms.tempfactors = old_bf
            self.u.atoms.occupancies = old_occ

        with self._at_analyzed_frame(k):
            self.u.atoms.write(str(output))
            dims = self.u.dimensions
        self._strip_placeholder_cryst1(output, dims)

        written("Trajectory statistics PDB", output, base=self.outdir)
        return output

    def write_chimerax_script(
        self,
        structure=None,
        output=None,
        palette="white:red",
        worm_radii=(0.25, 2.0),
        background_color="lightgray",
    ):
        """
        ChimeraX command script (.cxc) for the trajectory statistics.

        Opens ``structure`` (default: :meth:`write_statistics_pdb` output),
        colors the protein by B-factor (histogram maximum) from white at 0 to
        red at the largest value, and shows a worm cartoon whose radius follows
        the occupancy (interval width). Atoms outside the protein keep
        ``background_color``.

        :param structure: Statistics PDB; None writes it first.
        :type structure: str or pathlib.Path or None
        :param output: Default ``<outdir>/glyco_stats.cxc``.
        :type output: str or pathlib.Path or None
        :param str palette: ChimeraX palette spread over ``0 .. maximum``;
            default ``'white:red'``.
        :param tuple worm_radii: Worm radius (Angstrom) at the smallest and
            largest interval width.
        :param str background_color: Color of atoms not in the protein.
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`.
        """
        self._require_run()
        if structure is None:
            structure = self.write_statistics_pdb()
        structure = Path(structure).resolve()
        if not structure.is_file():
            self._error(
                f"ChimeraX script: structure not found: {structure}",
                exc=FileNotFoundError,
            )

        stats = self.residue_statistics()
        bmax = float(stats["mode"].max()) if stats["mode"].size else 0.0
        bmax = max(bmax, 1.0)  # avoid an empty range when all counts are 0
        wmax = float(stats["width"].max()) if stats["width"].size else 0.0
        if wmax == 0:
            warnings.warn(
                "all interval widths are 0 (e.g. a single analyzed frame); "
                "the worm has a constant radius",
                stacklevel=2,
            )

        spec = "(#1 & protein)"
        path = structure.as_posix().replace('"', '\\"')
        r0, r1 = worm_radii
        lines = [
            "# Auto-generated by glyco (GLYCO): trajectory statistics",
            f"# B-factor  = histogram maximum, colored white (0) -> red ({bmax:g})",
            "# occupancy = width of the 95% interval of the glycan count",
            "close session",
            f'open "{path}"',
            "hide atoms",
            "show cartoons",
            f"color #1 {background_color}",
            f"color byattribute bfactor {spec} palette {palette} "  # noqa: ISC004
            f"range 0,{bmax:g} noValueColor {background_color}",
            f"cartoon byattribute occupancy {spec} min:{r0:g} max:{r1:g}",
            "view",
        ]
        output = self._output_path(output, "glyco_stats.cxc")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written("ChimeraX script", output, base=self.outdir)
        return output

    def plot_histograms(self, residues=None, output=None, max_plots=12):
        """
        Bar plots of the glycan-count histograms (requires matplotlib).

        :param residues: Indices into ``prot_atoms``; None plots the residues
            with the widest intervals.
        :param output: Default ``<outdir>/residue_histograms.png``.
        :param int max_plots: Maximum number of residues plotted.
        :rtype: pathlib.Path
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        values, hist = self.residue_histograms()
        stats = self.residue_statistics()
        if residues is None:
            residues = np.argsort(stats["width"])[::-1][:max_plots]
        _, cols = self._identity_columns(self.prot_atoms, atoms=False)
        n = len(residues)
        fig, axes = plt.subplots(
            (n + 3) // 4,
            min(n, 4),
            squeeze=False,
            figsize=(4 * min(n, 4), 3 * ((n + 3) // 4)),
        )
        for ax, i in zip(axes.ravel(), residues):
            ax.bar(values, hist[i], width=0.9)
            ax.axvline(stats["mode"][i], color="k", lw=1)
            ax.axvspan(stats["lower"][i], stats["upper"][i], alpha=0.2)
            ax.set_title(f"{cols[0][i]}:{cols[2][i]}{cols[1][i]}")
            ax.set_xlabel("glycan count")
            ax.set_ylabel("frames")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        output = self._output_path(output, "residue_histograms.png")
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output

    def _diag(self, *parts):
        """Dimmed diagnostic output, only with -v/--verbose."""
        if self._verbose:
            diagnostic(*parts)

    def _residue_atom_rows(self):
        """
        Atoms of the CA residues and the ``all_counts`` column of each.

        Columns follow ``prot_atoms`` order (one CA per residue), so every
        atom gets the count of its own residue. Atoms are sorted by index.

        :return: ``(atoms, columns)``
        :rtype: tuple[MDAnalysis.core.groups.AtomGroup, numpy.ndarray]
        """
        atoms = self.prot_atoms.residues.atoms
        cols = self._rows_for(self.prot_atoms.resindices, atoms.resindices)
        order = np.argsort(atoms.indices, kind="stable")
        return atoms[order], cols[order]

    @staticmethod
    def _rows_for(residue_resindices, atom_resindices):
        """Row (position in ``residue_resindices``) of each atom's residue."""
        residue_resindices = np.asarray(residue_resindices)
        order = np.argsort(residue_resindices)
        pos = np.searchsorted(residue_resindices, atom_resindices, sorter=order)
        return order[pos]

    @staticmethod
    def _atom_attr(atom_group, attribute, default=""):
        """
        Whole-group attribute as a stripped string array (``default`` if missing).

        :rtype: numpy.ndarray (dtype object)
        """
        try:
            values = getattr(atom_group, attribute)
        except (AttributeError, mda.exceptions.NoDataError):
            return np.full(len(atom_group), default, dtype=object)
        return np.asarray([str(v).strip() for v in values], dtype=object)

    def _identity_columns(self, atom_group, prefix="", atoms=True):
        """
        Identity columns in the fixed order chain, resid, resname
        [, atom_name, atom_serial].

        :param atom_group: Atoms whose identity is written.
        :param str prefix: Column prefix ('' for the residue itself).
        :param bool atoms: Include atom name and PDB serial.
        :return: ``(names, columns)``
        :rtype: tuple[list[str], list[numpy.ndarray]]
        """
        p = f"{prefix}_" if prefix else ""
        names = [f"{p}chain", f"{p}resid", f"{p}resname"]
        cols = [
            self._atom_attr(atom_group, "chainIDs"),
            _resid_labels(atom_group),
            self._atom_attr(atom_group, "resnames"),
        ]
        if atoms:
            names += [f"{p}atom_name", f"{p}atom_serial"]
            cols += [self._atom_attr(atom_group, "names"), pdb_atom_ids(atom_group)]
        return names, cols

    def _output_path(self, output, default_name):
        """
        Resolve a writer's output path and create its directory.

        :rtype: pathlib.Path
        """
        path = Path(output) if output is not None else Path(self.outdir) / default_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _ensure_tempfactors(self):
        """Add a ``tempfactors`` topology attribute if the Universe lacks one."""
        try:
            self.u.atoms.tempfactors  # noqa: B018
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr("tempfactors")

    # endregion

    # --------------------------------------------------------------
    # region Writers
    # --------------------------------------------------------------

    def _output_frame(self, frame_index):
        """
        Resolve the analyzed frame an output is written for.

        :param frame_index: Analyzed-frame index (negative counts from the
            end); None means the last analyzed frame.
        :type frame_index: int or None
        :return: ``(k, trajectory_frame)``
        :rtype: tuple[int, int]
        :raises RuntimeError: Before :meth:`run`.
        """
        self._require_run()
        k = self._resolve_frame_index(frame_index)
        return k, int(self._analyzed_frames()[k])

    def _write_csv(self, output, default_name, header, batches, label):
        """
        Write a CSV file with ``header`` and the rows of every batch.

        Batches are written one at a time, so large outputs never have to be
        held in memory as a whole.

        :param output: Explicit path, or None for ``<outdir>/<default_name>``.
        :type output: str or pathlib.Path or None
        :param str default_name: File name inside ``outdir``.
        :param list header: Column names.
        :param batches: Iterable of row iterables.
        :param str label: Name used in the console message.
        :return: Path of the written file.
        :rtype: pathlib.Path
        """
        output = self._output_path(output, default_name)
        n = 0
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for batch in batches:
                batch = list(batch)
                writer.writerows(batch)
                n += len(batch)
        written(label, output, n, base=self.outdir)
        return output

    def write_blockers_csv(
        self, frame_index=None, output=None, chunk=100_000, mode=None, blocking=None
    ):
        """
        blockers.csv: one row per (start atom, blocking atom, glycan atom) of
        one frame.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: Default ``<outdir>/blockers.csv``.
        :type output: str or pathlib.Path or None
        :param int chunk: Rows written per batch.
        :param mode: None follows the run.
        :type mode: str or None
        :param blocking: None follows the run.
        :type blocking: str or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`.
        :raises ValueError: For the box test (no blocking atoms).
        """
        k, tf = self._output_frame(frame_index)
        *_, blockers, distances = self._frame_result(
            k, mode, blocking, need_blockers=True
        )

        s_names, s_cols = self._identity_columns(self.start_atoms, "start")
        b_names, b_cols = self._identity_columns(self.blocker_atoms, "blocker")
        g_names, g_cols = self._identity_columns(self.gly_atoms, "glycan")
        header = [
            "analyzed_frame",
            "trajectory_frame",
            *s_names,
            *b_names,
            *g_names,
            "pair_distance",
        ]

        def batches():
            s_i, b_i, g_i = blockers.T
            for lo in range(0, len(blockers), chunk):
                hi = min(lo + chunk, len(blockers))
                s, b, g = s_i[lo:hi], b_i[lo:hi], g_i[lo:hi]
                yield zip(
                    repeat(k),
                    repeat(tf),
                    *(c[s].tolist() for c in s_cols),
                    *(c[b].tolist() for c in b_cols),
                    *(c[g].tolist() for c in g_cols),
                    [_fmt_float(x) for x in distances[lo:hi]],
                )

        return self._write_csv(output, "blockers.csv", header, batches(), "Blockers")

    def write_bfactors(self, frame_index=None, output=None):
        """
        Write b_factors.pdb (one frame) and b_factors.npy (all frames) next to it.

        Every atom of a residue in ``prot_atoms`` (all atoms, including
        hydrogens) gets that residue's glycan count as its B-factor; all other
        atoms get 0. The trajectory is restored to its previous frame
        afterwards; the values remain set as ``tempfactors`` on the Universe.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: PDB path; default ``<outdir>/b_factors.pdb``.
        :type output: str or pathlib.Path or None
        :rtype: pathlib.Path
        """
        k, _ = self._output_frame(frame_index)
        output = self._output_path(output, "b_factors.pdb")
        self._ensure_tempfactors()

        values = self.all_counts[k]
        if values.size and values.max() > 999.99:
            warnings.warn(
                f"max B-factor value {values.max():g} exceeds the PDB "
                "B-factor column width (999.99); use b_factors.npy for "
                "the raw values",
                stacklevel=2,
            )
        atoms, cols = self._residue_atom_rows()
        self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
        atoms.tempfactors = values[cols]

        with self._at_analyzed_frame(k):
            self.u.atoms.write(str(output))
            dims = self.u.dimensions
        self._strip_placeholder_cryst1(output, dims)

        np.save(output.with_suffix(".npy"), self.all_counts)
        written("B-factors", output, base=self.outdir)
        return output

    def write_frame_summary(self, output=None):
        """
        frame_summary.csv: one row per analyzed frame.

        :param output: Default ``<outdir>/frame_summary.csv``.
        :type output: str or pathlib.Path or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`.
        """
        self._require_run()
        r = self.results
        rows = (
            [
                k,
                int(tf),
                int(r.n_surface_residues[k]),
                int(r.n_blocked_pairs[k]),
                _fmt_float(r.mean_blocked_distance[k]),
            ]
            for k, tf in enumerate(self._analyzed_frames())
        )
        return self._write_csv(
            output,
            "frame_summary.csv",
            [
                "analyzed_frame",
                "trajectory_frame",
                "n_surface_residues",
                "n_blocked_pairs",
                "mean_blocked_distance",
            ],
            [rows],
            "Frame summary",
        )

    def write_residue_csv(self, frame_index=None, output=None):
        """
        residue_summary.csv: glycan count and SASA of every residue with a
        nonzero count.

        Columns: analyzed_frame, trajectory_frame, chain, resid (with
        insertion code), resname, glycan_count, sasa.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: Default ``<outdir>/residue_summary.csv``.
        :type output: str or pathlib.Path or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`.
        """
        k, tf = self._output_frame(frame_index)
        names, cols = self._identity_columns(self.prot_atoms, atoms=False)
        counts, sasa = self.all_counts[k], self.results.sasa[k]
        keep = np.flatnonzero(counts > 0)
        rows = zip(
            repeat(k),
            repeat(tf),
            *(c[keep].tolist() for c in cols),
            counts[keep].tolist(),
            [_fmt_float(x) for x in sasa[keep]],
        )
        return self._write_csv(
            output,
            "residue_summary.csv",
            ["analyzed_frame", "trajectory_frame", *names, "glycan_count", "sasa"],
            [rows],
            "Residue summary",
        )

    def write_excluded_glycan_atoms_csv(
        self, frame_index=None, output=None, mode=None, blocking=None
    ):
        """
        Glycan atoms missing from each surface residue's visibility union.

        One row per excluded glycan atom, for every surface residue whose
        count in the run is nonzero. ``reason`` is
        ``outside_distance_cutoff`` (no start atom of the residue within the
        cutoff) or ``blocked_for_all_start_atoms``. With a ``mode`` /
        ``blocking`` override, ``glycan_count`` still comes from the run and
        a warning is issued per residue whose recomputed union differs.

        :param frame_index: Analyzed-frame index; None means the last.
        :type frame_index: int or None
        :param output: Default ``<outdir>/excluded_glycan_atoms.csv``.
        :type output: str or pathlib.Path or None
        :param mode: None follows the run.
        :type mode: str or None
        :param blocking: None follows the run.
        :type blocking: str or None
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`.
        """
        k, tf = self._output_frame(frame_index)
        start_idx, cand_keys, vis_keys, _, _ = self._frame_result(k, mode, blocking)
        n_gly = len(self.gly_atoms)
        empty = np.empty(0, dtype=np.int64)

        candidates = self._group_by_residue(cand_keys, n_gly)
        visible = self._group_by_residue(vis_keys, n_gly)
        r_names, r_cols = self._identity_columns(self.prot_atoms, atoms=False)
        g_names, g_cols = self._identity_columns(self.gly_atoms, "glycan")
        ca_of_row = {int(r): i for i, r in enumerate(self._prot_atom_res_row)}
        surface_rows = (
            np.unique(self._start_res_row[start_idx]) if len(start_idx) else empty
        )

        def batches():
            for row in surface_rows.tolist():
                ca = ca_of_row.get(row)
                if ca is None:
                    continue
                vis = visible.get(row, empty)
                is_visible = np.zeros(n_gly, dtype=bool)
                is_visible[vis] = True
                is_candidate = np.zeros(n_gly, dtype=bool)
                is_candidate[candidates.get(row, empty)] = True

                count = int(self.all_counts[k, ca])
                identity = [c[ca] for c in r_cols]
                if len(vis) != count:
                    warnings.warn(
                        f"Residue {identity[0]}:{identity[2]}{identity[1]}: "
                        f"union size {len(vis)} != reported count {count}",
                        stacklevel=3,
                    )
                if count == 0:
                    continue
                excluded = np.flatnonzero(~is_visible)
                if len(excluded) == 0:
                    continue

                reasons = np.where(
                    is_candidate[excluded],
                    "blocked_for_all_start_atoms",
                    "outside_distance_cutoff",
                )
                yield zip(
                    repeat(k),
                    repeat(tf),
                    *(repeat(v) for v in identity),
                    repeat(count),
                    *(c[excluded].tolist() for c in g_cols),
                    reasons.tolist(),
                )

        return self._write_csv(
            output,
            "excluded_glycan_atoms.csv",
            [
                "analyzed_frame",
                "trajectory_frame",
                *r_names,
                "glycan_count",
                *g_names,
                "reason",
            ],
            batches(),
            "Excluded glycan atoms",
        )

    @staticmethod
    def _group_by_residue(keys, n_gly):
        """Sorted unique keys -> ``{residue_row: glycan indices}``."""
        if len(keys) == 0:
            return {}
        rows = keys // n_gly
        uniq, first = np.unique(rows, return_index=True)
        return dict(zip(uniq.tolist(), np.split(keys % n_gly, first[1:])))

    def write_run_config(self, output=None):
        """
        run_config.json: mode, blocking test, effective selections, all
        parameters, analyzed frames and package versions.

        :param output: Default ``<outdir>/run_config.json``.
        :type output: str or pathlib.Path or None
        :rtype: pathlib.Path
        """
        self._require_run()
        output = self._output_path(output, "run_config.json")
        config = {
            "program": "glyco (GLYCO, GLYcan COverage)",
            "mode": self._mode,
            "blocking": self._blocking,
            "effective_selections": {
                "protein": self._validated_prot_sel,
                "glycan": self._validated_gly_sel,
                "blockers": self._validated_block_sel,
            },
            "parameters": {n: getattr(self, n) for n in self._param_names},
            "trajectory_frames": self._analyzed_frames().tolist(),
            "versions": _package_versions(),
        }
        output.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        written("Run configuration", output, base=self.outdir)
        return output

    def _write_progress(self, current, total):
        """Machine-readable progress JSON (``GLYCO_PROGRESS_FILE``), written
        atomically."""
        if not self.progress_file:
            return
        temp_file = self.progress_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "current": current,
                    "total": total,
                    "percent": int(100 * current / total) if total else 0,
                },
                f,
            )
        os.replace(temp_file, self.progress_file)

    @staticmethod
    def _tcl_quote(value):
        """Quote a string as one Tcl word."""
        value = str(value)
        if "{" not in value and "}" not in value and not value.endswith("\\"):
            return "{" + value + "}"
        return '"' + re.sub(r'([\\"$\[\]])', r"\\\1", value) + '"'

    @staticmethod
    def _index_ranges(sorted_indices):
        """Compress sorted ints into VMD ``'a to b'`` ranges."""
        if len(sorted_indices) == 0:
            return ""
        breaks = np.flatnonzero(np.diff(sorted_indices) != 1) + 1
        return " ".join(
            str(run[0]) if len(run) == 1 else f"{run[0]} to {run[-1]}"
            for run in np.split(sorted_indices, breaks)
        )

    def write_vmd_user_script(self, output=None, frame_offset=0):
        """
        VMD Tcl script assigning per-residue counts to ``user`` for every
        analyzed frame.

        Every atom of a CA residue gets that residue's count (the same
        mapping as b_factors.pdb).

        :param output: Default ``<outdir>/load_user_attributes.vmd``.
        :type output: str or pathlib.Path or None
        :param int frame_offset: Added to each trajectory frame index.
        :return: Path of the written file.
        :rtype: pathlib.Path
        :raises RuntimeError: Before :meth:`run`, or if the counts do not
            match ``prot_atoms``.
        :raises ValueError: Topology / trajectory filename unknown (library use).
        :raises FileNotFoundError: Missing topology / trajectory (library use).
        """
        self._require_run()
        u = self.u
        topology = getattr(u, "filename", None)
        if not topology:
            self._error("--write-vmd: could not determine the topology filename")
        topology = Path(topology).resolve()
        if not topology.is_file():
            self._error(
                f"--write-vmd: topology file not found: {topology}",
                exc=FileNotFoundError,
            )

        reader = u.trajectory
        trajectory_files = getattr(reader, "filenames", None)
        if trajectory_files is None:
            filename = getattr(reader, "filename", None)
            trajectory_files = [filename] if filename else []
        trajectory_files = [
            Path(p).resolve() for p in trajectory_files if p is not None
        ]
        if not trajectory_files:
            self._error("--write-vmd: could not determine the trajectory filename(s)")
        for path in trajectory_files:
            if not path.is_file():
                self._error(
                    f"--write-vmd: trajectory file not found: {path}",
                    exc=FileNotFoundError,
                )

        counts = self.all_counts  # (n_frames, n_CA)
        if counts.shape[1] != len(self.prot_atoms):
            raise RuntimeError(
                f"{counts.shape[1]} count columns but {len(self.prot_atoms)} CA atoms"
            )
        atoms, cols = self._residue_atom_rows()
        analyzed = self._analyzed_frames()
        data_max = float(counts.max()) if counts.size and counts.max() > 0 else 1.0

        lines = [
            "# Auto-generated by glyco (GLYCO)",
            "# Bulk per-frame assignment of residue values",
            "",
            f"set molid [mol new {self._tcl_quote(topology.as_posix())} waitfor all]",
            "",
        ]
        lines += [
            f"mol addfile {self._tcl_quote(t.as_posix())} waitfor all molid $molid"
            for t in trajectory_files
        ]
        lines += [
            "",
            "mol modselect 0 $molid protein",
            "mol modstyle 0 $molid NewCartoon",
            "mol modcolor 0 $molid User",
            "",
            f"set n_traj_frames {int(reader.n_frames)}",
            "set n_vmd_frames [molinfo $molid get numframes]",
            "",
            "if {$n_vmd_frames == $n_traj_frames + 1} {",
            "    animate delete beg 0 end 0 $molid",
            '    puts "Deleted duplicate first frame."',
            "} elseif {$n_vmd_frames == $n_traj_frames} {",
            '    puts "Frame counts match; no frame deleted."',
            "} else {",
            (
                '    error "Frame mismatch: VMD has $n_vmd_frames frames, but the '
                "trajectory has $n_traj_frames. Expected equal counts or exactly "
                'one extra VMD frame."'
            ),
            "}",
            "",
            "# Set User color scale",
            f"mol scaleminmax $molid 0 0 {data_max:.10g}",
            "",
            f"set sel [atomselect $molid {{index {self._index_ranges(atoms.indices)}}}]",
            "",
        ]
        for k in range(counts.shape[0]):
            vmd_frame = frame_offset + int(analyzed[k])
            values = " ".join(str(int(v)) for v in counts[k, cols])
            lines += [
                f"# Data frame {k} -> VMD frame {vmd_frame}",
                f"$sel frame {vmd_frame}",
                f"$sel set user {{{values}}}",
                "",
            ]
        lines += [
            "$sel delete",
            'puts "Finished assigning per-residue user values."',
            "",
        ]

        output = self._output_path(output, VMD_SCRIPT_NAME)
        output.write_text("\n".join(lines), encoding="utf-8")
        written("VMD script", output, base=self.outdir)
        return output

    # endregion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_job(args, top, traj, outdir, verbose, parser=None):
    """Load one structure, run the analysis and write its outputs."""
    try:
        u = load_universe(top, traj)
    except _LOAD_ERRORS as e:
        msg = (
            f"could not load {top}"
            + (
                " with -b/--traj (do all trajectory files "
                "match the topology's atom count?)"
                if traj
                else ""
            )
            + f": {e}"
        )
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg) from e
    GlycoAnalysis.from_args(
        u, args, parser=parser, top=top, traj=traj, indir=None, outdir=outdir
    ).run(verbose=verbose, progressbar_kwargs={"desc": "Running GLYCO Analysis"})


def _structure_outdirs(structures, base_outdir):
    """One output subfolder per structure, disambiguating clashing stems."""
    stems = [Path(p).stem for p in structures]
    return [
        os.path.join(
            base_outdir,
            stem if stems.count(stem) == 1 else Path(path).name.replace(".", "_"),
        )
        for path, stem in zip(structures, stems)
    ]


def main(argv=None):
    """
    Command-line entry point.

    :param argv: Arguments (default: ``sys.argv[1:]``).
    :type argv: list[str] or None
    :return: Exit status (0 on success, 1 on failure or declined overwrite).
    :rtype: int
    """
    parser = base_parser()
    args = parser.parse_args(argv)
    set_color_mode(args.color)
    validate_arguments(args, parser.error)  # before loading any files

    is_streamlit = os.environ.get("GLYCO_STREAMLIT") == "1"
    verbose = False if is_streamlit else args.verbose
    log_path = os.path.join(args.outdir, "mdanalysis_warnings.log")

    if args.indir is None:
        if not confirm_overwrite(args.outdir, args):
            return 1
        os.makedirs(args.outdir, exist_ok=True)
        with warning_log(log_path):
            _run_job(args, args.top, args.traj, args.outdir, verbose, parser=parser)
        return 0

    # --indir: the base directory only holds subfolders and the log
    os.makedirs(args.outdir, exist_ok=True)
    failures = []
    with warning_log(log_path):
        outdirs = _structure_outdirs(args.structures, args.outdir)
        for n, (path, outdir) in enumerate(zip(args.structures, outdirs), start=1):
            heading(f"[{n}/{len(args.structures)}] {path}")
            if not confirm_overwrite(outdir, args):
                failures.append((path, "output exists and overwrite declined"))
                continue
            try:
                _run_job(args, path, None, outdir, verbose)
            except _JOB_ERRORS as e:
                error(f"{path}: {e}")
                failures.append((path, str(e)))

    if failures:
        error(f"{len(failures)} of {len(args.structures)} structure(s) failed:")
        for path, reason in failures:
            print(f"    {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
