#!/usr/bin/env python3
# ruff: noqa: EXE001

import argparse
import glob
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from scipy.spatial import cKDTree

from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.parsers import base_parser

VMD_SCRIPT_NAME = "load_user_attributes.vmd"
DEFAULT_OUTDIR_NAME = "glyco_output"

# Hydrogen atom names: optional leading digits then 'H' (H1, HB2, 1HB, HO2).
# Anchored at the START so heavy atoms like NH1, OH, CH2 are not matched.
_HYDROGEN_NAME = re.compile(r"^\d*H")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def configure_warning_log(path):
    """
    Route Python warnings (incl. MDAnalysis') into a log file. Called from
    main() rather than at import time, so importing this module has no
    side effects.
    """
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    abspath = os.path.abspath(path)
    for existing in warnings_logger.handlers:
        if isinstance(existing, logging.FileHandler) and existing.baseFilename == abspath:
            return
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False


def heavy_atoms(ag):
    """
    Return `ag` without hydrogens. An atom is a hydrogen if its element is
    H/D (when the topology has elements) OR its name starts with H (after
    optional digits). Name matching is anchored at the start so that heavy
    atoms such as ARG NH1/NH2, TYR OH or TRP CH2 are kept.
    """
    names = np.asarray(ag.names, dtype=str)
    is_h = np.fromiter(
        (bool(_HYDROGEN_NAME.match(n)) for n in names),
        dtype=bool, count=len(names),
    )
    try:
        elements = np.char.upper(np.asarray(ag.elements, dtype=str))
        is_h |= np.isin(elements, ("H", "D"))
    except (AttributeError, mda.exceptions.NoDataError):
        pass
    return ag[~is_h]


def find_structures(indir, extensions):
    """
    Find independent structure files in `indir` matching `extensions`
    (a list of extensions without leading dots, e.g. ['pdb', 'gro']).
    Each match is treated as a standalone, single-file structure -- no
    trajectory pairing or chain reading involved.
    """
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(indir, f'*.{ext}'))))
    return list(dict.fromkeys(files))


def validate_arguments(cfg, error, check_inputs=True):
    """
    Validate (and normalize) parameters on `cfg`, which can be an argparse
    Namespace or a GlycoAnalysis instance. Needs no Universe, so the CLI can
    run it BEFORE loading any files and fail with a clean usage message.

    `error` is a callable that reports a message and does not return
    (parser.error for the CLI, or a function raising ValueError).
    With check_inputs=False the input-file checks are skipped (library use,
    where a Universe is passed in directly).

    Sets cfg.structures (list of files for --indir, else []) and normalizes
    cfg.frame_list to a list of ints or None.
    """
    top_given = cfg.top is not None
    traj_given = bool(cfg.traj)
    indir_given = cfg.indir is not None
    cfg.structures = []

    if check_inputs:
        if not top_given and not indir_given:
            error("one of -a/--top(+-b/--traj) or --indir must be given")
        if indir_given and (top_given or traj_given):
            error("--indir is mutually exclusive with -a/--top and -b/--traj")
        if traj_given and not top_given:
            error("-b/--traj requires -a/--top")

        if top_given:
            if not os.path.isfile(cfg.top):
                error(f"-a/--top: no such file: {cfg.top!r}")
            for t in (cfg.traj or []):
                if not os.path.isfile(t):
                    error(f"-b/--traj: no such file: {t!r}")
        else:
            if not os.path.isdir(cfg.indir):
                error(f"--indir: no such directory: {cfg.indir!r}")
            extensions = [e.strip().lstrip('.') for e in cfg.in_types.split(',')
                          if e.strip()]
            structures = find_structures(cfg.indir, extensions)
            if not structures:
                error(
                    f"--indir: no files with extension(s) {extensions} "
                    f"found in {cfg.indir!r}"
                )
            cfg.structures = structures

    slice_given = any(v is not None
                      for v in (cfg.frame_start, cfg.frame_stop, cfg.frame_step))
    frames_given = bool(cfg.frame_list)

    if frames_given and slice_given:
        error("--frames is not allowed with --start/--stop/--step")

    if indir_given and (frames_given or slice_given):
        error(
            "--frames/--start/--stop/--step are not meaningful with "
            "--indir (each structure is a single, independent frame)"
        )

    for name in ('distance_cutoff', 'area_cutoff', 'radius', 'probe'):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            error(f"--{name.replace('_', '-')} must be positive, got {value}")

    if cfg.frame_step is not None and cfg.frame_step == 0:
        error("--step cannot be 0")

    start, stop, step = cfg.frame_start, cfg.frame_stop, cfg.frame_step
    if (start is not None and stop is not None and start >= 0 and stop >= 0
            and (step is None or step > 0) and start >= stop):
        error(f"--start ({start}) must be less than --stop ({stop})")

    if frames_given:
        try:
            frame_ints = [int(f) for f in cfg.frame_list]
        except (TypeError, ValueError):
            error(f"--frames must be integers, got {cfg.frame_list!r}")
        if any(f < 0 for f in frame_ints):
            error("--frames must be non-negative")
        cfg.frame_list = frame_ints
    else:
        cfg.frame_list = None

    if not cfg.outdir:
        cfg.outdir = default_outdir(cfg)


def default_outdir(cfg):
    """Pick an output directory name from the inputs when -o isn't given."""
    if cfg.indir is not None:
        return f"glyco_{Path(cfg.indir).resolve().name}"
    if cfg.top is not None:
        return f"glyco_{Path(cfg.top).stem}"
    return DEFAULT_OUTDIR_NAME  # library use: Universe passed directly


def planned_outputs(outdir, write_vmd=False):
    """Paths of the files a run writes into `outdir`."""
    names = ['b_factors.pdb', 'b_factors.npy', 'frame_summary.csv']
    if write_vmd:
        names.append(VMD_SCRIPT_NAME)
    return [os.path.join(outdir, n) for n in names]


def confirm_overwrite(outdir, force=False, write_vmd=False):
    """
    If any of the files a run would write already exist in `outdir`, prompt
    the user to overwrite them unless force=True. Returns True if execution
    should proceed, False if the user declined (or no prompt is possible).

    The check is on the specific output files, not on the directory, so the
    default outdir of '.' does not prompt every time.
    """
    existing = [os.path.basename(p) for p in planned_outputs(outdir, write_vmd)
                if os.path.exists(p)]
    if not existing:
        return True

    listing = ", ".join(existing)
    if force:
        print(f"[info] overwriting in {outdir!r}: {listing} (--force given).")
        return True

    if not sys.stdin.isatty():
        print(
            f"[error] {outdir!r} already contains: {listing}; no TTY is "
            f"available to prompt, pass --force to overwrite "
            f"non-interactively.",
            file=sys.stderr,
        )
        return False

    while True:
        try:
            answer = input(
                f"{outdir!r} already contains: {listing}. Overwrite? [y/N] "
            ).strip().lower()
        except EOFError:
            return False
        if answer in ('y', 'yes'):
            return True
        if answer in ('', 'n', 'no'):
            return False
        print("Please answer 'y' or 'n'.")


def load_universe(top, traj=None):
    """
    Load a Universe. Several trajectories go through the MDAnalysis chain
    reader and must all match the topology's atom count.
    """
    if not traj:
        return mda.Universe(top)
    return mda.Universe(top, traj if len(traj) > 1 else traj[0])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

class GlycoAnalysis(AnalysisBase):
    """
    Run Glyco and other useful output functions.

    Semantics: only the line-of-sight STARTING POINTS (protein CA atoms) are
    restricted to surface residues. Blocker candidates are ALL protein heavy
    atoms (except those in the starting residue itself).
    """

    # Results are stored on the instance (not merged from workers) and the
    # nested FreeSASA run covers the whole slice, so this is serial-only.
    _analysis_algorithm_is_parallelizable = False

    @classmethod
    def get_supported_backends(cls):
        return ('serial',)

    def __init__(self, u, parser=None, verbose=False, args=None, **kwargs):
        super().__init__(u.trajectory, verbose=verbose)
        self.u = u
        self.parser = parser
        self.progress_file = os.environ.get("GLYCO_PROGRESS_FILE")
        self._run_slice = {}

        # Single source of truth for defaults: base_parser() itself, so
        # library usage (no parser passed) and CLI usage always agree.
        defaults = vars(base_parser().parse_args([]))

        # Only touch sys.argv (via parser.parse_args()) if a parser was
        # supplied WITHOUT an already-parsed `args`. Pure library usage
        # (GlycoAnalysis(u, prot_sel=..., gly_sel=...)) never parses
        # sys.argv, since it may contain unrelated arguments (pytest's own
        # CLI args, a notebook kernel's argv, ...).
        source = args if args is not None else parser
        self.apply_params(defaults=defaults, args=source, **kwargs)

        self._validate_args()
        # belt and braces: never let a None outdir reach a path/f-string
        if not self.outdir:
            self.outdir = default_outdir(self)
        self._validate_atom_selections()

    # -- configuration ------------------------------------------------------

    def _error(self, message):
        """
        Report a configuration error. Uses self.parser.error() (prints
        usage, exits status 2) when running from the CLI; raises
        ValueError instead when used as a library with no parser, since
        calling sys.exit() on an importing caller's process would be a
        surprising and unrecoverable side effect for a Python API.
        """
        if self.parser is not None:
            self.parser.error(message)
        raise ValueError(message)

    def _validate_args(self):
        # Library usage: a Universe was passed in directly and no input
        # files were named, so there is nothing about inputs to validate.
        library_mode = (self.parser is None and self.top is None
                        and self.indir is None and not self.traj)
        validate_arguments(self, self._error, check_inputs=not library_mode)

    def apply_params(self, args=None, defaults=None, **kwargs):
        """
        Merge parameters onto `self`, lowest to highest precedence:
        `defaults`, then `args` (an argparse Namespace, or an
        ArgumentParser, which is parsed from sys.argv), then kwargs.
        """
        values = dict(defaults or {})
        if args is not None:
            if isinstance(args, argparse.ArgumentParser):
                args = args.parse_args()
            values.update(vars(args))
        values.update(kwargs)

        for name, value in values.items():
            setattr(self, name, value)

    def run(self, start=None, stop=None, step=None, frames=None, **kwargs):
        """
        Like AnalysisBase.run, but falls back to the frame slice given on
        the command line / as parameters when none is passed explicitly. The
        slice is remembered so the nested FreeSASA run covers exactly the
        same frames as this one.
        """
        if all(v is None for v in (start, stop, step, frames)):
            start, stop, step = self.frame_start, self.frame_stop, self.frame_step
            frames = self.frame_list
        self._run_slice = {"start": start, "stop": stop, "step": step,
                           "frames": frames}
        return super().run(**self._run_slice, **kwargs)

    def _validate_atom_selections(self):
        """
        Validate the protein and glycan selections against the Universe.
        Per-selection results are kept in self.selection_validation_results
        (see selection_report()); any failure is reported via self._error.
        """
        prot_sel = (self.prot_sel or "").strip()
        gly_sel = (self.gly_sel or "").strip()
        self.prot_sel = prot_sel
        self.gly_sel = gly_sel

        results = {}
        for name, selection in (("protein", prot_sel), ("glycan", gly_sel)):
            if not selection:
                results[name] = {"valid": False,
                                 "message": "Selection cannot be empty."}
                continue
            try:
                atoms = self.u.select_atoms(f"({selection})")
            except Exception as e:  # noqa: BLE001
                results[name] = {
                    "valid": False,
                    "message": f"Invalid selection ('{selection}'): {e}",
                }
                continue
            if len(atoms) == 0:
                results[name] = {"valid": False,
                                 "message": f"0 atoms selected ('{selection}')"}
            else:
                results[name] = {"valid": True,
                                 "message": f"{len(atoms):,} atoms"}

        if results["protein"]["valid"]:
            n_ca = len(self.u.select_atoms(f"({prot_sel}) and name CA"))
            if n_ca == 0:
                results["protein"] = {
                    "valid": False,
                    "message": f"no CA atoms in protein selection ('{prot_sel}')",
                }

        self.selection_validation_results = results
        self.selections_valid = all(r["valid"] for r in results.values())

        if not self.selections_valid:
            problems = "; ".join(f"{name}: {r['message']}"
                                 for name, r in results.items() if not r["valid"])
            self._error(f"invalid atom selection(s): {problems}")

        self._validated_prot_sel = prot_sel
        self._validated_gly_sel = gly_sel

    def selection_report(self):
        """Human-readable summary of the validated selections."""
        lines = ["Atom selections:"]
        for name, r in self.selection_validation_results.items():
            flag = "ok" if r["valid"] else "INVALID"
            lines.append(f"  {name}: {flag} - {r['message']}")
        return "\n".join(lines)

    # -- run lifecycle -------------------------------------------------------

    def _prepare(self):
        self.frame_count = 0
        os.makedirs(self.outdir, exist_ok=True)
        self._write_progress(0, self.n_frames)
        if self._verbose:
            print(self.selection_report())

        prot_sel = self._validated_prot_sel
        gly_sel = self._validated_gly_sel

        # Heavy atoms only (to match FreeSASA input). Selections are
        # parenthesized so 'a or b' selections compose correctly.
        self.selag = heavy_atoms(self.u.select_atoms(f"({prot_sel})"))
        self.prot_atoms = self.u.select_atoms(f"({prot_sel}) and name CA")
        self.gly_atoms = heavy_atoms(self.u.select_atoms(f"({gly_sel})"))

        fsa = FreesasaAnalysis(self.selag, self.probe).run(
            verbose=self._verbose,
            progressbar_kwargs={'desc': 'Running FreeSASA Analysis'},
            **self._run_slice,
        )
        # (n_analyzed_frames, n_residues); row k <-> k-th analyzed frame
        self.residue_areas = np.asarray(fsa.results.residue_areas)

        # Residue-row mapping shared by selag (blocker candidates) and
        # prot_atoms (CA atoms, segment endpoints), both expressed relative
        # to the SAME column ordering as residue_areas. This is what makes
        # surface filtering and same-residue exclusion consistent.
        unique_resindices = np.unique(self.selag.resindices)
        if self.residue_areas.shape[1] != len(unique_resindices):
            raise ValueError(
                f"FreeSASA returned {self.residue_areas.shape[1]} residues "
                f"but the heavy-atom selection has {len(unique_resindices)}"
            )
        self._selag_atom_res_row = np.searchsorted(unique_resindices,
                                                   self.selag.resindices)

        if not np.isin(self.prot_atoms.resindices, unique_resindices).all():
            raise ValueError(
                "Some CA atoms belong to residues not present in the "
                "FreeSASA selection (prot_sel heavy-atom set); prot_sel "
                "selections for CA and heavy atoms must cover the same "
                "residues."
            )
        if len(np.unique(self.prot_atoms.resindices)) != self.prot_atoms.n_atoms:
            raise ValueError(
                "More than one 'CA' atom per residue in the protein "
                "selection; results are per residue and need exactly one."
            )
        self._prot_atom_res_row = np.searchsorted(unique_resindices,
                                                  self.prot_atoms.resindices)

        n_ca = self.prot_atoms.n_atoms
        self.all_counts = np.zeros([self.n_frames, n_ca])

        self.results = Results()
        self.results.residue_counts = self.all_counts
        self.results.n_surface_ca = np.zeros(self.n_frames, dtype=int)
        self.results.n_blocked_pairs = np.zeros(self.n_frames, dtype=int)
        self.results.mean_blocked_distance = np.full(self.n_frames, np.nan)

        # throttle progress-file writes: ~100 updates over the whole run
        # instead of one fopen+json.dump+os.replace per frame
        self._progress_every = max(1, self.n_frames // 100)

    def _single_frame(self):
        k = self._frame_index

        ca_idx = self.surface_ca_indices(k)
        blockers, dists, gtree = self.get_blockers(ca_idx)
        self.all_counts[k] = self.count_glycan_neighbors(blockers, gtree, ca_idx)

        # per-frame summary: unique blocked (CA, glycan) pairs and their
        # mean CA-glycan distance
        self.results.n_surface_ca[k] = len(ca_idx)
        if len(blockers):
            n_gly = len(self.gly_atoms)
            pair_key = (blockers[:, 0].astype(np.int64) * n_gly
                        + blockers[:, 2].astype(np.int64))
            _, first = np.unique(pair_key, return_index=True)
            self.results.n_blocked_pairs[k] = len(first)
            self.results.mean_blocked_distance[k] = float(dists[first].mean())

        self.frame_count += 1
        if self.frame_count % self._progress_every == 0 or self.frame_count == self.n_frames:
            self._write_progress(self.frame_count, self.n_frames)

    def _conclude(self):
        os.makedirs(self.outdir, exist_ok=True)
        self.write_bfactors(frame_index=self.bfactor_frame)
        self.write_frame_summary()
        if self.write_vmd:
            self.write_vmd_user_script()

    # -- geometry ------------------------------------------------------------

    def outermost(self, frame_index=None):
        """
        Residue rows (columns of residue_areas) that are SURFACE residues in
        one analyzed frame, i.e. SASA >= area_cutoff.
        """
        if frame_index is None:
            frame_index = self._frame_index
        return np.flatnonzero(self.residue_areas[frame_index] >= self.area_cutoff)

    def surface_ca_indices(self, frame_index=None):
        """Indices (into self.prot_atoms) of CA atoms in surface residues."""
        surf_res_rows = self.outermost(frame_index)
        return np.flatnonzero(np.isin(self._prot_atom_res_row, surf_res_rows))

    def _iter_pairs(self, P, gtree, ca_chunk):
        """
        Yield (i, j, dist) arrays of protein/glycan atom pairs within
        distance_cutoff, computed for chunks of P at a time so the full
        pair list for a large glycoprotein never exists in memory at once.
        i indexes P, j indexes the glycan atoms.
        """
        for s in range(0, len(P), ca_chunk):
            sub = P[s:s + ca_chunk]
            pairs = cKDTree(sub).sparse_distance_matrix(
                gtree, self.distance_cutoff, output_type="ndarray"
            )
            yield (pairs["i"].astype(np.intp) + s,
                   pairs["j"].astype(np.intp),
                   pairs["v"])

    def get_blockers(self, ca_idx=None, chunk=20000, ca_chunk=2000):
        """
        For each (CA atom, glycan atom) pair within distance_cutoff, find
        the protein heavy atoms lying between them that would block that
        line of sight.

        Segment endpoints: CA atoms of SURFACE residues this frame
        (`ca_idx`, indices into self.prot_atoms; computed if None).
        Column 0 of the result is in the GLOBAL self.prot_atoms index space,
        which is what count_glycan_neighbors expects.

        Candidate blockers: ALL protein heavy atoms (self.selag), not just
        surface ones, except atoms belonging to the same residue as the
        segment's CA (its own N, C, CB... sit within ~1.5 A of the CA and
        would otherwise "block" nearly every segment). Column 1 holds the
        blocker's index into self.selag and is informational only.

        Returns (blockers (n, 3), distances (n,), gtree or None).
        """
        if ca_idx is None:
            ca_idx = self.surface_ca_indices()

        P = np.ascontiguousarray(self.prot_atoms.positions[ca_idx], dtype=np.float64)
        B = np.ascontiguousarray(self.selag.positions, dtype=np.float64)
        G = np.ascontiguousarray(self.gly_atoms.positions, dtype=np.float64)

        gtree = cKDTree(G) if len(G) else None
        empty = (np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.float64))
        if len(P) == 0 or len(B) == 0 or len(G) == 0:
            return (*empty, gtree)

        btree = cKDTree(B)
        ca_res_rows = self._prot_atom_res_row[ca_idx]   # per row of P
        blocker_res_rows = self._selag_atom_res_row     # per row of B
        r2 = self.radius ** 2

        out = []
        dist_out = []

        for ii, gg, distances in self._iter_pairs(P, gtree, ca_chunk):
            for s in range(0, len(ii), chunk):
                i_c = ii[s:s + chunk]
                g_c = gg[s:s + chunk]
                d_c = distances[s:s + chunk]

                b = P[i_c] - G[g_c]
                half = 0.5 * np.linalg.norm(b, axis=-1)
                mid = 0.5 * (P[i_c] + G[g_c])

                # 2. Candidate blockers near the segment midpoint
                cand = btree.query_ball_point(mid, half + self.radius)
                counts = np.fromiter(map(len, cand), dtype=np.intp, count=len(cand))
                if counts.sum() == 0:
                    continue

                jj = np.concatenate(cand).astype(np.intp, copy=False)
                rep = np.repeat(np.arange(len(i_c)), counts)

                # same-residue exclusion
                other_res = blocker_res_rows[jj] != ca_res_rows[i_c[rep]]
                jj = jj[other_res]
                rep = rep[other_res]
                if len(jj) == 0:
                    continue

                # 3. Exact blocker test (point-to-segment distance)
                bb = b[rep]
                a = P[i_c[rep]] - B[jj]

                seg_len2 = np.maximum(np.einsum("ij,ij->i", bb, bb), 1e-12)
                t = np.einsum("ij,ij->i", a, bb) / seg_len2
                perp = a - t[:, None] * bb
                d2 = np.einsum("ij,ij->i", perp, perp)

                keep = (t > 0) & (t < 1) & (d2 < r2)

                if keep.any():
                    out.append(
                        np.stack(
                            [
                                ca_idx[i_c[rep][keep]],  # -> self.prot_atoms index space
                                jj[keep],                # index into self.selag
                                g_c[rep][keep],          # -> self.gly_atoms index space
                            ],
                            axis=1,
                        )
                    )
                    dist_out.append(d_c[rep][keep])

        if not out:
            return (*empty, gtree)

        res = np.concatenate(out)
        dist_res = np.concatenate(dist_out)
        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))

        return res[order], dist_res[order], gtree

    def count_glycan_neighbors(self, blockers, gtree=None, ca_idx=None, chunk=20000):
        """
        Count non-blocked glycan neighbors for each SURFACE protein CA atom.

        Returns an array of length len(self.prot_atoms). Entries for CA
        atoms in non-surface residues are 0: they are not line-of-sight
        starting points and are not computed.

        blockers[:, 0] and blockers[:, 2] are in the same index spaces as
        self.prot_atoms and self.gly_atoms respectively.
        """
        P = self.prot_atoms.positions
        n_prot = len(P)
        n_gly = len(self.gly_atoms)

        counts = np.zeros(n_prot, dtype=np.intp)

        if ca_idx is None:
            ca_idx = self.surface_ca_indices()
        if n_prot == 0 or n_gly == 0 or len(ca_idx) == 0:
            return counts

        if gtree is None:
            gtree = cKDTree(self.gly_atoms.positions)

        if len(blockers):
            excluded = np.unique(
                blockers[:, 0].astype(np.int64) * n_gly
                + blockers[:, 2].astype(np.int64)
            )
        else:
            excluded = np.empty(0, dtype=np.int64)

        for s in range(0, len(ca_idx), chunk):
            idx = ca_idx[s:s + chunk]

            neighbors = gtree.query_ball_point(P[idx], self.distance_cutoff)
            sizes = np.fromiter(
                (len(x) for x in neighbors), dtype=np.intp, count=len(neighbors),
            )
            if sizes.sum() == 0:
                continue

            local_prot = np.repeat(idx.astype(np.int64), sizes)
            gly_indices = np.concatenate(neighbors).astype(np.int64, copy=False)
            pair_ids = local_prot * n_gly + gly_indices

            if len(excluded):
                local_prot = local_prot[~np.isin(pair_ids, excluded)]

            counts += np.bincount(local_prot, minlength=n_prot)

        return counts

    # -- output --------------------------------------------------------------

    @staticmethod
    def _rows_for(residue_resindices, atom_resindices):
        """
        Row (position in `residue_resindices`) of each atom's residue. Does
        not assume `residue_resindices` is sorted.
        """
        residue_resindices = np.asarray(residue_resindices)
        order = np.argsort(residue_resindices)
        pos = np.searchsorted(residue_resindices, atom_resindices, sorter=order)
        return order[pos]

    def _analyzed_frames(self):
        """Trajectory frame index of each analyzed frame (row of all_counts)."""
        frames = getattr(self, "frames", None)
        if frames is not None and len(frames) == self.n_frames:
            return np.asarray(frames, dtype=int)
        return np.arange(self.n_frames)

    def _resolve_frame_index(self, frame_index=None):
        """Normalize an index into the analyzed frames (None -> last)."""
        n = np.asarray(self.all_counts).shape[0]
        if frame_index is None:
            return n - 1
        k = frame_index + n if frame_index < 0 else frame_index
        if not 0 <= k < n:
            raise IndexError(
                f"frame index {frame_index} out of range for {n} analyzed frame(s)"
            )
        return k

    def _ensure_tempfactors(self):
        try:
            self.u.atoms.tempfactors  # noqa: B018
        except (AttributeError, mda.exceptions.NoDataError):
            self.u.add_TopologyAttr('tempfactors')

    def write_bfactors(self, frame_index=None):
        """
        Write b_factors.pdb: the WHOLE structure (every atom in the
        Universe, hydrogens and non-protein atoms included) for one analyzed
        frame (last by default), plus b_factors.npy with the counts for all
        analyzed frames.

        B-factor of each atom = the value of the CA atom in the same
        residue. Atoms in residues without a CA in prot_atoms (glycans,
        water, ions, ...) get 0.
        """
        os.makedirs(self.outdir, exist_ok=True)
        self._ensure_tempfactors()

        # put the Universe on the frame whose counts are written, so the
        # PDB coordinates and B-factors describe the same frame
        k = self._resolve_frame_index(frame_index)
        self.u.trajectory[int(self._analyzed_frames()[k])]

        self._populate_b_factors(k)
        output_ag = self.u.select_atoms(f'{self.prot_sel} or {self.gly_sel}')
        output_ag.atoms.write(os.path.join(self.outdir, 'b_factors.pdb'))
        np.save(os.path.join(self.outdir, 'b_factors'),
                np.asarray(self.all_counts))

    def _populate_b_factors(self, frame_index=None):
        """
        Set tempfactors on the whole Universe from self.all_counts: every
        atom of a residue receives the value of that residue's CA atom, and
        all other atoms are set to 0.

        Parameters
        ----------
        frame_index : int or None
            Which row of self.all_counts to use if it is 2D (shape
            (n_frames, n_ca)); negative counts from the end. Ignored if it
            is 1D. Defaults to the last frame.

        Notes
        -----
        Column i of self.all_counts corresponds to the i-th CA atom of
        self.prot_atoms (one CA per residue is enforced in _prepare).
        """
        data = np.asarray(self.all_counts)

        if data.ndim == 1:
            ca_values = data
        elif data.ndim == 2:
            ca_values = data[self._resolve_frame_index(frame_index)]
        else:
            raise ValueError(
                f"self.all_counts must be 1D or 2D, got shape {data.shape}"
            )

        ca_resindices = self.prot_atoms.resindices
        if len(ca_values) != len(ca_resindices):
            raise ValueError(
                f"self.all_counts has {len(ca_values)} values "
                f"but self.prot_atoms has {len(ca_resindices)} CA atoms"
            )

        if ca_values.size and ca_values.max() > 999.99:
            warnings.warn(
                f"max B-factor value {ca_values.max():g} exceeds the PDB "
                "B-factor column width (999.99); the PDB columns will be "
                "misaligned. Use b_factors.npy for the raw values.",
                stacklevel=2,
            )

        # every atom in a CA residue -> that residue's CA value
        res_atoms = self.prot_atoms.residues.atoms
        rows = self._rows_for(ca_resindices, res_atoms.resindices)

        self.u.atoms.tempfactors = np.zeros(self.u.atoms.n_atoms)
        res_atoms.tempfactors = ca_values[rows]

    def write_frame_summary(self):
        """Write frame_summary.csv: one row per analyzed frame."""
        r = self.results
        table = np.column_stack([
            self._analyzed_frames(), r.n_surface_ca, r.n_blocked_pairs,
            r.mean_blocked_distance,
        ])
        np.savetxt(
            os.path.join(self.outdir, 'frame_summary.csv'), table,
            delimiter=",", comments="", fmt=["%d", "%d", "%d", "%.6g"],
            header="frame,n_surface_ca,n_blocked_pairs,mean_blocked_distance",
        )

    def _write_progress(self, current, total):
        """Write machine-readable progress for the Streamlit app."""
        if not self.progress_file:
            return

        temp_file = self.progress_file + ".tmp"

        with open(temp_file, "w") as f:
            json.dump({
                "current": current,
                "total": total,
                "percent": int(100 * current / total) if total else 0,
            }, f)

        os.replace(temp_file, self.progress_file)

    @staticmethod
    def _tcl_quote(value):
        """
        Quote a string as one Tcl word. Braces are used when safe (inside
        braces a backslash is literal, so Windows-style paths survive);
        otherwise fall back to double quotes with Tcl metacharacters escaped.
        """
        value = str(value)
        if "{" not in value and "}" not in value and not value.endswith("\\"):
            return "{" + value + "}"
        return '"' + re.sub(r'([\\"$\[\]])', r'\\\1', value) + '"'

    @staticmethod
    def _index_ranges(sorted_indices):
        """Compress sorted ints into VMD 'a to b' ranges."""
        if len(sorted_indices) == 0:
            return ""
        breaks = np.flatnonzero(np.diff(sorted_indices) != 1) + 1
        parts = []
        for run in np.split(sorted_indices, breaks):
            parts.append(str(run[0]) if len(run) == 1 else f"{run[0]} to {run[-1]}")
        return " ".join(parts)

    def write_vmd_user_script(self, output=None, frame_offset=0):
        """
        Generate a VMD Tcl script that loads the topology and trajectory
        from self.u, then assigns per-residue data to VMD's 'user'
        attribute for every ANALYZED frame.

        Creates one atom selection and moves it between frames, with one
        bulk 'user' assignment per analyzed frame. Defaults to writing into
        self.outdir.
        """
        u = self.u

        topology = getattr(u, "filename", None)
        if not topology:
            raise ValueError("Could not determine topology filename from Universe")
        topology = Path(topology).resolve()
        if not topology.is_file():
            raise FileNotFoundError(topology)

        reader = u.trajectory
        trajectory_files = getattr(reader, "filenames", None)
        if trajectory_files is None:
            filename = getattr(reader, "filename", None)
            trajectory_files = [filename] if filename else []
        trajectory_files = [Path(p).resolve() for p in trajectory_files
                            if p is not None]
        if not trajectory_files:
            raise ValueError("Could not determine trajectory filename(s)")
        for path in trajectory_files:
            if not path.is_file():
                raise FileNotFoundError(path)

        # data: (n_residues, n_analyzed_frames), one row per CA residue
        data = np.asarray(self.all_counts).T
        residues = self.prot_atoms.residues
        if data.shape[0] != len(residues):
            raise ValueError(
                f"Data has {data.shape[0]} rows, but prot_atoms contains "
                f"{len(residues)} residues"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

        n_frames = data.shape[1]
        analyzed = self._analyzed_frames()
        n_traj_frames = int(reader.n_frames)

        # every atom of every CA residue, in ascending atom-index order
        # (the order VMD uses when assigning a list to a selection)
        atoms = residues.atoms
        atom_indices = np.asarray(atoms.indices, dtype=np.int64)
        atom_rows = self._rows_for(residues.resindices, atoms.resindices)
        order = np.argsort(atom_indices, kind="stable")
        atom_indices, atom_rows = atom_indices[order], atom_rows[order]

        data_max = float(np.max(data)) if data.size else 0.0
        if data_max <= 0:
            data_max = 1.0

        lines = [
            "# Auto-generated VMD Tcl script",
            "# Bulk per-frame assignment of residue values",
            "",
            f"set molid [mol new {self._tcl_quote(topology.as_posix())} waitfor all]",
            "",
        ]
        for traj in trajectory_files:
            lines.append(
                f"mol addfile {self._tcl_quote(traj.as_posix())} waitfor all molid $molid"
            )
        lines += [
            "",
            "mol modselect 0 $molid protein",
            "mol modstyle 0 $molid NewCartoon",
            "mol modcolor 0 $molid User",
            "",
            f"set n_traj_frames {n_traj_frames}",
            "set n_vmd_frames [molinfo $molid get numframes]",
            "",
            "if {$n_vmd_frames == $n_traj_frames + 1} {",
            "    animate delete beg 0 end 0 $molid",
            '    puts "Deleted duplicate first frame."',
            "} elseif {$n_vmd_frames == $n_traj_frames} {",
            '    puts "Frame counts match; no frame deleted."',
            "} else {",
            '''    error "Frame mismatch: VMD has $n_vmd_frames frames, but the
            trajectory has $n_traj_frames. Expected equal counts or exactly
            one extra VMD frame."''',
            "}",
            "",
            "# Set User color scale",
            f"mol scaleminmax $molid 0 0 {data_max:.10g}",
            "",
            f"set sel [atomselect $molid {{index {self._index_ranges(atom_indices)}}}]",
            "",
        ]

        for k in range(n_frames):
            vmd_frame = frame_offset + int(analyzed[k])
            values = " ".join(f"{v:.10g}" for v in data[atom_rows, k])
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

        if output is None:
            output = os.path.join(self.outdir, VMD_SCRIPT_NAME)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))

        print(f"VMD script written to: {output}")
        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_job(args, parser, top, traj, outdir, verbose):
    """Load one structure/trajectory, run the analysis, write outputs."""
    u = load_universe(top, traj)
    analysis = GlycoAnalysis(
        u, parser=parser, args=args,
        top=top, traj=traj, indir=None, outdir=outdir,
    )
    analysis.run(
        verbose=verbose,
        progressbar_kwargs={'desc': 'Running GlyCo Analysis'},
    )


def _structure_outdirs(structures, base_outdir):
    """One output subfolder per structure, disambiguating clashing stems."""
    stems = [Path(p).stem for p in structures]
    dirs = []
    for path, stem in zip(structures, stems):
        name = stem if stems.count(stem) == 1 else Path(path).name.replace('.', '_')
        dirs.append(os.path.join(base_outdir, name))
    return dirs


def main(argv=None):
    parser = base_parser()
    args = parser.parse_args(argv)

    # Validate everything that does not need a Universe BEFORE loading any
    # files, so bad paths/options give a clean usage error, not a traceback.
    validate_arguments(args, parser.error)

    is_streamlit = os.environ.get("GLYCO_STREAMLIT") == "1"
    verbose = False if is_streamlit else args.verbose

    os.makedirs(args.outdir, exist_ok=True)
    configure_warning_log(os.path.join(args.outdir, "mdanalysis_warnings.log"))

    if args.indir is None:
        # single topology (+ optional trajectories): errors exit via parser
        if not confirm_overwrite(args.outdir, args.force, args.write_vmd):
            return 1
        try:
            u = load_universe(args.top, args.traj)
        except Exception as e:  # noqa: BLE001
            parser.error(
                "could not load -a/--top with -b/--traj (do all trajectory "
                f"files match the topology's atom count?): {e}"
            )
        GlycoAnalysis(u, parser=parser, args=args).run(
            verbose=verbose,
            progressbar_kwargs={'desc': 'Running GlyCo Analysis'},
        )
        return 0

    # --indir: each structure is analyzed independently into its own
    # subfolder; one bad structure does not abort the rest.
    outdirs = _structure_outdirs(args.structures, args.outdir)
    failures = []
    for n, (path, outdir) in enumerate(zip(args.structures, outdirs), start=1):
        print(f"[{n}/{len(args.structures)}] {path}")
        if not confirm_overwrite(outdir, args.force, args.write_vmd):
            failures.append((path, "output exists and overwrite declined"))
            continue
        try:
            # parser=None -> configuration errors raise ValueError here
            # instead of exiting the whole batch
            _run_job(args, None, path, None, outdir, verbose)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {path}: {e}", file=sys.stderr)
            failures.append((path, str(e)))

    if failures:
        print(f"{len(failures)} of {len(args.structures)} structure(s) failed:",
              file=sys.stderr)
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
