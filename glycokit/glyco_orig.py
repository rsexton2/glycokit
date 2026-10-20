#!/usr/bin/env python noqa: EXE001

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from scipy.spatial import cKDTree

from glycokit.mdafreesasa import FreesasaAnalysis
from glycokit.parsers import base_parser

# Redirect the warnings machinery into the logging system
logging.captureWarnings(True)
warnings_logger = logging.getLogger("py.warnings")
handler = logging.FileHandler("mdanalysis_warnings.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
warnings_logger.addHandler(handler)
warnings_logger.setLevel(logging.WARNING)
warnings_logger.propagate = False


class GlycoAnalysis(AnalysisBase):
    """
    Run Glyco and other useful output functions
    """
    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ('serial', 'multiprocessing', 'dask')


    def __init__(self, u, parser=None, verbose=False, **kwargs):
        super().__init__(u.trajectory, verbose=verbose)
        self.u = u
        self.parser = parser
        self.progress_file = os.environ.get("GLYCO_PROGRESS_FILE")

        # Single source of truth for defaults: base_parser() itself, so
        # library usage (no parser passed) and CLI usage always agree,
        # rather than maintaining a second hardcoded defaults dict here
        # that can drift out of sync with parsers.py.
        defaults = vars(base_parser().parse_args([]))

        # Only touch sys.argv (via parser.parse_args()) if a parser was
        # actually supplied -- i.e. the CLI entry point. Pure library
        # usage (GlycoAnalysis(u, prot_sel=..., gly_sel=...)) should never
        # parse sys.argv at all, since it may contain unrelated arguments
        # (e.g. pytest's own CLI args, or a notebook kernel's argv).
        parsed_args = parser.parse_args() if parser is not None else None

        self.apply_params(defaults=defaults, args=parsed_args, **kwargs)

        self._validate_args()
        self._validate_atom_selections()

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
        else:
            raise ValueError(message)

    def _validate_args(self):
        """
        Validate parameters already set on self by apply_params. Works
        identically whether those values came from a real CLI parse or
        from base_parser()'s own defaults plus kwargs (library usage) --
        everything here reads self.<attr>, never a separate args object.
        """
        top_given = self.top is not None
        traj_given = self.traj is not None
        indir_given = self.indir is not None

        modes_given = sum([top_given, indir_given])
        if modes_given == 0:
            self._error("one of -a/--top(+-b/--traj) or --indir must be given")
        if modes_given > 1:
            self._error("-a/--top(+-b/--traj) and --indir are mutually exclusive")

        if top_given:
            if not os.path.isfile(self.top):
                self._error(f"-a/--top: no such file: {self.top!r}")

            if traj_given:
                for t in self.traj:
                    if not os.path.isfile(t):
                        self._error(f"-b/--traj: no such file: {t!r}")

                try:
                    val_u = mda.Universe(self.top)
                    val_u.load_new(self.traj)
                except Exception as e:  # noqa: BLE001
                    self._error(
                        "could not load -a/--top with -b/--traj via the "
                        "chain reader (do all trajectory files match the "
                        f"topology's atom count?): {e}"
                    )

        else:  # indir_given
            if not os.path.isdir(self.indir):
                self._error(f"--indir: no such directory: {self.indir!r}")

            extensions = [e.strip().lstrip('.') for e in self.in_types.split(',') if e.strip()]
            structures = self.find_structures(self.indir, extensions)
            if not structures:
                self._error(
                    f"--indir: no files with extension(s) {extensions} "
                    f"found in {self.indir!r}"
                )
            self.structures = structures

        slice_given = any(v is not None for v in (self.start, self.stop, self.step))
        frames_given = self.frames is not None and len(self.frames) > 0

        if frames_given and slice_given:
            self._error("--frames is not allowed with --start/--stop/--step")

        if indir_given and (frames_given or slice_given):
            self._error(
                "--frames/--start/--stop/--step are not meaningful with "
                "--indir (each structure is a single, independent frame)"
            )

        for name in ('distance_cutoff', 'area_cutoff', 'radius', 'probe'):
            value = getattr(self, name)
            if value is not None and value <= 0:
                self._error(f"--{name.replace('_', '-')} must be positive, got {value}")

        if self.step is not None and self.step == 0:
            self._error("--step cannot be 0")

        if self.start is not None and self.stop is not None and self.start >= self.stop:
            self._error(f"--start ({self.start}) must be less than --stop ({self.stop})")

        if frames_given:
            try:
                frame_ints = [int(f) for f in self.frames]
            except ValueError:
                self._error(f"--frames must be integers, got {self.frames!r}")
            if any(f < 0 for f in frame_ints):
                self._error("--frames must be non-negative")
            self.frames = frame_ints

    def find_structures(self, indir, extensions):
        """
        Find independent structure files in `indir` matching `extensions`
        (a list of extensions without leading dots, e.g. ['pdb', 'gro']).
        Each match is treated as a standalone, single-file structure -- no
        trajectory pairing or chain reading involved.
        """
        files = []
        for ext in extensions:
            files.extend(sorted(glob.glob(os.path.join(indir, f'*.{ext}'))))
        return files

    def _confirm_overwrite(self, outdir=None, force=None):
        """
        If outdir exists, prompt the user to overwrite unless force=True.
        Returns True if execution should proceed, False if the user declined.

        outdir/force default to self.outdir/self.force (set directly
        on the instance by apply_params -- there is no self.args wrapper),
        but can be overridden explicitly (e.g. for per-structure output
        folders under --indir).
        """
        if outdir is None:
            outdir = self.outdir
        if force is None:
            force = self.force

        if not os.path.exists(outdir):
            return True

        if force:
            print(f"[info] {outdir!r} exists, overwriting (--force given).")
            return True

        if not sys.stdin.isatty():
            print(
                f"[error] {outdir!r} exists and no TTY is available to prompt; "
                f"pass --force to overwrite non-interactively.",
                file=sys.stderr,
            )
            return False

        while True:
            answer = input(f"Output folder {outdir!r} already exists. Overwrite? [y/N] ").strip().lower()
            if answer in ('y', 'yes'):
                return True
            if answer in ('', 'n', 'no'):
                return False
            print("Please answer 'y' or 'n'.")

    def apply_params(self, args=None, defaults=None, **kwargs):
        """
        Merge parameters from an argparse Namespace/ArgumentParser and/or
        kwargs onto `self`.

        `defaults` is only consulted for names NOT present on `args`
        (i.e. when calling with kwargs only, no parser). If `args` is
        given, its own argparse-level defaults are treated as authoritative
        for anything the user didn't override via kwargs.
        """
        defaults = defaults or {}
        values = {}

        if args is not None:
            if isinstance(args, argparse.ArgumentParser):
                args = args.parse_args()
            values.update(vars(args))

        for name, value in kwargs.items():
            values[name] = value  # kwargs always win

        all_names = set(defaults) | set(values)

        for name in all_names:
            if name in values:
                value = values[name]
            elif name in defaults:
                value = defaults[name]
                if value is ...:
                    raise ValueError(f"Missing required parameter: {name!r}")
            else:
                continue
            setattr(self, name, value)

    def _prepare(self):
        self.frame_count = 0
        self._write_progress(0, self.n_frames)

        # create selection atom group (without hydrogen to match freesasa output)
        self.selag = self.u.select_atoms(
            self.prot_sel + ' and not type H and not name H*'
        )

        fsa = FreesasaAnalysis(self.selag, self.probe_radius).run(
            verbose=self._verbose,
            progressbar_kwargs={'desc': 'Running FreeSASA Analysis'})
        self.residue_areas = np.array(fsa.results.residue_areas)  # (n_frames, n_residues)

        # Residue-row mapping shared by selag (heavy atoms, used below as
        # blocker candidates) and prot_atoms (CA atoms, used as segment
        # endpoints), both expressed relative to the SAME column ordering
        # as residue_areas -- this is what makes the two atom groups'
        # surface-residue filtering consistent with each other.
        unique_resindices = np.unique(self.selag.resindices)
        self._selag_atom_res_row = np.searchsorted(unique_resindices, self.selag.resindices)

        self.prot_atoms = self.u.select_atoms(f'{self.prot_sel} +  and name CA')
        self.gly_atoms = self.u.select_atoms(f'{self.gly_sel} and not name H*')

        missing_res = np.isin(self.prot_atoms.resindices, unique_resindices)
        if missing_res.any():
            raise ValueError(
                "Some CA atoms belong to residues not present in the "
                "FreeSASA selection (prot_sel heavy-atom set); prot_sel "
                "selections for CA and heavy atoms must cover the same "
                "residues."
            )
        self._prot_atom_res_row = np.searchsorted(unique_resindices, self.prot_atoms.resindices)

        self.all_counts = np.zeros([self.u.trajectory.n_frames, self.prot_atoms.n_atoms])

        # throttle progress-file writes: at this frequency you get ~100
        # updates over the whole run instead of one fopen+json.dump+
        # os.replace per frame
        self._progress_every = max(1, self.n_frames // 100)

    def outermost(self, frame_index=None):
        """Residue-area rows (columns of residue_areas) above area_cutoff for one frame."""
        if frame_index is None:
            frame_index = self._frame_index
        return np.where(self.residue_areas[frame_index] >= self.area_cutoff)[0]

    def _single_frame(self):
        # get protein/glycan atom pairs that are blocked by other protein atoms
        blockers, _dists, gtree = self.get_blockers()
        self.all_counts[self._frame_index] = self.count_glycan_neighbors(blockers, gtree)

        self.frame_count += 1
        if self.frame_count % self._progress_every == 0 or self.frame_count == self.n_frames:
            self._write_progress(self.frame_count, self.n_frames)

    def _conclude(self):
        self.write_bfactors()
        if self.write_vmd:
            self.write_vmd_user_script()

    def write_bfactors(self):
        self._populate_b_factors()
        self.prot_atoms.write(f'{self.outdir}/b_factors.pdb')
        np.save(f'{self.outdir}/b_factors', np.array(self.all_counts))

    def _populate_b_factors(self, frame_index=None):
        """
        Populate self.prot_atoms.tempfactors with per-residue values from
        self.all_counts, broadcasting each residue's value to all of its
        atoms.

        Parameters
        ----------
        frame_index : int or None
            Which row of self.all_counts to use, if self.all_counts is
            2D (shape (n_frames, n_residues)). Ignored if self.all_counts
            is already 1D (shape (n_residues,) -- e.g. a single-frame
            result). Defaults to the last frame if None and the array is
            2D.

        Notes
        -----
        Residue order in self.all_counts is assumed to match
        self.prot_atoms.residues order (i.e. row/column i of self.all_counts
        corresponds to self.prot_atoms.residues[i]). If self.all_counts was built
        from a different atom group's residues, values will be assigned
        to the wrong residues silently -- there's no way to detect that
        from array shape alone if the residue counts happen to match.
        """
        residues = self.prot_atoms.residues
        n_residues = len(residues)

        data = np.asarray(self.all_counts)

        if data.ndim == 1:
            residue_values = data
        elif data.ndim == 2:
            if frame_index is None:
                frame_index = data.shape[0] - 1
            residue_values = data[frame_index]
        else:
            raise ValueError(
                f"self.all_counts must be 1D or 2D, got shape {data.shape}"
            )

        if len(residue_values) != n_residues:
            raise ValueError(
                f"self.all_counts has {len(residue_values)} residue values "
                f"but self.prot_atoms has {n_residues} residues"
            )

        # Map each atom in self.prot_atoms to its residue's row in residue_values.
        # enumerate(residues) gives a stable 0..n_residues-1 index per
        # residue, matched against each atom's resindex via searchsorted
        # on the (already-sorted) unique resindices of self.prot_atoms's residues.
        unique_resindices = residues.resindices  # already one row per residue, in order
        atom_res_row = np.searchsorted(unique_resindices, self.prot_atoms.resindices)

        self.prot_atoms.tempfactors = residue_values[atom_res_row]

    def _validate_atom_selections(self):
        """Validate protein and glycan selections."""

        u = getattr(self, "u", None)

        prot_sel = (getattr(self, "prot_sel", "") or "").strip()
        gly_sel = (getattr(self, "gly_sel", "") or "").strip()

        self.prot_sel = prot_sel
        self.gly_sel = gly_sel

        if u is None:
            return False

        results = {}
        validation_passed = True

        for name, selection in [
            ("protein", prot_sel),
            ("glycan", gly_sel),
        ]:

            if not selection:
                results[name] = {
                    "valid": False,
                    "message": "Selection cannot be empty.",
                }
                validation_passed = False
                continue

            try:
                atoms = u.select_atoms(selection)

                if len(atoms) == 0:
                    results[name] = {
                        "valid": False,
                        "message": f"0 atoms selected ('{selection}')",
                    }
                    validation_passed = False
                else:
                    results[name] = {
                        "valid": True,
                        "message": f"{len(atoms):,} atoms",
                    }

            except Exception as e:  # noqa: BLE001
                results[name] = {
                    "valid": False,
                    "message": f"Invalid selection ('{selection}'): {e}",
                }
                validation_passed = False

        self.selection_validation_results = results
        self.selections_valid = validation_passed

        if validation_passed:
            self._validated_prot_sel = prot_sel
            self._validated_gly_sel = gly_sel
        else:
            if hasattr(self, "_validated_prot_sel"):
                del self._validated_prot_sel
            if hasattr(self, "_validated_gly_sel"):
                del self._validated_gly_sel

            invalid_names = [name for name, r in results.items() if not r["valid"]]
            raise ValueError(
                f"Atom selection(s) contain no atoms: {', '.join(invalid_names)}"
            )

        return validation_passed

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

    def get_blockers(self, chunk=20000):
        """
        For each (CA atom, glycan atom) pair within distance_cutoff, determine
        which protein heavy atoms lie between them and would physically
        block that line of sight.

        Segment endpoints: CA atoms (self.prot_atoms) restricted to surface
        residues this frame. Returned column 0 indices are mapped back into
        the GLOBAL self.prot_atoms index space, matching what
        count_glycan_neighbors expects.

        Candidate blockers: heavy protein atoms (self.selag) restricted to
        surface residues this frame. Their identity (column 1) is only used
        geometrically -- to test whether an atom occludes a given segment --
        and is not indexed against any other atom group downstream.

        Design note: CA atoms belonging to buried (non-surface) residues
        never appear as segment endpoints here, so count_glycan_neighbors
        will treat their glycan-neighbor counts as entirely unblocked (raw
        neighbor count, no line-of-sight filtering). This matches the
        original "compute only for surface residues" intent, now applied
        consistently to both the protein and glycan side of the geometry.
        """
        surf_res_rows = self.outermost()

        ca_surf_mask = np.isin(self._prot_atom_res_row, surf_res_rows)
        ca_global_idx = np.nonzero(ca_surf_mask)[0]  # indices into self.prot_atoms
        P = np.ascontiguousarray(self.prot_atoms.positions[ca_global_idx], dtype=np.float64)

        blocker_atoms = self.selag
        B = np.ascontiguousarray(blocker_atoms.positions, dtype=np.float64)
        G = np.ascontiguousarray(self.gly_atoms.positions, dtype=np.float64)

        if len(P) == 0 or len(B) == 0 or len(G) == 0:
            gtree = cKDTree(G) if len(G) else cKDTree(np.zeros((1, 3)))
            return (
                np.empty((0, 3), dtype=np.intp),
                np.empty(0, dtype=np.float64),
                gtree,
            )

        ptree = cKDTree(P)
        btree = cKDTree(B)
        gtree = cKDTree(G)

        # 1. Protein/glycan pairs within cutoff
        pairs = ptree.sparse_distance_matrix(
            gtree, self.distance_cutoff, output_type="ndarray"
        )

        ii, gg = pairs["i"], pairs["j"]
        distances = pairs["v"]

        out = []
        dist_out = []

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

            jj = np.concatenate(cand)
            rep = np.repeat(np.arange(len(i_c)), counts)

            # 3. Exact blocker test
            bb = b[rep]
            a = P[i_c[rep]] - B[jj]

            t = np.einsum("ij,ij->i", a, bb) / np.einsum("ij,ij->i", bb, bb)
            perp = a - t[:, None] * bb
            d2 = np.einsum("ij,ij->i", perp, perp)

            keep = (t > 0) & (t < 1) & (d2 < self.radius ** 2)

            if keep.any():
                out.append(
                    np.stack(
                        [
                            ca_global_idx[i_c[rep][keep]],  # -> self.prot_atoms index space
                            jj[keep],                       # blocker atom, local to blocker_atoms; informational only
                            g_c[rep][keep],                 # -> self.gly_atoms index space
                        ],
                        axis=1,
                    )
                )
                dist_out.append(d_c[rep][keep])

        if not out:
            return (
                np.empty((0, 3), dtype=np.intp),
                np.empty(0, dtype=np.float64),
                gtree,
            )

        res = np.concatenate(out)
        dist_res = np.concatenate(dist_out)

        order = np.lexsort((res[:, 2], res[:, 1], res[:, 0]))

        return res[order], dist_res[order], gtree

    def count_glycan_neighbors(self, blockers, gtree=None, chunk=20000):
        """
        Count non-blocked glycan neighbors for each protein CA atom.

        blockers[:, 0] and blockers[:, 2] are expected in the SAME index
        spaces as self.prot_atoms and self.gly_atoms respectively -- this
        is guaranteed by get_blockers as of the fix above.
        """
        P = self.prot_atoms.positions
        G = self.gly_atoms.positions

        n_prot = len(P)
        n_gly = len(G)

        counts = np.zeros(n_prot, dtype=np.intp)

        if n_prot == 0 or n_gly == 0:
            return counts

        if gtree is None:
            gtree = cKDTree(G)

        if len(blockers):
            excluded = (
                blockers[:, 0].astype(np.int64) * n_gly
                + blockers[:, 2].astype(np.int64)
            )
            excluded = np.unique(excluded)
        else:
            excluded = np.empty(0, dtype=np.int64)

        for s in range(0, n_prot, chunk):
            e = min(s + chunk, n_prot)

            neighbors = gtree.query_ball_point(P[s:e], self.distance_cutoff)

            sizes = np.fromiter(
                (len(x) for x in neighbors), dtype=np.intp, count=len(neighbors),
            )

            if sizes.sum() == 0:
                continue

            local_prot = np.repeat(np.arange(s, e, dtype=np.intp), sizes)
            gly_indices = np.concatenate(neighbors).astype(np.intp, copy=False)

            pair_ids = local_prot.astype(np.int64) * n_gly + gly_indices

            if len(excluded):
                keep = ~np.isin(pair_ids, excluded)
                local_prot = local_prot[keep]

            counts += np.bincount(local_prot, minlength=n_prot)

        return counts

    @staticmethod
    def _tcl_quote(value):
        """Wrap a string as a Tcl braced argument."""
        value = str(value).replace("\\", "\\\\")
        value = value.replace("{", "\\{").replace("}", "\\}")
        return "{" + value + "}"

    def write_vmd_user_script(self, output="load_user_attributes.vmd", frame_offset=0):
        """
        Generate a VMD Tcl script that loads the topology and trajectory
        from self.u, then assigns per-residue data to VMD's 'user'
        attribute.

        Uses one atom selection and one bulk 'user' assignment per frame
        for improved performance.
        """
        atoms = self.selag.atoms
        u = self.u  # was undefined before -- this is the fix

        # ---------------------------------------------------------
        # Get topology path
        # ---------------------------------------------------------
        topology = getattr(u, "filename", None)

        if not topology:
            raise ValueError("Could not determine topology filename from Universe")

        topology = Path(topology).resolve()

        if not topology.is_file():
            raise FileNotFoundError(topology)

        # ---------------------------------------------------------
        # Get trajectory path(s)
        # ---------------------------------------------------------
        reader = u.trajectory

        trajectory_files = getattr(reader, "filenames", None)

        if trajectory_files is None:
            filename = getattr(reader, "filename", None)
            trajectory_files = [filename] if filename else []

        trajectory_files = [
            Path(p).resolve() for p in trajectory_files if p is not None
        ]

        if not trajectory_files:
            raise ValueError("Could not determine trajectory filename(s)")

        for path in trajectory_files:
            if not path.is_file():
                raise FileNotFoundError(path)

        # ---------------------------------------------------------
        # Prepare data
        # ---------------------------------------------------------
        data = self.all_counts.T

        if data.ndim == 1:
            data = data[:, None]

        if data.ndim != 2:
            raise ValueError("data must be a 1D or 2D array")

        residues = atoms.residues
        n_residues = len(residues)

        if data.shape[0] != n_residues:
            raise ValueError(
                f"Data has {data.shape[0]} rows, but AtomGroup contains "
                f"{n_residues} residues"
            )

        if not np.all(np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

        n_frames = data.shape[1]

        # ---------------------------------------------------------
        # Build one flat atom-index list and residue mapping
        # ---------------------------------------------------------
        atom_indices = []
        atom_residue_rows = []

        for res_i, residue in enumerate(residues):
            res_indices = residue.atoms.indices.tolist()
            atom_indices.extend(res_indices)
            atom_residue_rows.extend([res_i] * len(res_indices))

        atom_indices = np.asarray(atom_indices, dtype=int)
        atom_residue_rows = np.asarray(atom_residue_rows, dtype=int)

        index_string = " ".join(map(str, atom_indices))

        # ---------------------------------------------------------
        # Write VMD Tcl script
        # ---------------------------------------------------------
        output = Path(output)

        with output.open("w") as f:
            f.write("# Auto-generated VMD Tcl script\n")
            f.write("# Bulk per-frame assignment of residue values\n\n")

            f.write(
                f"set molid [mol new {self._tcl_quote(topology)} waitfor all]\n\n"
            )

            for traj in trajectory_files:
                f.write(
                    f"mol addfile {self._tcl_quote(traj)} waitfor all molid $molid\n"
                )

            f.write("\n")
            f.write("mol modselect 0 $molid protein\n")
            f.write("mol modstyle 0 $molid NewCartoon\n")
            f.write("mol modcolor 0 $molid User\n")
            f.write("\n")

            f.write(f"set n_data_frames {n_frames}\n")
            f.write("set n_vmd_frames [molinfo $molid get numframes]\n\n")

            f.write("if {$n_vmd_frames == $n_data_frames + 1} {\n")
            f.write("    animate delete beg 0 end 0 $molid\n")
            f.write("    puts \"Deleted duplicate first frame.\"\n")
            f.write("} elseif {$n_vmd_frames == $n_data_frames} {\n")
            f.write("    puts \"Frame counts match; no frame deleted.\"\n")
            f.write("} else {\n")
            f.write(
                '    error "Frame mismatch: VMD has $n_vmd_frames frames, '
                'but data has $n_data_frames frames. '
                'Expected equal counts or exactly one extra VMD frame."\n'
            )
            f.write("}\n\n")

            data_max = float(np.max(data))

            f.write("# Set User color scale\n")
            f.write("mol modcolor 0 $molid User\n")
            f.write(f"mol scaleminmax 0 0 0 {data_max:.10g}\n\n")

            for frame in range(n_frames):
                vmd_frame = frame_offset + frame
                atom_values = data[atom_residue_rows, frame]
                values_string = " ".join(f"{value:.10g}" for value in atom_values)

                f.write(f"# Data frame {frame} -> VMD frame {vmd_frame}\n")
                f.write(
                    f'set sel [atomselect $molid "index {index_string}" frame {vmd_frame}]\n'
                )
                f.write(f"$sel set user {{{values_string}}}\n")
                f.write("$sel delete\n\n")

            f.write('puts "Finished assigning per-residue user values."\n')

        print(f"VMD script written to: {output}")

        return output

def main():
    parser = base_parser()
    args = parser.parse_args()

    u = mda.Universe(args.top)
    u.load_new(args.traj)

    is_streamlit = os.environ.get("GLYCO_STREAMLIT") == "1"
    verbose = False if is_streamlit else args.verbose

    GlycoAnalysis(u, parser=parser).run(
        verbose=verbose,
        progressbar_kwargs={'desc': 'Running GlyCo Analysis'})


if __name__ == "__main__":
    sys.exit(main())