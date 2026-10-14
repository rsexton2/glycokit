import warnings

import freesasa
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.guesser.tables import vdwradii

freesasa.setVerbosity(1)

# Radii FreeSASA uses when it guesses an unknown atom from its element.
# Verify these against YOUR freesasa executable (see "Verification" below).
_FREESASA_ELEMENT_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80,
}


def _freesasa_resname(name):
    """Convert an MDAnalysis residue name to a PDB/FreeSASA name."""
    return str(name).strip().upper()[:3]


def _chain_label(chain):
    """
    FreeSASA chain labels are single characters. A blank chain is passed as
    ' ' (stripped to '' in keys); longer chain IDs are truncated, and the
    MDAnalysis-side keys use the same truncated label.
    """
    chain = str(chain).strip()
    return chain[:1] if chain else " "


def _optional_attr(group, attribute, default):
    """Per-atom attribute as a list, or `default` for every atom if absent."""
    try:
        return list(getattr(group, attribute))
    except (AttributeError, mda.exceptions.NoDataError):
        return [default] * len(group)


class FreesasaAnalysis(AnalysisBase):
    """
    Run FreeSASA and parse results.

    Residue order guarantee: results.residue_areas[:, i] is the area of the
    i-th residue of the AtomGroup in ascending resindex order, i.e. of
    np.unique(ag.resindices). With a valid FreeSASA structure, the values
    are mapped by residue KEY (chain, resnum + insertion code, resname),
    never by FreeSASA's dictionary order.

    Parameters
    ----------
    ag : AtomGroup
        Atoms to compute SASA for.
    probe_radius : float
        Probe radius in Angstrom.
    keep_atom_areas : bool
        If True, also store per-atom areas for every frame in
        ``results.atom_areas`` (n_frames x n_atoms). Off by default since
        this can be large and most callers only need residue areas.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ('serial', 'multiprocessing', 'dask')

    def __init__(self, ag, probe_radius, keep_atom_areas=False):
        super().__init__(ag.universe.trajectory)
        self.ag = ag
        self.probe_radius = probe_radius
        self.keep_atom_areas = keep_atom_areas

    def __getstate__(self):
        # freesasa.Parameters (and, transitively, its C struct members)
        # cannot be pickled. Drop it before this instance crosses a
        # process boundary; __setstate__ rebuilds it on arrival.
        state = self.__dict__.copy()
        state.pop('params', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'probe_radius'):
            self.params = self._make_params()

    # -- setup -------------------------------------------------------------

    def _make_params(self):
        params = freesasa.Parameters()
        params.setProbeRadius(self.probe_radius)
        params.setNThreads(1)
        return params

    def _atom_elements(self):
        """Upper-cased element symbols, falling back to atom types."""
        try:
            elems = np.asarray(self.ag.atoms.elements, dtype=str)
        except (AttributeError, mda.exceptions.NoDataError):
            elems = np.asarray(self.ag.atoms.types, dtype=str)
        return np.char.upper(np.char.strip(elems))

    def _prepare(self):
        self._setup_atoms()
        n_res = len(self._unique_resindices)
        self.results = Results()
        self.results.residue_areas = np.zeros([self.n_frames, n_res])
        self.results.total_area = np.zeros([self.n_frames])
        if self.keep_atom_areas:
            self.results.atom_areas = np.zeros([self.n_frames, self.ag.n_atoms])

    def _setup_atoms(self):
        """
        Per-atom FreeSASA fields, residue keys, structure check and radii.
        Needs no trajectory slicing, so it can be called before run().
        """
        self.params = self._make_params()
        atoms = self.ag.atoms

        self._atom_names = [str(n).strip() for n in atoms.names]
        self._atom_resnames = [_freesasa_resname(n) for n in atoms.resnames]
        icodes = _optional_attr(atoms, "icodes", "")
        self._atom_resnums = [f"{int(resid)}{str(icode).strip()}"
                              for resid, icode in zip(atoms.resids, icodes)]
        self._atom_chains = [_chain_label(c)
                             for c in _optional_attr(atoms, "chainIDs", "")]

        (self._unique_resindices,
         first_atom,
         self._resindex_inverse) = np.unique(
            self.ag.resindices, return_index=True, return_inverse=True)
        self._residue_keys = [
            (self._atom_chains[i].strip(), self._atom_resnums[i],
             self._atom_resnames[i])
            for i in first_atom
        ]

        # set fixed radii
        self._fixed_radii = self._executable_radii()[0]

        self.structure_valid, self.structure_message = self._check_residue_shapes()
        self.radii_source = "classifier"
        self.n_guessed_radii = 0
        self.unguessable_atoms = []

        if not self.structure_valid:
            (self._radii, self.n_guessed_radii,
             self.unguessable_atoms) = self._executable_radii()
            self.radii_source = "executable"
            if self.unguessable_atoms:
                warnings.warn(
                    f"{len(self.unguessable_atoms)} atom(s) have no classifier "
                    "radius and no known element; radius 0 is used, as the "
                    f"freesasa executable does (first: {self.unguessable_atoms[:5]})",
                    stacklevel=2,
                )

    # -- FreeSASA structure ------------------------------------------------

    def _build_structure(self):
        structure = freesasa.Structure()
        positions = np.asarray(self.ag.positions, dtype=np.float64)
        positions.setflags(write=False)       # any in-place write now raises
        xs, ys, zs = (positions[:, k].tolist() for k in range(3))

        try:
            structure.addAtoms(
                self._atom_names, self._atom_resnames, self._atom_resnums,
                self._atom_chains, xs, ys, zs,
            )
        except AttributeError:
            # Older FreeSASA Python modules without addAtoms()
            for i in range(self.ag.n_atoms):
                structure.addAtom(
                    self._atom_names[i], self._atom_resnames[i],
                    self._atom_resnums[i], self._atom_chains[i],
                    xs[i], ys[i], zs[i],
                )

        for i, r in enumerate(self._fixed_radii):
            structure.setRadius(i, float(r))

        return structure

    def _compute_freesasa_result(self):
        structure = self._build_structure()
        radii = np.array([structure.radius(i) for i in range(structure.nAtoms())])
        bad = np.flatnonzero(~np.isfinite(radii) | (radii <= 0))
        if bad.size:
            raise RuntimeError(
                "invalid radii for atoms "
                + ", ".join(f"{structure.residueName(i).strip()}:{structure.atomName(i).strip()}"
                            for i in bad[:10]))
        return freesasa.calc(structure, self.params)

    def _calc_from_coords(self):
        # NOTE: residue areas are not available from calcCoord results, so
        # callers sum atom areas per residue instead.
        coords = np.asarray(self.ag.positions, dtype=np.float64).ravel().tolist()
        return freesasa.calcCoord(coords, self._radii.tolist(), self.params)

    @staticmethod
    def _freesasa_residue_key(chain, residue_area):
        return (
            str(chain).strip(),
            str(residue_area.residueNumber).strip(),
            _freesasa_resname(residue_area.residueType),
        )

    def _check_residue_shapes(self):
        """
        Check that FreeSASA and MDAnalysis describe the same residues with
        unique keys, so that areas can be mapped by key.

        Returns (valid, message).
        """
        if len(set(self._residue_keys)) != len(self._residue_keys):
            seen, dups = set(), []
            for key in self._residue_keys:
                if key in seen:
                    dups.append(key)
                seen.add(key)
            return False, ("several MDAnalysis residues share a FreeSASA key "
                           f"(chain, resnum, resname): {dups[:5]}")

        result = self._compute_freesasa_result()

        if result.nAtoms() != self.ag.n_atoms:
            return False, (f"FreeSASA added {result.nAtoms()} of "
                           f"{self.ag.n_atoms} atoms (unclassifiable atoms "
                           "are not added by Structure.addAtom)")

        freesasa_keys = [
            self._freesasa_residue_key(chain, area)
            for chain, residue_dict in result.residueAreas().items()
            for area in residue_dict.values()
        ]

        if (len(freesasa_keys) != len(self._residue_keys)
                or set(freesasa_keys) != set(self._residue_keys)):
            missing = sorted(set(self._residue_keys) - set(freesasa_keys))
            extra = sorted(set(freesasa_keys) - set(self._residue_keys))
            return False, ("FreeSASA residues do not match MDAnalysis residues; "
                           f"missing {missing[:5]}, unexpected {extra[:5]}")

        return True, ""

    def _ordered_residue_areas(self, result):
        """
        Residue areas in the order of self._residue_keys, i.e. ascending
        resindex, mapped by key.
        """
        area_by_key = {}
        for chain, residue_dict in result.residueAreas().items():
            for residue_area in residue_dict.values():
                key = self._freesasa_residue_key(chain, residue_area)
                if key in area_by_key:
                    raise ValueError(f"Duplicate FreeSASA residue key: {key}")
                area_by_key[key] = float(residue_area.total)

        missing = [key for key in self._residue_keys if key not in area_by_key]
        if missing:
            raise ValueError(
                f"FreeSASA did not return areas for these residues: {missing[:10]}"
            )

        return np.asarray([area_by_key[key] for key in self._residue_keys],
                          dtype=float)

    def _executable_radii(self):
        """
        Radii as the freesasa executable assigns them with default options:
        ProtOr (default classifier) for atoms it knows, the element radius for
        unknown atoms, and 0 when the element cannot be guessed.
        Returns (radii, n_guessed, unguessable_indices).
        """
        classifier = freesasa.Classifier()          # default classifier (ProtOr)
        elements = self._atom_elements()
        # radii = np.empty(self.ag.n_atoms, dtype=float)

        radii = np.full(self.ag.n_atoms, np.nan) # instead of np.empty(n_atoms)

        n_guessed, unguessable = 0, []
        for i, (resname, name, element) in enumerate(
                zip(self._atom_resnames, self._atom_names, elements)):
            r = classifier.radius(resname, name)
            if r < 0:                               # unknown to the classifier
                r = _FREESASA_ELEMENT_RADII.get(element)
                if r is None:
                    unguessable.append(i)
                    r = 0.0                         # executable: radius 0
                else:
                    n_guessed += 1
            radii[i] = r

        bad = np.flatnonzero(~np.isfinite(radii) | (radii <= 0))
        if bad.size:
            raise RuntimeError(f"unassigned/invalid radii for atoms {bad[:10]} "
                            f"({[ (self.ag[i].resname, self.ag[i].name) for i in bad[:5]] })")

        return radii, n_guessed, unguessable

    # -- per frame ---------------------------------------------------------

    def _single_frame(self):
        if self.structure_valid:
            result = self._compute_freesasa_result()
        else:
            result = self._calc_from_coords()

        n_atoms = result.nAtoms()
        atom_areas = np.fromiter(
            (result.atomArea(i) for i in range(n_atoms)),
            dtype=float, count=n_atoms,
        )

        if self.structure_valid:
            residue_areas = self._ordered_residue_areas(result)
        else:
            residue_areas = self.sum_areas_per_residue(atom_areas)

        if len(residue_areas) != len(self._unique_resindices):
            raise ValueError(
                "Unexpected number of residue areas: "
                f"{len(residue_areas)}; expected {len(self._unique_resindices)}"
            )

        self.results.residue_areas[self._frame_index, :] = residue_areas
        self.results.total_area[self._frame_index] = result.totalArea()

        if self.keep_atom_areas:
            if len(atom_areas) != self.ag.n_atoms:
                raise ValueError(
                    "FreeSASA returned an unexpected number of atom areas: "
                    f"{len(atom_areas)}; expected {self.ag.n_atoms}"
                )
            self.results.atom_areas[self._frame_index, :] = atom_areas

    def sum_areas_per_residue(self, areas):
        if len(areas) != self.ag.n_atoms:
            raise ValueError(
                f"areas has {len(areas)} entries but atomgroup has "
                f"{self.ag.n_atoms} atoms"
            )
        return np.bincount(
            self._resindex_inverse, weights=areas,
            minlength=len(self._unique_resindices),
        )

    # -- parallel backends -------------------------------------------------

    def _get_aggregator(self):
        lookup = {
            "residue_areas": ResultsGroup.ndarray_vstack,   # (frames, residues)
            "total_area": ResultsGroup.ndarray_hstack,      # (frames,)
        }
        if self.keep_atom_areas:
            lookup["atom_areas"] = ResultsGroup.ndarray_vstack
        return ResultsGroup(lookup=lookup)
