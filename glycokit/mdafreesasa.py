"""
mdafreesasa: FreeSASA solvent-accessible surface areas as an MDAnalysis analysis.

:class:`FreesasaAnalysis` computes per-residue (and optionally per-atom) SASA
for every frame of a trajectory, with radii assigned as the ``freesasa``
executable assigns them with its default options.
"""

import warnings

import freesasa
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.guesser.default_guesser import DefaultGuesser
from MDAnalysis.guesser.tables import vdwradii

freesasa.setVerbosity(1)

# Radii FreeSASA uses when it guesses an unknown atom from its element.
# Check them against your freesasa executable if v1/v2 must match it exactly.
_FREESASA_ELEMENT_RADII = {
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
}

# Fallback for other elements (ions, metals); may differ from the executable.
_VDW_RADII = {str(k).strip().upper(): float(v) for k, v in vdwradii.items()}


def _freesasa_resname(name):
    """
    Residue name as FreeSASA sees it: stripped, upper case, at most 3 characters.

    :param name: MDAnalysis residue name.
    :rtype: str
    """
    return str(name).strip().upper()[:3]


def _chain_label(chain):
    """
    FreeSASA chain label (a single character).

    A blank chain is passed as ``' '`` (stripped to ``''`` in residue keys);
    longer chain IDs are truncated, and the MDAnalysis-side keys use the same
    truncated label.

    :param chain: MDAnalysis chain ID.
    :rtype: str
    """
    chain = str(chain).strip()
    return chain[:1] if chain else " "


def _optional_attr(group, attribute, default):
    """
    Per-atom attribute as a list, or ``default`` for every atom if absent.

    :param group: Atoms to read.
    :type group: MDAnalysis.core.groups.AtomGroup
    :param str attribute: Attribute name (e.g. ``'icodes'``).
    :param default: Value used for every atom if the attribute is missing.
    :rtype: list
    """
    try:
        return list(getattr(group, attribute))
    except (AttributeError, mda.exceptions.NoDataError):
        return [default] * len(group)


class FreesasaAnalysis(AnalysisBase):
    """
    FreeSASA solvent-accessible surface area for every frame of a trajectory.

    Residue order: ``results.residue_areas[:, i]`` is the area of the i-th
    residue of ``ag`` in ascending resindex order, i.e. of
    ``np.unique(ag.resindices)``. With a valid FreeSASA structure, areas are
    mapped by residue key (chain, resnum + insertion code, resname), never by
    FreeSASA's dictionary order.

    Radii follow the freesasa executable with default options: ProtOr
    (default classifier) radii for atoms it knows, FreeSASA's element radii
    for unknown H/C/N/O/P/S atoms, and MDAnalysis' vdW table for any other
    element (with a warning). If FreeSASA's residues do not match the
    MDAnalysis residues (``structure_valid`` is False), areas are computed
    from the coordinates with the same radii and summed per residue.

    Results after :meth:`run`:

    * ``results.residue_areas``: shape (n_frames, n_residues)
    * ``results.total_area``: shape (n_frames,)
    * ``results.atom_areas``: shape (n_frames, n_atoms), only with
      ``keep_atom_areas=True``

    Attributes set during setup: ``structure_valid``, ``structure_message``,
    ``radii_source`` (``'classifier'`` or ``'executable'``),
    ``n_guessed_radii`` (FreeSASA element radii) and ``n_vdw_radii``
    (MDAnalysis vdW fallback).

    :param ag: Atoms to compute SASA for.
    :type ag: MDAnalysis.core.groups.AtomGroup
    :param float probe_radius: Probe radius in Angstrom.
    :param bool keep_atom_areas: Also store per-atom areas for every frame
        (can be large; most callers only need residue areas).
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        """
        Backends usable with :meth:`run`.

        :rtype: tuple[str]
        """
        return ("serial", "multiprocessing", "dask")

    def __init__(self, ag, probe_radius, keep_atom_areas=False):
        super().__init__(ag.universe.trajectory)
        self.ag = ag
        self.probe_radius = probe_radius
        self.keep_atom_areas = keep_atom_areas
        self.params = None
        # Populated by _setup_atoms during the AnalysisBase preparation hook.
        self._atom_names = None
        self._atom_resnames = None
        self._atom_resnums = None
        self._atom_chains = None
        self._unique_resindices = None
        self._resindex_inverse = None
        self._residue_keys = None
        self.n_guessed_radii = 0
        self.n_vdw_radii = 0
        self._fixed_radii = None
        self._radii = None
        self.structure_valid = None
        self.structure_message = None
        self.radii_source = None

    def __getstate__(self):
        """
        Pickle state without the FreeSASA parameters.

        ``freesasa.Parameters`` (and its C struct members) cannot be pickled;
        it is dropped before the instance crosses a process boundary and
        rebuilt by :meth:`__setstate__`.

        :rtype: dict
        """
        state = self.__dict__.copy()
        state.pop("params", None)
        return state

    def __setstate__(self, state):
        """Restore the pickled state and rebuild the FreeSASA parameters."""
        self.__dict__.update(state)
        if hasattr(self, "probe_radius"):
            self.params = self._make_params()

    # -- setup -------------------------------------------------------------

    def _make_params(self):
        """
        FreeSASA parameters: probe radius, one thread (parallelism is per frame).

        :rtype: freesasa.Parameters
        """
        params = freesasa.Parameters()
        params.setProbeRadius(self.probe_radius)
        params.setNThreads(1)
        return params

    def _atom_elements(self):
        """
        Upper-case element symbol of every atom.

        Missing or blank elements are guessed from the atom names with
        MDAnalysis' element guesser (atom types are not used: in force-field
        topologies they are types such as ``CT1``, not elements).

        :rtype: numpy.ndarray (dtype object)
        """
        atoms = self.ag.atoms
        try:
            elements = [str(e).strip().upper() for e in atoms.elements]
        except (AttributeError, mda.exceptions.NoDataError):
            elements = [""] * len(atoms)
        missing = [i for i, e in enumerate(elements) if not e]
        if missing:
            guess = DefaultGuesser(self.ag.universe).guess_atom_element
            names = atoms.names
            for i in missing:
                elements[i] = str(guess(names[i])).strip().upper()
        return np.asarray(elements, dtype=object)

    def _prepare(self):
        """Set up atoms and radii and allocate results (AnalysisBase hook)."""
        self._setup_atoms()
        n_res = len(self._unique_resindices)
        self.results = Results()
        self.results.residue_areas = np.zeros([self.n_frames, n_res])
        self.results.total_area = np.zeros([self.n_frames])
        if self.keep_atom_areas:
            self.results.atom_areas = np.zeros([self.n_frames, self.ag.n_atoms])

    def _setup_atoms(self):
        """
        Per-atom FreeSASA fields, residue keys, radii and the structure check.

        Needs no trajectory slicing, so it can be called before :meth:`run`.

        :raises RuntimeError: If an atom has no radius (unknown element).
        """
        self.params = self._make_params()
        atoms = self.ag.atoms

        self._atom_names = [str(n).strip() for n in atoms.names]
        self._atom_resnames = [_freesasa_resname(n) for n in atoms.resnames]
        icodes = _optional_attr(atoms, "icodes", "")
        self._atom_resnums = [
            f"{int(resid)}{str(icode).strip()}"
            for resid, icode in zip(atoms.resids, icodes)
        ]
        self._atom_chains = [
            _chain_label(c) for c in _optional_attr(atoms, "chainIDs", "")
        ]

        (self._unique_resindices, first_atom, self._resindex_inverse) = np.unique(
            self.ag.resindices, return_index=True, return_inverse=True
        )
        self._residue_keys = [
            (
                self._atom_chains[i].strip(),
                self._atom_resnums[i],
                self._atom_resnames[i],
            )
            for i in first_atom
        ]

        # radii once; used by the FreeSASA structure and the coordinate path
        radii, self.n_guessed_radii, self.n_vdw_radii = self._executable_radii()
        self._fixed_radii = radii
        self._radii = radii
        if self.n_vdw_radii:
            warnings.warn(
                f"{self.n_vdw_radii} atom(s) have an element without a FreeSASA "
                "element radius and use MDAnalysis vdW radii; areas may differ "
                "from the freesasa executable",
                stacklevel=2,
            )

        self.structure_valid, self.structure_message = self._check_residue_shapes()
        self.radii_source = "classifier" if self.structure_valid else "executable"

    # -- FreeSASA structure ------------------------------------------------

    def _build_structure(self):
        """
        FreeSASA structure of the current frame with the fixed radii.

        :rtype: freesasa.Structure
        """
        structure = freesasa.Structure()
        positions = np.asarray(self.ag.positions, dtype=np.float64)
        positions.setflags(write=False)  # any in-place write now raises
        xs, ys, zs = (positions[:, k].tolist() for k in range(3))

        try:
            structure.addAtoms(
                self._atom_names,
                self._atom_resnames,
                self._atom_resnums,
                self._atom_chains,
                xs,
                ys,
                zs,
            )
        except AttributeError:
            # older FreeSASA Python modules without addAtoms()
            for i in range(self.ag.n_atoms):
                structure.addAtom(
                    self._atom_names[i],
                    self._atom_resnames[i],
                    self._atom_resnums[i],
                    self._atom_chains[i],
                    xs[i],
                    ys[i],
                    zs[i],
                )

        for i, r in enumerate(self._fixed_radii):
            structure.setRadius(i, float(r))
        return structure

    def _compute_freesasa_result(self):
        """
        FreeSASA calculation on a structure of the current frame.

        :rtype: freesasa.Result
        :raises RuntimeError: If any radius is not positive and finite.
        """
        structure = self._build_structure()
        radii = np.array([structure.radius(i) for i in range(structure.nAtoms())])
        bad = np.flatnonzero(~np.isfinite(radii) | (radii <= 0))
        if bad.size:
            raise RuntimeError(
                "invalid radii for atoms "
                + ", ".join(
                    f"{structure.residueName(i).strip()}:"
                    f"{structure.atomName(i).strip()}"
                    for i in bad[:10]
                )
            )
        return freesasa.calc(structure, self.params)

    def _calc_from_coords(self):
        """
        FreeSASA calculation from coordinates and radii only.

        Used when the structure check fails. Residue areas are not available
        from ``calcCoord`` results, so callers sum atom areas per residue.

        :rtype: freesasa.Result
        """
        coords = np.asarray(self.ag.positions, dtype=np.float64).ravel().tolist()
        return freesasa.calcCoord(coords, self._radii.tolist(), self.params)

    @staticmethod
    def _freesasa_residue_key(chain, residue_area):
        """
        (chain, resnum + icode, resname) key of a FreeSASA residue area.

        :rtype: tuple[str, str, str]
        """
        return (
            str(chain).strip(),
            str(residue_area.residueNumber).strip(),
            _freesasa_resname(residue_area.residueType),
        )

    def _check_residue_shapes(self):
        """
        Check that FreeSASA and MDAnalysis describe the same residues with
        unique keys, so that areas can be mapped by key.

        :return: ``(valid, message)``; the message explains a failed check.
        :rtype: tuple[bool, str]
        """
        if len(set(self._residue_keys)) != len(self._residue_keys):
            seen, dups = set(), []
            for key in self._residue_keys:
                if key in seen:
                    dups.append(key)
                seen.add(key)
            return False, (
                "several MDAnalysis residues share a FreeSASA key "
                f"(chain, resnum, resname): {dups[:5]}"
            )

        result = self._compute_freesasa_result()
        if result.nAtoms() != self.ag.n_atoms:
            return False, (
                f"FreeSASA added {result.nAtoms()} of "
                f"{self.ag.n_atoms} atoms (unclassifiable atoms "
                "are not added by Structure.addAtom)"
            )

        freesasa_keys = [
            self._freesasa_residue_key(chain, area)
            for chain, residue_dict in result.residueAreas().items()
            for area in residue_dict.values()
        ]
        if len(freesasa_keys) != len(self._residue_keys) or set(freesasa_keys) != set(
            self._residue_keys
        ):
            missing = sorted(set(self._residue_keys) - set(freesasa_keys))
            extra = sorted(set(freesasa_keys) - set(self._residue_keys))
            return False, (
                "FreeSASA residues do not match MDAnalysis residues; "
                f"missing {missing[:5]}, unexpected {extra[:5]}"
            )
        return True, ""

    def _ordered_residue_areas(self, result):
        """
        Residue areas in the order of the residue keys (ascending resindex),
        mapped by key.

        :param result: FreeSASA result of a valid structure.
        :type result: freesasa.Result
        :rtype: numpy.ndarray of float
        :raises ValueError: On duplicate or missing FreeSASA residue keys.
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
        return np.asarray([area_by_key[key] for key in self._residue_keys], dtype=float)

    def _executable_radii(self):
        """
        Radii as the freesasa executable assigns them with default options.

        ProtOr (default classifier) for atoms it knows; FreeSASA's element
        radius for unknown H/C/N/O/P/S atoms; MDAnalysis' vdW radius for any
        other element.

        :return: ``(radii, n_guessed, n_vdw)``: the radii, the number of
            FreeSASA element radii and the number of vdW fallback radii.
        :rtype: tuple[numpy.ndarray, int, int]
        :raises RuntimeError: If an atom has no radius (unknown element) or
            the classifier returns a non-positive radius.
        """
        classifier = freesasa.Classifier()  # default classifier (ProtOr)
        elements = self._atom_elements()
        radii = np.full(self.ag.n_atoms, np.nan)
        n_guessed = n_vdw = 0
        for i, (resname, name, element) in enumerate(
            zip(self._atom_resnames, self._atom_names, elements)
        ):
            r = classifier.radius(resname, name)
            if r < 0:  # unknown to the classifier
                r = _FREESASA_ELEMENT_RADII.get(element)
                if r is not None:
                    n_guessed += 1
                else:
                    r = _VDW_RADII.get(element)
                    if r is None:
                        continue  # reported below
                    n_vdw += 1
            radii[i] = r

        bad = np.flatnonzero(~(radii > 0))  # also catches nan
        if bad.size:
            listed = ", ".join(
                f"{self.ag[i].resname}:{self.ag[i].name} (element {elements[i] or '?'})"
                for i in bad[:10]
            )
            raise RuntimeError(f"no valid radius for {bad.size} atom(s): {listed}")
        return radii, n_guessed, n_vdw

    # -- per frame ---------------------------------------------------------

    def _single_frame(self):
        """
        Residue (and optionally atom) areas of the current frame (AnalysisBase hook).

        :raises ValueError: On an unexpected number of residue or atom areas.
        """
        if self.structure_valid:
            result = self._compute_freesasa_result()
        else:
            result = self._calc_from_coords()

        n_atoms = result.nAtoms()
        atom_areas = np.fromiter(
            (result.atomArea(i) for i in range(n_atoms)), dtype=float, count=n_atoms
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
        """
        Sum per-atom areas per residue (ascending resindex order).

        :param numpy.ndarray areas: One area per atom of ``ag``.
        :rtype: numpy.ndarray of float
        :raises ValueError: If ``areas`` does not have one entry per atom.
        """
        if len(areas) != self.ag.n_atoms:
            raise ValueError(
                f"areas has {len(areas)} entries but atomgroup has "
                f"{self.ag.n_atoms} atoms"
            )
        return np.bincount(
            self._resindex_inverse,
            weights=areas,
            minlength=len(self._unique_resindices),
        )

    # -- parallel backends -------------------------------------------------

    def _get_aggregator(self):
        """
        How results from parallel workers are combined.

        :rtype: MDAnalysis.analysis.results.ResultsGroup
        """
        lookup = {
            "residue_areas": ResultsGroup.ndarray_vstack,  # (frames, residues)
            "total_area": ResultsGroup.ndarray_hstack,  # (frames,)
        }
        if self.keep_atom_areas:
            lookup["atom_areas"] = ResultsGroup.ndarray_vstack
        return ResultsGroup(lookup=lookup)
