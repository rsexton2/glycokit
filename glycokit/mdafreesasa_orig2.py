import freesasa
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.guesser.tables import vdwradii

freesasa.setVerbosity(1)

def _freesasa_resname(name):
    """Convert an MDAnalysis residue name to a PDB/FreeSASA name."""
    return str(name).strip().upper()[:3]


def _vstack(arrays):
    return np.vstack(
        [
            np.asarray(array)
            for array in arrays
        ]
    )


def _residue_area_items(result):
    """
    Return FreeSASA ResidueArea objects across FreeSASA API layouts.
    """
    areas = result.residueAreas()

    items = []

    for chain_areas in areas.values():
        if isinstance(chain_areas, dict):
            # Current FreeSASA Python API:
            # chain -> residue number -> ResidueArea
            items.extend(chain_areas.values())
        else:
            # Compatibility with versions returning a list/sequence
            items.extend(chain_areas)

    return items

def _concat(arrays):
    return np.concatenate(arrays, axis=0)


class FreesasaAnalysis(AnalysisBase):
    """
    Run FreeSASA and parse results.

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
        self.params = self._make_params()

        self._atom_names = np.asarray(
            self.ag.atoms.names,
            dtype=str,
        )

        self._atom_resnames = np.asarray(
            [
                _freesasa_resname(name)
                for name in self.ag.atoms.resnames
            ],
            dtype=str,
        )

        self._atom_resids = [
            str(resid)
            for resid in self.ag.atoms.resids
        ]

        try:
            self._atom_chains = [
                str(chain)
                for chain in self.ag.atoms.chainIDs
            ]
        except (AttributeError, mda.exceptions.NoDataError):
            self._atom_chains = ["A"] * self.ag.n_atoms

        if self._check_residue_shapes():
            self.structure_valid = True
        else:
            self.structure_valid = False
            elems = self._atom_elements()
            unknown = sorted({e for e in elems if e not in vdwradii})
            if unknown:
                raise ValueError(
                    "Cannot assign van der Waals radii for element(s) "
                    f"{unknown}; add an 'elements' attribute to the topology."
                )
            self._radii = np.array([vdwradii[e] for e in elems])

        self._unique_resindices, self._resindex_inverse = np.unique(
            self.ag.resindices, return_inverse=True
        )

        self.results = Results()
        self.results.residue_areas = np.zeros([self.n_frames, self.ag.n_residues])
        self.results.total_area = np.zeros([self.n_frames])
        if self.keep_atom_areas:
            self.results.atom_areas = np.zeros([self.n_frames, self.ag.n_atoms])

    # def _single_frame(self):
    #     if self.structure_valid:
    #         result = self._compute_freesasa_result()
    #     else:
    #         result = self._calc_from_coords()

    #     total_area = result.totalArea()
    #     n_atoms = result.nAtoms()
    #     atom_areas = np.fromiter(
    #         (result.atomArea(i) for i in range(n_atoms)),
    #         dtype=float, count=n_atoms,
    #     )

    #     if self.structure_valid:
    #         residue_areas = self._ordered_residue_areas(result)
    #     else:
    #         residue_areas = self.sum_areas_per_residue(atom_areas)

    #     # self.results.residue_areas[self._frame_index] = residue_areas

    #     residue_areas = np.asarray(
    #         residue_areas,
    #         dtype=float,
    #     ).reshape(-1)

    #     if len(residue_areas) != self.ag.n_residues:
    #         raise ValueError(
    #             "FreeSASA returned an unexpected number of residue areas: "
    #             f"{len(residue_areas)}; expected {self.ag.n_residues}"
    #         )

    #     self.results.residue_areas[
    #         self._frame_index,
    #         :
    #     ] = residue_areas

    #     self.results.total_area[self._frame_index] = total_area
    #     if self.keep_atom_areas:
    #         self.results.atom_areas[self._frame_index] = atom_areas
    def _single_frame(self):
        if self.structure_valid:
            result = self._compute_freesasa_result()
        else:
            result = self._calc_from_coords()

        total_area = result.totalArea()
        n_atoms = result.nAtoms()

        atom_areas = np.fromiter(
            (
                result.atomArea(i)
                for i in range(n_atoms)
            ),
            dtype=float,
            count=n_atoms,
        )

        if self.structure_valid:
            residue_areas = [
                sublist[item].total
                for sublist in result.residueAreas().values()
                for item in sublist
            ]
        else:
            residue_areas = self.sum_areas_per_residue(atom_areas)

        residue_areas = np.asarray(
            residue_areas,
            dtype=float,
        ).reshape(-1)

        if len(residue_areas) != self.ag.n_residues:
            raise ValueError(
                "FreeSASA returned an unexpected number of residue areas: "
                f"{len(residue_areas)}; expected {self.ag.n_residues}"
            )

        self.results.residue_areas[
            self._frame_index,
            :
        ] = residue_areas

        self.results.total_area[
            self._frame_index
        ] = total_area

        if self.keep_atom_areas:
            atom_areas = np.asarray(
                atom_areas,
                dtype=float,
            ).reshape(-1)

            if len(atom_areas) != self.ag.n_atoms:
                raise ValueError(
                    "FreeSASA returned an unexpected number of atom areas: "
                    f"{len(atom_areas)}; expected {self.ag.n_atoms}"
                )

            self.results.atom_areas[
                self._frame_index,
                :
            ] = atom_areas

    def _residue_key(self, residue):
        try:
            chain = str(residue.atoms.chainIDs[0]).strip()
        except (AttributeError, mda.exceptions.NoDataError):
            chain = ""

        return (
            chain,
            str(residue.resid).strip(),
            str(residue.resname).strip().upper()[:3],
        )

    def _ordered_residue_areas(self, result):
        """
        Return residue SASA values in exactly the order of self.ag.residues.
        """
        free_areas = result.residueAreas()

        by_key = {}

        for chain, residue_dict in free_areas.items():
            for residue_number, residue_area in residue_dict.items():
                key = (
                    str(chain).strip(),
                    str(residue_number).strip(),
                    str(residue_area.residueType).strip().upper()[:3],
                )

                by_key[key] = float(residue_area.total)

        values = []
        missing = []

        for residue in self.ag.residues:
            key = self._residue_key(residue)

            if key not in by_key:
                missing.append(key)
                values.append(0.0)
            else:
                values.append(by_key[key])

        if missing:
            raise ValueError(
                "FreeSASA did not return areas for these MDAnalysis "
                f"residues: {missing[:10]}"
            )

        return np.asarray(values, dtype=float)

    # def _get_aggregator(self):
    #     lookup = {'residue_areas': _concat, 'total_area': _concat}
    #     if self.keep_atom_areas:
    #         lookup['atom_areas'] = _concat
    #     return ResultsGroup(lookup=lookup)

    def _get_aggregator(self):
        lookup = {
            "residue_areas": ResultsGroup.ndarray_vstack,
            "total_area": ResultsGroup.ndarray_vstack,
        }

        if self.keep_atom_areas:
            lookup["atom_areas"] = ResultsGroup.ndarray_vstack

        return ResultsGroup(
            lookup=lookup
        )

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

    def _build_structure(self):
        structure = freesasa.Structure()
        positions = self.ag.positions

        names = self._atom_names
        resnames = self._atom_resnames
        resids = self._atom_resids
        chains = self._atom_chains

        for i in range(self.ag.n_atoms):
            x, y, z = positions[i]

            structure.addAtom(
                str(names[i]).strip(),
                str(resnames[i]).strip(),
                str(resids[i]).strip(),
                str(chains[i]).strip(),
                float(x),
                float(y),
                float(z),
            )

        return structure

    def _compute_freesasa_result(self):
        structure = self._build_structure()
        return freesasa.calc(structure, self.params)

    def _calc_from_coords(self):
        # NOTE: residue areas are not available from calcCoord results, so
        # callers sum atom areas per residue instead.
        coords = self.ag.positions.flatten()
        return freesasa.calcCoord(coords, self._radii, self.params)

    def _check_residue_shapes(self):
        """
        Check that FreeSASA and MDAnalysis contain the same residues.
        Ordering is handled separately by _ordered_residue_areas().
        """
        result = self._compute_freesasa_result()

        freesasa_keys = []

        for chain, residue_dict in result.residueAreas().items():
            for residue_area in residue_dict.values():
                freesasa_keys.append(
                    self._freesasa_residue_key(
                        chain,
                        residue_area,
                    )
                )

        mda_keys = [
            self._mdanalysis_residue_key(residue)
            for residue in self.ag.residues
        ]

        return (
            len(freesasa_keys) == len(mda_keys)
            and set(freesasa_keys) == set(mda_keys)
        )

    @staticmethod
    def _canonical_resname(name):
        return str(name).strip().upper()[:3]


    def _mdanalysis_residue_key(self, residue):
        try:
            chain = str(residue.atoms.chainIDs[0]).strip()
        except (AttributeError, mda.exceptions.NoDataError):
            chain = ""

        return (
            chain,
            str(residue.resid).strip(),
            self._canonical_resname(residue.resname),
        )


    def _freesasa_residue_key(self, chain, residue_area):
        return (
            str(chain).strip(),
            str(residue_area.residueNumber).strip(),
            self._canonical_resname(residue_area.residueType),
        )


    def _ordered_residue_areas(self, result):
        """
        Return FreeSASA residue areas in exactly the order of
        self.ag.residues.
        """
        residue_area_map = {}

        for chain, residue_dict in result.residueAreas().items():
            # In the installed FreeSASA API, residue_dict is:
            #
            #     residue_number -> ResidueArea
            #
            for residue_area in residue_dict.values():
                key = self._freesasa_residue_key(
                    chain,
                    residue_area,
                )

                if key in residue_area_map:
                    raise ValueError(
                        f"Duplicate FreeSASA residue key: {key}"
                    )

                residue_area_map[key] = float(
                    residue_area.total
                )

        ordered = []
        missing = []

        for residue in self.ag.residues:
            key = self._mdanalysis_residue_key(residue)

            if key not in residue_area_map:
                missing.append(key)
            else:
                ordered.append(residue_area_map[key])

        if missing:
            raise ValueError(
                "FreeSASA did not return areas for these residues: "
                f"{missing[:10]}"
            )

        if len(ordered) != self.ag.n_residues:
            raise ValueError(
                "FreeSASA residue-area count does not match the "
                "MDAnalysis residue count: "
                f"{len(ordered)} != {self.ag.n_residues}"
            )

        return np.asarray(ordered, dtype=float)