# import freesasa
# import MDAnalysis as mda
# import numpy as np
# from MDAnalysis.analysis.base import AnalysisBase, Results

# freesasa.setVerbosity(1)


# class FreesasaAnalysis(AnalysisBase):
#     """
#     Run FreeSASA and parse results
#     """

#     def __init__(self, ag, probe_radius, verbose=True, **kwargs):
#         super().__init__(ag.universe.trajectory)
#         self.ag = ag
#         self.probe_radius = probe_radius
#         self._verbose = verbose

#     def _prepare(self):
#         self.params = freesasa.Parameters()
#         self.params.setProbeRadius(self.probe_radius)
#         self.params.setNThreads(10)
#         self.params.setAlgorithm(freesasa.LeeRichards)

#         # --- cache static per-atom metadata (identical every frame) ---
#         self._atom_names = self.ag.atoms.names
#         self._atom_resnames = self.ag.atoms.resnames
#         self._atom_resids = [str(r) for r in self.ag.atoms.resids]
#         try:
#             self._atom_chains = [str(c) for c in self.ag.atoms.chainIDs]
#         except (AttributeError, mda.exceptions.NoDataError):
#             self._atom_chains = ['A'] * self.ag.n_atoms

#         if self._check_residue_shapes():
#             self.structure_valid = True
#         else:
#             self.structure_valid = False
#             from MDAnalysis.guesser.tables import vdwradii
#             self._radii = np.array(
#                 [vdwradii.get(elem) for elem in self.ag.atoms.types]
#             )

#         # cache resindex grouping once, instead of every _calc_from_coords call
#         self._unique_resindices, self._resindex_inverse = np.unique(
#             self.ag.resindices, return_inverse=True
#         )

#         trajlen = len(self.ag.universe.trajectory)
#         self.results = Results()
#         self.results.atom_areas = np.zeros([trajlen, self.ag.n_atoms])
#         self.results.residue_areas = np.zeros([trajlen, self.ag.n_residues])
#         self.results.total_area = np.zeros([trajlen])

#     def _single_frame(self):
#         if self.structure_valid:
#             result = self._compute_freesasa_result()
#         else:
#             result = self._calc_from_coords()

#         total_area = result.totalArea()
#         n_atoms = result.nAtoms()
#         atom_areas = np.fromiter(
#             (result.atomArea(i) for i in range(n_atoms)),
#             dtype=float, count=n_atoms,
#         )

#         if self.structure_valid:
#             residue_areas = [sublist[item].total for sublist in
#                               result.residueAreas().values() for item in sublist]
#         else:
#             residue_areas = self.sum_areas_per_residue(atom_areas)

#         self.results.atom_areas[self._frame_index] = atom_areas
#         self.results.residue_areas[self._frame_index] = residue_areas
#         self.results.total_area[self._frame_index] = total_area

#     def sum_areas_per_residue(self, areas):
#         """
#         Sum per-atom areas on a per-residue basis, using the resindex
#         grouping cached in _prepare.
#         """
#         if len(areas) != self.ag.n_atoms:
#             raise ValueError(
#                 f"areas has {len(areas)} entries but atomgroup has "
#                 f"{self.ag.n_atoms} atoms"
#             )
#         return np.bincount(
#             self._resindex_inverse, weights=areas,
#             minlength=len(self._unique_resindices),
#         )

#     def _build_structure(self):
#         """
#         Build a freesasa.Structure directly from current coordinates,
#         without writing/reparsing a PDB file each frame.
#         """
#         structure = freesasa.Structure()
#         positions = self.ag.positions
#         names, resnames = self._atom_names, self._atom_resnames
#         resids, chains = self._atom_resids, self._atom_chains

#         for i in range(self.ag.n_atoms):
#             x, y, z = positions[i]
#             structure.addAtom(
#                 names[i], resnames[i], resids[i], chains[i],
#                 float(x), float(y), float(z),
#             )
#         return structure

#     def _compute_freesasa_result(self):
#         structure = self._build_structure()
#         return freesasa.calc(structure, self.params)

#     def _calc_from_coords(self):
#         # NOTE: if using _calc_from_coords(), residue areas are not available
#         # in the result object
#         coords = self.ag.positions.flatten()
#         return freesasa.calcCoord(coords, self._radii, self.params)

#     def _check_residue_shapes(self):
#         # make sure the residues from freesasa and mdanalysis match size and
#         # type
#         result = self._compute_freesasa_result()

#         fsa_res_types = np.array([sublist[item].residueType for sublist in
#                                   result.residueAreas().values() for item in
#                                   sublist])
#         mda_res_types = self.ag.residues.resnames

#         return fsa_res_types.shape == mda_res_types.shape

# import freesasa
# import MDAnalysis as mda
# import numpy as np
# from MDAnalysis.analysis.base import AnalysisBase, Results
# from MDAnalysis.analysis.results import ResultsGroup

# freesasa.setVerbosity(1)


# class FreesasaAnalysis(AnalysisBase):
#     """
#     Run FreeSASA and parse results
#     """

#     _analysis_algorithm_is_parallelizable = True

#     @classmethod
#     def get_supported_backends(cls):
#         return ('serial', 'multiprocessing', 'dask')

#     def __init__(self, ag, probe_radius):
#         super().__init__(ag.universe.trajectory)
#         self.ag = ag
#         self.probe_radius = probe_radius

#     def _prepare(self):
#         # This runs independently in EVERY worker process (each worker
#         # gets its own copy of the instance), so freesasa.Parameters --
#         # which is unlikely to survive pickling cleanly -- gets rebuilt
#         # fresh here rather than in __init__.
#         self.params = freesasa.Parameters()
#         self.params.setProbeRadius(self.probe_radius)
#         self.params.setNThreads(1)

#         self._atom_names = self.ag.atoms.names
#         self._atom_resnames = self.ag.atoms.resnames
#         self._atom_resids = [str(r) for r in self.ag.atoms.resids]
#         try:
#             self._atom_chains = [str(c) for c in self.ag.atoms.chainIDs]
#         except (AttributeError, mda.exceptions.NoDataError):
#             self._atom_chains = ['A'] * self.ag.n_atoms

#         if self._check_residue_shapes():
#             self.structure_valid = True
#         else:
#             self.structure_valid = False
#             from MDAnalysis.guesser.tables import vdwradii
#             self._radii = np.array(
#                 [vdwradii.get(elem) for elem in self.ag.atoms.types]
#             )

#         self._unique_resindices, self._resindex_inverse = np.unique(
#             self.ag.resindices, return_inverse=True
#         )

#         # self.n_frames here is scoped to THIS WORKER's assigned frame
#         # range, not the whole trajectory -- that's how AnalysisBase's
#         # parallel framework splits work, and it's why sizing arrays off
#         # self.n_frames (rather than len(self.ag.universe.trajectory))
#         # is required for this to work correctly under any backend.
#         self.results = Results()
#         self.results.atom_areas = np.zeros([self.n_frames, self.ag.n_atoms])
#         self.results.residue_areas = np.zeros([self.n_frames, self.ag.n_residues])
#         self.results.total_area = np.zeros([self.n_frames])

#     def _single_frame(self):
#         if self.structure_valid:
#             result = self._compute_freesasa_result()
#         else:
#             result = self._calc_from_coords()

#         total_area = result.totalArea()
#         n_atoms = result.nAtoms()
#         atom_areas = np.fromiter(
#             (result.atomArea(i) for i in range(n_atoms)),
#             dtype=float, count=n_atoms,
#         )

#         if self.structure_valid:
#             residue_areas = [sublist[item].total for sublist in
#                               result.residueAreas().values() for item in sublist]
#         else:
#             residue_areas = self.sum_areas_per_residue(atom_areas)

#         # self._frame_index is LOCAL to this worker's chunk (0-based
#         # within its own slice), which is exactly what indexing into
#         # the per-worker arrays above requires.
#         self.results.atom_areas[self._frame_index] = atom_areas
#         self.results.residue_areas[self._frame_index] = residue_areas
#         self.results.total_area[self._frame_index] = total_area

#     def _get_aggregator(self):
#         # Each worker returns a chunk of consecutive frames; concatenating
#         # along axis 0 in the order the workers ran reassembles the full
#         # trajectory-length array, since MDAnalysis's runner assigns and
#         # merges frame chunks in order.
#         concat = staticmethod(lambda arrs: np.concatenate(arrs, axis=0))
#         return ResultsGroup(lookup={
#             'atom_areas': concat.__func__,
#             'residue_areas': concat.__func__,
#             'total_area': concat.__func__,
#         })

#     def sum_areas_per_residue(self, areas):
#         if len(areas) != self.ag.n_atoms:
#             raise ValueError(
#                 f"areas has {len(areas)} entries but atomgroup has "
#                 f"{self.ag.n_atoms} atoms"
#             )
#         return np.bincount(
#             self._resindex_inverse, weights=areas,
#             minlength=len(self._unique_resindices),
#         )

#     def _build_structure(self):
#         structure = freesasa.Structure()
#         positions = self.ag.positions
#         names, resnames = self._atom_names, self._atom_resnames
#         resids, chains = self._atom_resids, self._atom_chains

#         for i in range(self.ag.n_atoms):
#             x, y, z = positions[i]
#             structure.addAtom(
#                 names[i], resnames[i], resids[i], chains[i],
#                 float(x), float(y), float(z),
#             )
#         return structure

#     def _compute_freesasa_result(self):
#         structure = self._build_structure()
#         return freesasa.calc(structure, self.params)

#     def _calc_from_coords(self):
#         coords = self.ag.positions.flatten()
#         return freesasa.calcCoord(coords, self._radii, self.params)

#     def _check_residue_shapes(self):
#         result = self._compute_freesasa_result()
#         fsa_res_types = np.array([sublist[item].residueType for sublist in
#                                   result.residueAreas().values() for item in
#                                   sublist])
#         mda_res_types = self.ag.residues.resnames
#         return fsa_res_types.shape == mda_res_types.shape


import freesasa
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup

freesasa.setVerbosity(1)


class FreesasaAnalysis(AnalysisBase):
    """
    Run FreeSASA and parse results
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ('serial', 'multiprocessing', 'dask')

    def __init__(self, ag, probe_radius):
        super().__init__(ag.universe.trajectory)
        self.ag = ag
        self.probe_radius = probe_radius

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
            self.params = freesasa.Parameters()
            self.params.setProbeRadius(self.probe_radius)
            self.params.setNThreads(1)

    def _prepare(self):
        self.params = freesasa.Parameters()
        self.params.setProbeRadius(self.probe_radius)
        self.params.setNThreads(1)

        self._atom_names = self.ag.atoms.names
        self._atom_resnames = self.ag.atoms.resnames
        self._atom_resids = [str(r) for r in self.ag.atoms.resids]
        try:
            self._atom_chains = [str(c) for c in self.ag.atoms.chainIDs]
        except (AttributeError, mda.exceptions.NoDataError):
            self._atom_chains = ['A'] * self.ag.n_atoms

        if self._check_residue_shapes():
            self.structure_valid = True
        else:
            self.structure_valid = False
            from MDAnalysis.guesser.tables import vdwradii
            self._radii = np.array(
                [vdwradii.get(elem) for elem in self.ag.atoms.types]
            )

        self._unique_resindices, self._resindex_inverse = np.unique(
            self.ag.resindices, return_inverse=True
        )

        self.results = Results()
        self.results.atom_areas = np.zeros([self.n_frames, self.ag.n_atoms])
        self.results.residue_areas = np.zeros([self.n_frames, self.ag.n_residues])
        self.results.total_area = np.zeros([self.n_frames])

    def _single_frame(self):
        if self.structure_valid:
            result = self._compute_freesasa_result()
        else:
            result = self._calc_from_coords()

        total_area = result.totalArea()
        n_atoms = result.nAtoms()
        atom_areas = np.fromiter(
            (result.atomArea(i) for i in range(n_atoms)),
            dtype=float, count=n_atoms,
        )

        if self.structure_valid:
            residue_areas = [sublist[item].total for sublist in
                              result.residueAreas().values() for item in sublist]
        else:
            residue_areas = self.sum_areas_per_residue(atom_areas)

        self.results.atom_areas[self._frame_index] = atom_areas
        self.results.residue_areas[self._frame_index] = residue_areas
        self.results.total_area[self._frame_index] = total_area

    def _get_aggregator(self):
        concat = staticmethod(lambda arrs: np.concatenate(arrs, axis=0))
        return ResultsGroup(lookup={
            'atom_areas': concat.__func__,
            'residue_areas': concat.__func__,
            'total_area': concat.__func__,
        })

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
        names, resnames = self._atom_names, self._atom_resnames
        resids, chains = self._atom_resids, self._atom_chains

        for i in range(self.ag.n_atoms):
            x, y, z = positions[i]
            structure.addAtom(
                names[i], resnames[i], resids[i], chains[i],
                float(x), float(y), float(z),
            )
        return structure

    def _compute_freesasa_result(self):
        structure = self._build_structure()
        return freesasa.calc(structure, self.params)

    def _calc_from_coords(self):
        coords = self.ag.positions.flatten()
        return freesasa.calcCoord(coords, self._radii, self.params)

    def _check_residue_shapes(self):
        result = self._compute_freesasa_result()
        fsa_res_types = np.array([sublist[item].residueType for sublist in
                                  result.residueAreas().values() for item in
                                  sublist])
        mda_res_types = self.ag.residues.resnames
        return fsa_res_types.shape == mda_res_types.shape