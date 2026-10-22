Quick start
===========

Run "glyco" to list all available subcommands. Running "glyco <command> --help"
will show the options of a command.

1. OPTIONAL: Standardize the input, so every residue has a unique label and
   glycans have their own chains. If the PDB file does not contain any duplicate
   atom identifiers, this step is not necessary. The standardized file will be
   written to ``model_std.pdb``:

   .. code-block:: bash

      glyco standardize model.pdb            # writes model_std.pdb

2. Run the analysis:

   .. code-block:: bash

      glyco run -a model_std.pdb --gs "resname NAG BMA MAN" -o results

   With a trajectory:

   .. code-block:: bash

      glyco run -a top.pdb -b run1.xtc run2.xtc --gs "resname NAG BMA" -j -1

3. Open ``results/residue_summary.csv``, or load ``results/b_factors.pdb``
   in a viewer and colour by B-factor.

From Python:

.. code-block:: python

   import MDAnalysis as mda
   from glycokit.glyco import GlycoAnalysis

   u = mda.Universe("model_std.pdb")
   analysis = GlycoAnalysis(u, gly_sel="resname NAG BMA MAN", outdir="results")
   analysis.run(write=False)          # results only, no files
   counts = analysis.results.glycan_counts