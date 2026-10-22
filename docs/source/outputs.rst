Output files
============

All files are written to ``--outdir``.

============================= ==================================================
file                          contents
============================= ==================================================
``residue_summary.csv``       ``analyzed_frame, trajectory_frame, chain, resid,
                              resname, glycan_count, sasa`` for every residue
                              with a nonzero count (``--bfactor-frame``)
``frame_summary.csv``         one row per analyzed frame: surface residues,
                              blocked pairs, mean blocked distance
``b_factors.pdb``             structure with each residue's count as B-factor
                              (0 for all other atoms)
``b_factors.npy``             counts of every analyzed frame,
                              shape (frames, residues)
``run_config.json``           mode, blocking test, parameters, frames, versions
``excluded_glycan_atoms.csv`` with ``--write-excluded``
``blockers.csv``              with ``--write-blockers`` (not with ``--mode v1``)
``load_user_attributes.vmd``  with ``--write-vmd``
``mdanalysis_warnings.log``   all warnings of the run
============================= ==================================================