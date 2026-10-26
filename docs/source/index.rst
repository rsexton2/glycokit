.. glycokit documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GlyCoKit's documentation!
=========================================================

**glyco** computes the glycan coverage of protein surface residues from a
structure, a trajectory or a directory of independent structures. FreeSASA
selects the surface residues; a line-of-sight test counts the glycan atoms
each residue can see.

It also reproduces the original GLYCO script (``--mode v1``) and GLYCO-2
(``--mode v2``) for comparison with published results.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   app
   modes
   outputs

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli/index
   api