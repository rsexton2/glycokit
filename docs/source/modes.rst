Counting modes
==============

``--mode`` selects the method; ``--blocking`` selects the blocking test
in the default mode.

=============  ==================  ==========================================
mode           blocking            description
=============  ==================  ==========================================
``default``    ``vdw`` (default)   van der Waals spheres, float64
``default``    ``cylinder``        cylinder of radius ``--cr``, float64
``v1``         box (fixed)         reproduces the original GLYCO script
``v2``         cylinder (fixed)    reproduces GLYCO-2 (float32 arithmetic)
=============  ==================  ==========================================

Workflow
--------

.. mermaid::

   flowchart TD
       A["glyco run"] --> B["select surface residues (FreeSASA, --ac)"]
       B --> C{"--mode"}
       C -->|default| D["float64 pairs below --dc"]
       C -->|v2| E["float32 pairs below --dc"]
       C -->|v1| F["pairs below int(--dc)"]
       D --> G{"--blocking"}
       G -->|vdw| H["segment touches vdW sphere"]
       G -->|cylinder| I["blocker within --cr of segment"]
       E --> J["cylinder, start residue can block"]
       F --> K["+-1 A box at sampled points"]
       H --> L["per-residue union of visible glycan atoms"]
       I --> L
       J --> L
       K --> L

(Paste the full combined chart and the "differences at a glance" table here.)