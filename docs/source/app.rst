Web app (Streamlit)
===================

The glyco web app is a graphical front end for ``glyco run``. You upload a
structure (and optionally trajectories), check the atom selections, choose
frames and analysis parameters, run the analysis and download the results
as a ZIP file. The app builds and runs the same ``glyco run`` command you
would type on the command line, so results are identical.

Installation
------------

The app needs Streamlit in addition to glycokit:

.. code-block:: bash

   pip install streamlit

Starting the app
----------------

.. code-block:: bash

   streamlit run glycokit/app.py

A local server starts and the app opens in a new browser tab. Stop the
server with :kbd:`Ctrl+C` in the terminal.

If the ``streamlit`` command is not on your ``PATH`` (or you configure an
IDE), run it as a Python module instead; this is equivalent:

.. code-block:: bash

   python -m streamlit run glycokit/app.py

Options for Streamlit itself (for example ``--server.port 8502``) go before
the script name. Arguments for the script must follow two dashes
(``streamlit run app.py -- <args>``); the glyco app does not take any.

Workflow
--------

The app is organized in six numbered sections, from top to bottom.

1. Load structure and trajectories
   Upload a topology / structure file (any format MDAnalysis can read) and,
   optionally, one or more trajectory files, then click **Load Universe**.
   Several trajectories are concatenated in upload order and must all match
   the atom count of the topology. Loading a new Universe clears the results
   of the previous run and revalidates the selections.

2. Universe information
   Number of atoms, residues, segments and frames; for trajectories also the
   time step and total time.

3. Atom selections
   MDAnalysis selections for the protein (default ``protein``; every residue
   needs one CA atom) and the glycan atoms (for example
   ``resname NAG BMA MAN``). Selections are checked when you leave the text
   box: each must be valid and select at least one atom. The run button stays
   disabled until both are valid.

4. Trajectory options
   Either **Start / Stop / Step** (blank fields mean the whole trajectory;
   a negative stop counts from the end, as in Python slicing) or an explicit
   **Frame list** (frame numbers separated by commas, spaces or new lines).
   The number of selected frames is shown below the inputs.

5. Analysis parameters
   All other ``glyco run`` options, generated directly from the command-line
   parser, so the app always offers the same options as the CLI; see
   :doc:`cli/run` for their meaning. Flags appear as checkboxes, choices as
   drop-down lists and numbers as number fields with the lower bound of the
   option (for example ``--dc`` must be positive). Optional numbers such as
   ``--bfactor-frame`` are text fields: leave them blank to use the program
   default. Invalid entries are listed in red and disable the run button.

6. Run analysis
   **Run Analysis** shows a progress bar while glyco runs. Afterwards the app
   shows the exact command that was run and the program output (both in
   collapsible panels) and offers **Download results (ZIP)** with every output
   file of the run (see :doc:`outputs`). The command can be copied and run
   on the command line to reproduce the results.

Inputs handled by the app
-------------------------

The following options are set by sections 1-4 or by the app itself and are
therefore not shown in section 5: ``--top``, ``--traj``, ``--indir``,
``--in-types``, ``--prot-sel``, ``--gly-sel``, the frame options
(``--start``, ``--stop``, ``--step``, ``--frames``), ``--outdir``,
``--force``, ``--verbose`` and ``--color``. Each run writes to a new
temporary output directory, which is packed into the ZIP download.
``--indir`` (batches of independent structures) is only available on the
command line.

Limits and notes
----------------

* **Upload size.** Streamlit limits each uploaded file to 200 MB by default.
  For larger trajectories raise ``server.maxUploadSize`` (in MB) in
  ``.streamlit/config.toml``:

  .. code-block:: toml

     [server]
     maxUploadSize = 2000

  or start the app with ``streamlit run glycokit/app.py
  --server.maxUploadSize 2000``. For very large trajectories, the command
  line is usually the better choice, because the files do not need to be
  uploaded.

* **Temporary files.** Uploaded files and output directories are stored in
  the system temporary directory. Output directories are removed when the
  Streamlit server stops; download the ZIP before closing the server.

* **Invalid mode combinations.** Some option combinations (for example
  ``--mode v1`` with ``--blocking vdw``) are only rejected by glyco itself.
  The run then fails, and the reason is shown in the analysis output.

* **Plain output.** The app sets ``GLYCO_STREAMLIT=1``, which disables
  colored terminal output and glyco's own progress bars; the app reads the
  progress from ``GLYCO_PROGRESS_FILE`` instead.

* **Remote development environments.** If the app hangs at "Please wait..."
  in GitHub Codespaces, make sure the forwarded port (8501 by default) is
  reachable; starting with ``--server.enableCORS false
  --server.enableXsrfProtection false`` is a known workaround for local
  development only.

Troubleshooting
---------------

``Failed to load Universe``
   MDAnalysis could not read the files: check the file formats and, for
   several trajectories, that they all match the topology's atom count.

``Input file(s) no longer exist``
   The temporary upload files were removed (for example after a server
   restart). Load the Universe again.

The run fails immediately
   Open **Analysis Output**: glyco prints the reason (for example a missing
   glycan selection or an invalid option combination), exactly as on the
   command line.