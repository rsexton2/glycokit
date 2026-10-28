Installation
============

To build GlyCoKit from source, we highly recommend using virtual
environments. If possible, we strongly recommend that you use
`Anaconda <https://docs.conda.io/en/latest/>`__ as your package manager.
Below we provide instructions both for ``conda`` and for ``pip``.

With conda
----------

Ensure that you have
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__
installed.

Create a virtual environment and activate it:

.. code-block:: bash

   conda create --name glycokit
   conda activate glycokit

Clone this repository and change into the cloned directory:

.. code-block:: bash

   git clone github.com/rsexton/glycokit.git
   cd glycokit

Install the development and documentation dependencies:

.. code-block:: bash

   conda env update --name glycokit --file devtools/conda-envs/test_env.yaml
   conda env update --name glycokit --file requirements.yaml

Build this package from source:

.. code-block:: bash

   pip install -e .

If you want to update your dependencies (which can be risky!), run:

.. code-block:: bash

   conda update --all

And when you are finished, you can exit the virtual environment with:

.. code-block:: bash

   conda deactivate

With pip
--------

To build the package from source, run:

.. code-block:: bash

   pip install .

If you want to create a development environment, install the dependencies
required for tests and docs with:

.. code-block:: bash

   pip install ".[test,doc]"