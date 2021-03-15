.. _installation-page:

Installation
============


The package can be cloned from GitHub and installed with pip:

.. code-block:: bash

    $ git clone https://github.com/xsuite/xfields
    $ pip install -e xfields

(The installation without the ``-e`` option is still untested).


Installation of cupy
--------------------

In order to use the :doc:`cupy platform<platforms>`, the cupy package needs to be installed.
In Anacoda or Miniconda this can be done as follows:

.. code-block:: bash

    $ conda install mamba -n base -c conda-forge
    $ pip install cupy-cuda101
    $ mamba install cudatoolkit=10.1.243

Installation of PyOpenCL
------------------------

