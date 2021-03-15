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

In order to use the :doc:`pocl platform<platforms>`, the PyOpenCL package needs to be installed.
In Anacoda or Miniconda this can be done as follows:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install pyopencl


Check that there is an OpenCL installation in the system:

.. code-block:: bash

    $ ls /etc/OpenCL/vendors


Make the OpenCL installation visible to pyopencl:

.. code-block:: bash

    $ conda install ocl-icd-system


For the pocl platform we will need the `gpyfft <https://github.com/geggo/gpyfft>`_ library.
For this we need to install cython.

.. code-block:: bash

    $ pip install cython


The we install clfft.

.. code-block:: bash

    $ conda install -c conda-forge clfft


We locate the library and headers here:

.. code-block:: bash

    $ ls ~/miniconda3/pkgs/clfft-2.12.2-h83d4a3d_1/
    # gives: include  info  lib



We install gpyffe install pip providing extra flags as follows:

.. code-block:: bash

    $ git clone https://github.com/geggo/gpyfft
    $ pip install --global-option=build_ext --global-option="-I/home/giadarol/miniconda3/pkgs/clfft-2.12.2-h83d4a3d_1/include" --global-option="-L/home/giadarol/miniconda3/pkgs/clfft-2.12.2-h83d4a3d_1/lib" gpyfft/




