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

## Install pyopencl
In order to use the :doc:`pocl platform<platforms>`, the PyOpenCL package needs to be installed.
In Anacoda or Miniconda this can be done as follows:

```
conda config --add channels conda-forge
conda install pyopencl
```

Check that there is an OpenCL installation in the system:
```
ls /etc/OpenCL/vendors
```

Make the OpenCL installation visible to pyopencl:
```bash
conda install ocl-icd-system
```