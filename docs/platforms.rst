Platforms
=========

.. TODO: explain default kernels.

Xfields supports different plaforms allowing the exploitation of different kinds of hardware (CPUs and GPUs).
A platform is initialized by instanciating objects from one of the platform classes, which is then passed to the other Xfields components.
Platforms are interchangeable as they expose the same API.

Three platforms are presently available:

 - The :ref:`Cupy platform<cupy_platform>`, based on `cupy`_-`cuda`_ to run on NVidia GPUs
 - The :ref:`Pocl platform<pocl_platform>`, bases on `PyOpenCL`_, to run on CPUs or GPUs throught PyOPENCL library.
 - The :ref:`CPU platform<cpu_platform>`, to use conventional CPUs

The corresponfig API is described in the following subsections.

.. _cupy: https://cupy.dev
.. _cuda: https://developer.nvidia.com/cuda-zone
.. _PyOpenCL: https://documen.tician.de/pyopencl/


.. _cupy_platform:

Cupy platform
-------------

.. autoclass:: xfields.platforms.XfCupyPlatform
    :members:
    :undoc-members:
    :member-order: bysource

.. _pocl_platform:

PyOpenCL platform
-----------------
.. autoclass:: xfields.platforms.XfPoclPlatform
    :members:
    :undoc-members:
    :member-order: bysource


.. _cpu_platform:

CPU platform
------------

.. autoclass:: xfields.platforms.XfCpuPlatform
    :members:
    :undoc-members:
    :member-order: bysource