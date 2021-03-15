Xfields platforms
=================

TODO: explain default kernels.

Xfields supports different plaforms allowing the exploitation of different kinds of hardware (CPUs and GPUs).
A platform is initialized by instanciating objects from one of the platform classes, which is then passed to the other Xfields components.

Three platforms are presently available:
 - The :ref:`CPU platform<cpu_platform>`, to use conventional CPUs
 - The :ref:`Cupy platform<cupy_platform>`, based on `cupy`_-`cuda` to run on NVidia GPUs
 - The :ref:`Pocl platform<pocl_platform>`, bases on `PyOpenCL_, to run on CPUs or GPUs throught PyOPENCL library.

.. _cupy: https://cupy.dev
.. _cuda: https://developer.nvidia.com/cuda-zone
.. _PyOpenCL: https://documen.tician.de/pyopencl/

.. _cpu_platform:

Xfields CPU platform
--------------------

.. automodule:: xfields.platforms.XfCpuPlatform
    :members:
    :undoc-members:
    :member-order: bysource

.. _cupy_platform:

Xfields cupy platform
---------------------

.. automodule:: xfields.platforms.XfCupyPlatform
    :members:
    :undoc-members:
    :member-order: bysource

.. _pocl_platform:

Xfields PyOpenCL platform
-------------------------
.. automodule:: xfields.platforms.XfPoclPlatform
    :members:
    :undoc-members:
    :member-order: bysource

