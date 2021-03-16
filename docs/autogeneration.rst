Code autogeneration
===================

This library uses code autogeneration to specialize kernel code for the different platforms.

Syntax
------

The developer writes a single C source code, providing additional information through the following comment strings, as described in the following.

``vectorize_over`` block
~~~~~~~~~~~~~~~~~~~~~~~~

The syntax is the following:

.. code-block:: C

    int myvar = 0; //vectorize_over myvar myvarlim

        [MY CODE]

        //end_vectorize

This is translated into a for loop in the CPU implementation and in a kernel function for the parallel implementations (cupy, pyopencl).

The generated cpu code will be:

.. code-block:: C

    int myvar; //autovectorized
    for (myvar=0; myvar<myvarlim; myvar++){ //autovectorized

        [MY CODE]

        }//end autovectorized

The generated CUDA code will be:

.. code-block:: C

    int myvar; //autovectorized
    myvar = blockDim.x * blockIdx.x + threadIdx.x;//autovectorized

        [MY CODE]

    //end autovectorized

The corresponding generated OpenCL code will be:

.. code-block:: C

    int myvar; //autovectorized
    myvar = get_global_id(0); //autovectorized

        [MY CODE]

    //end autovectorized






