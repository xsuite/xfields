// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2023.                   //
// ########################################### //

#ifndef BROADCAST_COMPLEX_PRODUCT_INPLACE_H
#define BROADCAST_COMPLEX_PRODUCT_INPLACE_H

/*gpufun*/ void complex_multiply_inplace(double* a, double* b)
{
   double res_real, res_imag;
   res_real = a[0] * b[0] - a[1] * b[1];
   res_imag = a[0] * b[1] + a[1] * b[0];
   a[0] = res_real;
   a[1] = res_imag;
}

/*gpukern*/ void broadcast_complex_product_inplace(
    /*gpuglmem*/ double *big,
    /*gpuglmem*/ double *small,
    uint64_t n0_big,
    uint64_t n1_big,
    uint64_t n2_big
) {
    uint64_t nn = n0_big * n1_big * n2_big;
    for (uint64_t ii = 0; ii < nn; ii++){ //vectorize_over ii nn
       uint64_t i2 = ii / (n0_big * n1_big);
       uint64_t i1 = (ii - i2 * n0_big * n1_big) / n0_big;
       uint64_t i0 = ii - i2 * n0_big * n1_big - i1 * n0_big;

       double* big_ptr = big + 2 * (i2 * n1_big * n0_big + i1 * n0_big + i0);
       double* small_ptr = small + 2 * (i1 * n0_big + i0);
       complex_multiply_inplace(big_ptr, small_ptr);
    }//end_vectorize
}

#endif // BROADCAST_COMPLEX_PRODUCT_INPLACE_H