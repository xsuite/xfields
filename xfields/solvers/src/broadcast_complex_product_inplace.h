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
    uint64_t n2_big,
    uint64_t nn
) {
    for (uint64_t ii = 0; ii < nn; ii++){ //vectorize_over ii nn
       uint64_t i2 = ii / (n0_big * n1_big);
       uint64_t i1 = (ii - i2 * n0_big * n1_big) / n0_big;
       uint64_t i0 = ii - i2 * n0_big * n1_big - i1 * n0_big;

       double* big_ptr = big + 2 * (i2 * n1_big * n0_big + i1 * n0_big + i0);
       double* small_ptr = small + 2 * (i1 * n0_big + i0);
       if (i2 == 15 && i1 == 121 && i0 == 120){
           printf("big_ptr: %e %e\n", big_ptr[0], big_ptr[1]);
           printf("small_ptr: %e %e\n", small_ptr[0], small_ptr[1]);
       }
    //    complex_multiply_inplace(big_ptr, small_ptr);
       double a0 = big_ptr[0];
       double a1 = big_ptr[1];
       double b0 = small_ptr[0];
       double b1 = small_ptr[1];
       double res0 = a0 * b0 - a1 * b1;
       double res1 = a0 * b1 + a1 * b0;
       big_ptr[0] = res0;
       big_ptr[1] = res1;
       if (i2 == 15 && i1 == 121 && i0 == 120){
           printf("res: %e %e\n", res0, res1);
           printf("big_ptr: %e %e\n", big_ptr[0], big_ptr[1]);
       }
    }//end_vectorize
}

#endif // BROADCAST_COMPLEX_PRODUCT_INPLACE_H