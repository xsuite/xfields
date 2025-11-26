// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef CENTRAL_DIFF_H
#define CENTRAL_DIFF_H

#include "xobjects/headers/common.h"


GPUKERN
void central_diff(
    const int nelem,
    const int row_size,
    const int stride_in_dbl,
    const double factor,
    GPUGLMEM const int8_t* matrix_buffer,
    const int64_t matrix_offset,
    GPUGLMEM int8_t* res_buffer,
    int64_t res_offset
) {
	GPUGLMEM const double* matrix =
	    (GPUGLMEM double*) (matrix_buffer + matrix_offset);
	GPUGLMEM       double*  res =
	    (GPUGLMEM double*) (res_buffer + res_offset);

	VECTORIZE_OVER(ii, nelem);
		if (ii-stride_in_dbl>=0 && ii+stride_in_dbl<nelem) {
			res[ii] = factor * (matrix[ii+stride_in_dbl]
			                    - matrix[ii-stride_in_dbl]);
		}
		int place_in_row = (ii/stride_in_dbl)%row_size;
		if (place_in_row==0 || place_in_row==row_size-1) {
			res[ii] = 0;
		}
	END_VECTORIZE;

}

#endif
