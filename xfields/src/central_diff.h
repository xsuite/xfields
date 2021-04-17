#ifndef CENTRAL_DIFF_H
#define CENTRAL_DIFF_H

/*gpukern*/
void central_diff(
	      const int     nelem,
	      const int     stride_in_dbl,
	      const double  factor,
/*gpuglmem*/  const double* matrix,
/*gpuglmem*/        double* res){

   for(int ii=0; ii<nelem; ii++){//vectorize_over ii nelem
      if (ii>0 && ii<nelem-1){
         res[ii] = factor * (matrix[ii+stride_in_dbl]
			   - matrix[ii-stride_in_dbl]);
      } 
   } 

}

#endif
