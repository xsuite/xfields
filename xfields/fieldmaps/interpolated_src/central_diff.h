#ifndef CENTRAL_DIFF_H
#define CENTRAL_DIFF_H

/*gpukern*/
void central_diff(
	      const int     nelem,
	      const int     row_size,
	      const int     stride_in_dbl,
	      const double  factor,
/*gpuglmem*/  const double* matrix,
/*gpuglmem*/        double* res){

   for(int ii=0; ii<nelem; ii++){//vectorize_over ii nelem
      if (ii-stride_in_dbl>=0 && ii+stride_in_dbl<nelem){
         res[ii] = factor * (matrix[ii+stride_in_dbl]
			   - matrix[ii-stride_in_dbl]);
      } 
      int place_in_row = (ii/stride_in_dbl)%row_size;
      if (place_in_row==0 || place_in_row==row_size-1){
         res[ii] = 0;
      }
   }//end_vectorize 

}

#endif
