#ifndef XFIELDS_CUBIC_INTERPOLATORS_H
#define XFIELDS_CUBIC_INTERPOLATORS_H

#include "tricubic_coefficients.h"

/*gpufun*/
double TriCubicInterpolatedFieldMap_interpolate_3d_derivatives(
	TricubicInterpolatedFieldMapData fmap,
	   const double x, const double y, const double z, 
	   double& px_kick, double& py_kick, double& ptau_kick){
	
	px_kick=1;
	return;
    // double val;

    // if (iw.ix < 0){
	 va// l = 0.;
    // }
    // else{
	val//  = 
    // 	       iw.w000 * map[iw.ix   + (iw.iy  ) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    // 	     + iw.w100 * map[iw.ix+1 + (iw.iy  ) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    // 	     + iw.w010 * map[iw.ix+  + (iw.iy+1) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    // 	     + iw.w110 * map[iw.ix+1 + (iw.iy+1) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    // 	     + iw.w001 * map[iw.ix   + (iw.iy  ) * iw.nx + (iw.iz+1) * iw.nx * iw.ny]
    // 	     + iw.w101 * map[iw.ix+1 + (iw.iy  ) * iw.nx + (iw.iz+1) * iw.nx * iw.ny]
    // 	     + iw.w011 * map[iw.ix+  + (iw.iy+1) * iw.nx + (iw.iz+1) * iw.nx * iw.ny]
    // 	     + iw.w111 * map[iw.ix+1 + (iw.iy+1) * iw.nx + (iw.iz+1) * iw.nx * iw.ny];
    // }

    // return val;
}

/*gpukern*/
void TriLinearInterpolatedFieldMap_interpolate_3d_map_vector(
    TriLinearInterpolatedFieldMapData  fmap,
                        const int64_t  n_points,
           /*gpuglmem*/ const double*  x,
           /*gpuglmem*/ const double*  y,
           /*gpuglmem*/ const double*  z,
                        const int64_t  n_quantities,
           /*gpuglmem*/ const int8_t*  buffer_mesh_quantities,
           /*gpuglmem*/ const int64_t* offsets_mesh_quantities,
           /*gpuglmem*/       double*  particles_quantities) {

    #pragma omp parallel for //only_for_context cpu_openmp 
    for (int pidx=0; pidx<n_points; pidx++){ //vectorize_over pidx n_points

	const IndicesAndWeights iw = 
		TriLinearInterpolatedFieldMap_compute_indeces_and_weights(
	                                      fmap, x[pidx], y[pidx], z[pidx]);
    	for (int iq=0; iq<n_quantities; iq++){
	    particles_quantities[iq*n_points + pidx] = 
		TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(
	           (/*gpuglmem*/ double*)(buffer_mesh_quantities + offsets_mesh_quantities[iq]),
		   iw);
	}
    }//end_vectorize
}
#endif
