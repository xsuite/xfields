// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_LINEAR_INTERPOLATORS_H
#define XFIELDS_LINEAR_INTERPOLATORS_H

typedef struct{
    int64_t ix;
    int64_t iy;
    int64_t iz;
    int64_t nx;
    int64_t ny;
    int64_t nz;
    double w000;
    double w100;
    double w010;
    double w110;
    double w001;
    double w101;
    double w011;
    double w111;
}IndicesAndWeights;


/*gpufun*/
IndicesAndWeights TriLinearInterpolatedFieldMap_compute_indeces_and_weights(
	TriLinearInterpolatedFieldMapData fmap,
	double x, double y, double z){

	IndicesAndWeights iw;

	const double dx = TriLinearInterpolatedFieldMapData_get_dx(fmap);
	const double dy = TriLinearInterpolatedFieldMapData_get_dy(fmap);
	const double dz = TriLinearInterpolatedFieldMapData_get_dz(fmap);
	const double x0 = TriLinearInterpolatedFieldMapData_get_x_min(fmap);
	const double y0 = TriLinearInterpolatedFieldMapData_get_y_min(fmap);
	const double z0 = TriLinearInterpolatedFieldMapData_get_z_min(fmap);
	const int64_t nx = TriLinearInterpolatedFieldMapData_get_nx(fmap);
	const int64_t ny = TriLinearInterpolatedFieldMapData_get_ny(fmap);
	const int64_t nz = TriLinearInterpolatedFieldMapData_get_nz(fmap);

    	iw.nx = nx;
    	iw.ny = ny;
    	iw.nz = nz;

    	// indices
    	iw.ix = floor((x - x0) / dx);
    	iw.iy = floor((y - y0) / dy);
    	iw.iz = floor((z - z0) / dz);

	
    	if (iw.ix >= 0 && iw.ix < nx - 1 && iw.iy >= 0 && iw.iy < ny - 1
	    	    && iw.iz >= 0 && iw.iz < nz - 1){

    	    // distances
    	    const double dxi = x - (x0 + iw.ix * dx);
    	    const double dyi = y - (y0 + iw.iy * dy);
    	    const double dzi = z - (z0 + iw.iz * dz);
	    
    	    // weights
    	    iw.w000 = (1.-dxi/dx) * (1.-dyi/dy) * (1.-dzi/dz);
    	    iw.w100 = (dxi/dx)    * (1.-dyi/dy) * (1.-dzi/dz);
    	    iw.w010 = (1.-dxi/dx) * (dyi/dy)    * (1.-dzi/dz);
    	    iw.w110 = (dxi/dx)    * (dyi/dy)    * (1.-dzi/dz);
    	    iw.w001 = (1.-dxi/dx) * (1.-dyi/dy) * (dzi/dz);
    	    iw.w101 = (dxi/dx)    * (1.-dyi/dy) * (dzi/dz);
    	    iw.w011 = (1.-dxi/dx) * (dyi/dy)    * (dzi/dz);
    	    iw.w111 = (dxi/dx)    * (dyi/dy)    * (dzi/dz);
	}
	else{
            iw.ix = -999; 
            iw.iy = -999; 
            iw.iz = -999; 
	}
	return iw;

}	

/*gpufun*/
double TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(
	/*gpuglmem*/ const double* map,
	   const IndicesAndWeights iw){
	
    double val;

    if (iw.ix < 0){
	 val = 0.;
    }
    else{
	val = 
    	       iw.w000 * map[iw.ix   + (iw.iy  ) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    	     + iw.w100 * map[iw.ix+1 + (iw.iy  ) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    	     + iw.w010 * map[iw.ix+  + (iw.iy+1) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    	     + iw.w110 * map[iw.ix+1 + (iw.iy+1) * iw.nx + (iw.iz  ) * iw.nx * iw.ny]
    	     + iw.w001 * map[iw.ix   + (iw.iy  ) * iw.nx + (iw.iz+1) * iw.nx * iw.ny]
    	     + iw.w101 * map[iw.ix+1 + (iw.iy  ) * iw.nx + (iw.iz+1) * iw.nx * iw.ny]
    	     + iw.w011 * map[iw.ix+  + (iw.iy+1) * iw.nx + (iw.iz+1) * iw.nx * iw.ny]
    	     + iw.w111 * map[iw.ix+1 + (iw.iy+1) * iw.nx + (iw.iz+1) * iw.nx * iw.ny];
    }

    return val;
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
