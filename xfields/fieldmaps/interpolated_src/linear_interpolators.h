#ifndef XFIELDS_LINEAR_INTERPOLATORS_H
#define XFIELDS_LINEAR_INTERPOLATORS_H

typedef struct{
    int64_t ix;
    int64_t iy;
    int64_t iz;
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

    	// indices
    	iw.ix = floor((x - x0) / dx);
    	iw.iy = floor((y - y0) / dy);
    	iw.iz = floor((z - z0) / dz);

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

	return iw;

}	

#endif
