// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_CUBIC_INTERPOLATORS_H
#define XFIELDS_CUBIC_INTERPOLATORS_H

/*gpufun*/
void TriCubicInterpolatedFieldMap_construct_b(
	TriCubicInterpolatedFieldMapData fmap,
	   const int64_t ix, const int64_t iy, const int64_t iz, 
       double* b_vector){

    /*gpuglmem*/ double* phi_taylor = TriCubicInterpolatedFieldMapData_getp1_phi_taylor(fmap, 0);
    // Optimization TODO: change int64 to int for less register pressure?
    const int64_t nx = TriCubicInterpolatedFieldMapData_get_nx(fmap);
    const int64_t ny = TriCubicInterpolatedFieldMapData_get_ny(fmap);

    // Optimization TODO: reorganize b_vector to align memory access
    for(int l = 0; l < 8; l++)
    {
        const int m = 8 * l;
        b_vector[m    ] = phi_taylor[ l + 8 * ( (ix    ) + nx * ( (iy    ) + ny * ( (iz    ) ) ) ) ];
        b_vector[m + 1] = phi_taylor[ l + 8 * ( (ix + 1) + nx * ( (iy    ) + ny * ( (iz    ) ) ) ) ];
        b_vector[m + 2] = phi_taylor[ l + 8 * ( (ix    ) + nx * ( (iy + 1) + ny * ( (iz    ) ) ) ) ];
        b_vector[m + 3] = phi_taylor[ l + 8 * ( (ix + 1) + nx * ( (iy + 1) + ny * ( (iz    ) ) ) ) ];
        b_vector[m + 4] = phi_taylor[ l + 8 * ( (ix    ) + nx * ( (iy    ) + ny * ( (iz + 1) ) ) ) ];
        b_vector[m + 5] = phi_taylor[ l + 8 * ( (ix + 1) + nx * ( (iy    ) + ny * ( (iz + 1) ) ) ) ];
        b_vector[m + 6] = phi_taylor[ l + 8 * ( (ix    ) + nx * ( (iy + 1) + ny * ( (iz + 1) ) ) ) ];
        b_vector[m + 7] = phi_taylor[ l + 8 * ( (ix + 1) + nx * ( (iy + 1) + ny * ( (iz + 1) ) ) ) ];
    }
    return ;
}

/*gpufun*/
int TriCubicInterpolatedFieldMap_interpolate_grad(
	TriCubicInterpolatedFieldMapData fmap,
	   const double x, const double y, const double z, 
	   double* dphi_dx, double* dphi_dy, double* dphi_dtau){
	
    double const x_min = TriCubicInterpolatedFieldMapData_get_x_min(fmap);
    double const y_min = TriCubicInterpolatedFieldMapData_get_y_min(fmap);
    double const z_min = TriCubicInterpolatedFieldMapData_get_z_min(fmap);

    double const inv_dx = 1. / TriCubicInterpolatedFieldMapData_get_dx(fmap);
    double const inv_dy = 1. / TriCubicInterpolatedFieldMapData_get_dy(fmap);
    double const inv_dz = 1. / TriCubicInterpolatedFieldMapData_get_dz(fmap);

    double const fx = ( x - x_min ) * inv_dx; // distance in normalized grid w.r.t. grid reference.
    double const fy = ( y - y_min ) * inv_dy; // normalized as in the coordinates of the
    double const fz = ( z - z_min ) * inv_dz; // grid points lie on the integer numbers.

    // Optimization TODO: change integer to int for less register pressure
    int64_t mirror_x = TriCubicInterpolatedFieldMapData_get_mirror_x(fmap);
    int64_t mirror_y = TriCubicInterpolatedFieldMapData_get_mirror_y(fmap);
    int64_t mirror_z = TriCubicInterpolatedFieldMapData_get_mirror_z(fmap);

    double const sign_x = (mirror_x == 1 && fx < 0.0 ) ?  -1. : 1.; // calculate if sign needs to be
    double const sign_y = (mirror_y == 1 && fy < 0.0 ) ?  -1. : 1.; // changed if mirroring about the
    double const sign_z = (mirror_z == 1 && fz < 0.0 ) ?  -1. : 1.; // origin is enabled

    double const sfx = sign_x * fx; // apply sign change if necessary
    double const sfy = sign_y * fy;
    double const sfz = sign_z * fz;

    double const ixf = floor(sfx); // find indices (keep them floating) (lower left corner of cell)
    double const iyf = floor(sfy);
    double const izf = floor(sfz);

    // Optimization TODO: change integer to int for less register pressure
    int64_t const ix = (int64_t) ixf; //convert floating point indices to integers
    int64_t const iy = (int64_t) iyf; 
    int64_t const iz = (int64_t) izf; 

    double const xn = sfx - ixf; // fractional part of distance. Equal to distance 
    double const yn = sfy - iyf; // w.r.t. grid point in the single cell
    double const zn = sfz - izf;

    // check that indices are within the grid
    // TODO: replace with ranges in x,y,z
    int indices_are_inside_box = ( ix >= 0 ) && ( ix <= ( TriCubicInterpolatedFieldMapData_get_nx(fmap) - 2 ) ) 
                              && ( iy >= 0 ) && ( iy <= ( TriCubicInterpolatedFieldMapData_get_ny(fmap) - 2 ) ) 
                              && ( iz >= 0 ) && ( iz <= ( TriCubicInterpolatedFieldMapData_get_nz(fmap) - 2 ) );

    if(!indices_are_inside_box){ // flag particle for death, it is outside the grid,
        return 1;                // no need for interpolation
    }

    double b_vector[64];
    TriCubicInterpolatedFieldMap_construct_b(fmap, ix, iy, iz, b_vector);

    double coefs[64];
    TriCubicInterpolatedFieldMap_construct_coefficients(b_vector, coefs);

    double x_power[4], y_power[4], z_power[4];
    x_power[0] = 1;
    y_power[0] = 1;
    z_power[0] = 1;

    x_power[1] = xn;
    y_power[1] = yn;
    z_power[1] = zn;

    x_power[2] = xn * xn;
    y_power[2] = yn * yn;
    z_power[2] = zn * zn;

    x_power[3] = x_power[2] * xn;
    y_power[3] = y_power[2] * yn;
    z_power[3] = z_power[2] * zn;

    for( int i = 1; i < 4; i++ ){
        for( int j = 0; j < 4; j++ ){
            for( int k = 0; k < 4; k++ ){
                *dphi_dx += i * ( ( ( coefs[i + 4 * j + 16 * k] * x_power[i-1] ) 
                                    * y_power[j] ) * z_power[k] );
            }
        }
    }
    *dphi_dx *= sign_x * inv_dx; 

    for( int i = 0; i < 4; i++ ){
        for( int j = 1; j < 4; j++ ){
            for( int k = 0; k < 4; k++ ){
                *dphi_dy += j * ( ( ( coefs[i + 4 * j + 16 * k] * x_power[i] ) 
                                    * y_power[j-1] ) * z_power[k] );
            }
        }
    }
    *dphi_dy *= sign_y * inv_dy; 

    for( int i = 0; i < 4; i++){
        for( int j = 0; j < 4; j++){
            for( int k = 1; k < 4; k++){
                *dphi_dtau += k * ( ( ( coefs[i + 4 * j + 16 * k] * x_power[i] ) 
                                    * y_power[j] ) * z_power[k-1] );
            }
        }
    }
    *dphi_dtau *= sign_z * inv_dz; 

	return 0;
}

#endif
