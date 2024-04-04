// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2023.                   //
// ########################################### //

#ifndef XFIELDS_BIGAUSSIAN_H_FIELDMAP
#define XFIELDS_BIGAUSSIAN_H_FIELDMAP

/*gpufun*/
void BiGaussianFieldMap_get_dphi_dx_dphi_dy(
           BiGaussianFieldMapData fmap,
                    const double  x,
                    const double  y,
                          double* dphi_dx,
                          double* dphi_dy){

    const double sigma_x = BiGaussianFieldMapData_get_sigma_x(fmap);
    const double sigma_y = BiGaussianFieldMapData_get_sigma_y(fmap);
    const double mean_x = BiGaussianFieldMapData_get_mean_x(fmap);
    const double mean_y = BiGaussianFieldMapData_get_mean_y(fmap);
    const double min_sigma_diff = BiGaussianFieldMapData_get_min_sigma_diff(fmap);

    double Ex, Ey;
    get_Ex_Ey_gauss(
             x-mean_x,
             y-mean_y,
             sigma_x,
             sigma_y,
             min_sigma_diff,
             &Ex,
             &Ey);

    *dphi_dx = -Ex;
    *dphi_dy = -Ey;
}

#endif // XFIELDS_BIGAUSSIAN_H_FIELDMAP