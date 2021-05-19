#ifndef XFIEDLS_BIGUASSIIAN_H
#define XFIEDLS_BIGUASSIIAN_H

/*gpufun*/
void get_Ex_Ey_gauss(
             const double  x, 
             const double  y,
             const double  sigma_x, 
             const double  sigma_y,
             const double  min_sigma_diff, 
             double* Ex_ptr,
             double* Ey_ptr){

	if (fabs(sigma_x-sigma_y)< min_sigma_diff){
	    double sigma = 0.5*(sigma_x+sigma_y);
	    	get_transv_field_gauss_round(sigma, 0., 0., x, y, Ex_ptr, Ey_ptr);
	}
	else{
	    get_transv_field_gauss_ellip(
	            sigma_x, sigma_y, 0., 0., x, y, Ex_ptr, Ey_ptr);

	}
}

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

#endif
