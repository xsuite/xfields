#ifndef XFIEDLS_BIGUASSIIAN_H
#define XFIEDLS_BIGUASSIIAN_H

/*gpufun*/
void BiGaussianFieldMap_get_Ex_Ey(
           BiGaussianFieldMapData fmap,
                    const double  x,
                    const double  y,
                          double* Ex,
                          double* Ey){

    const double sigma_x = BiGaussianFieldMapData_get_sigma_x(fmap);
    const double sigma_y = BiGaussianFieldMapData_get_sigma_y(fmap);
    const double mean_x = BiGaussianFieldMapData_get_mean_x(fmap);
    const double mean_y = BiGaussianFieldMapData_get_mean_y(fmap);

}



#endif
