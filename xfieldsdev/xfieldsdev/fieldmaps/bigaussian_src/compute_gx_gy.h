// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_COMPUTE_GX_GY
#define XFIELDS_COMPUTE_GX_GY

/*gpufun*/ void compute_Gx_Gy(
        const double x,
        const double y,
        const double sigma_x,
        const double sigma_y,
        const double min_sigma_diff,
        const double Ex,
        const double Ey,
        double* Gx_ptr,
        double* Gy_ptr) {

    double Gx, Gy;

    if (fabs(sigma_x-sigma_y) < min_sigma_diff) {
        const double sigma = 0.5 * (sigma_x+sigma_y);
        if ((x*x + y*y) < 1e-14) {
            Gx = 1./(8 * PI * EPSILON_0 * sigma * sigma);
            Gy = Gx;
        }
	    else {
            Gx = 1/(2.*(x*x+y*y))*(y*Ey-x*Ex+1./(2*PI*EPSILON_0*sigma*sigma)
                    *x*x*exp(-(x*x+y*y)/(2.*sigma*sigma)));
            Gy = 1./(2*(x*x+y*y))*(x*Ex-y*Ey+1./(2*PI*EPSILON_0*sigma*sigma)
                    *y*y*exp(-(x*x+y*y)/(2.*sigma*sigma)));
	    }
    }
    else {
        const double Sig_11 = sigma_x*sigma_x;
        const double Sig_33 = sigma_y*sigma_y;

        Gx = -1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)
                *(sigma_y/sigma_x*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
        Gy = 1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*
                (sigma_x/sigma_y*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));

    }

    *Gx_ptr = Gx;
    *Gy_ptr = Gy;
}

#endif //XFIELDS_COMPUTE_GX_GY
