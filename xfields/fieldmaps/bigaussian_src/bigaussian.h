#ifndef XFIEDLS_BIGUASSIIAN_H
#define XFIEDLS_BIGUASSIIAN_H

// for quick test with gcc
#include "constants.h" //only_for_context none
#include "complex_error_function.h" //only_for_context none

/*gpufun*/
void get_transv_field_gauss_round(
    double sigma, double Delta_x, double Delta_y,
    double x, double y,
    double* Ex,
    double* Ey)
{
  double r2, temp;

  r2 = (x-Delta_x)*(x-Delta_x)+(y-Delta_y)*(y-Delta_y);
  if (r2<1e-20) temp = sqrt(r2)/(2.*PI*EPSILON_0*sigma); //linearised
  else          temp = (1-exp(-0.5*r2/(sigma*sigma)))/(2.*PI*EPSILON_0*r2);

  (*Ex) = temp * (x-Delta_x);
  (*Ey) = temp * (y-Delta_y);
}

/*gpufun*/
void get_transv_field_gauss_ellip(
        double sigma_x,  double sigma_y,
        double Delta_x,  double Delta_y,
        const double x,
	const double y,
        double* Ex_out,
        double* Ey_out)
{
  double sigmax = sigma_x;
  double sigmay = sigma_y;

  // I always go to the first quadrant and then apply the signs a posteriori
  // numerically more stable (see http://inspirehep.net/record/316705/files/slac-pub-5582.pdf)

  double abx = fabs(x - Delta_x);
  double aby = fabs(y - Delta_y);

  //printf("x = %.2e y = %.2e abx = %.2e aby = %.2e", xx, yy, abx, aby);

  double S, factBE, Ex, Ey;
  double etaBE_re, etaBE_im, zetaBE_re, zetaBE_im;
  double w_etaBE_re, w_etaBE_im, w_zetaBE_re, w_zetaBE_im;
  double expBE;

  if (sigmax>sigmay){
    S = sqrt(2.*(sigmax*sigmax-sigmay*sigmay));
    factBE = 1./(2.*EPSILON_0*SQRT_PI*S);

    etaBE_re = sigmay/sigmax*abx;
    etaBE_im = sigmax/sigmay*aby;

    zetaBE_re = abx;
    zetaBE_im = aby;

    //w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re/S, zetaBE_im/S)
    cerrf(zetaBE_re/S, zetaBE_im/S , &(w_zetaBE_re), &(w_zetaBE_im));
    //w_etaBE_re, w_etaBE_im = wfun(etaBE_re/S, etaBE_im/S)
    cerrf(etaBE_re/S, etaBE_im/S , &(w_etaBE_re), &(w_etaBE_im));

    expBE = exp(-abx*abx/(2*sigmax*sigmax)-aby*aby/(2*sigmay*sigmay));

    Ex = factBE*(w_zetaBE_im - w_etaBE_im*expBE);
    Ey = factBE*(w_zetaBE_re - w_etaBE_re*expBE);
  }
  else if (sigmax<sigmay){
    S = sqrt(2.*(sigmay*sigmay-sigmax*sigmax));
    factBE = 1./(2.*EPSILON_0*SQRT_PI*S);

    etaBE_re = sigmax/sigmay*aby;
    etaBE_im = sigmay/sigmax*abx;

    zetaBE_re = aby;
    zetaBE_im = abx;

    //w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re/S, zetaBE_im/S)
    cerrf(zetaBE_re/S, zetaBE_im/S , &(w_zetaBE_re), &(w_zetaBE_im));
    //w_etaBE_re, w_etaBE_im = wfun(etaBE_re/S, etaBE_im/S)
    cerrf(etaBE_re/S, etaBE_im/S , &(w_etaBE_re), &(w_etaBE_im));

    expBE = exp(-aby*aby/(2*sigmay*sigmay)-abx*abx/(2*sigmax*sigmax));

    Ey = factBE*(w_zetaBE_im - w_etaBE_im*expBE);
    Ex = factBE*(w_zetaBE_re - w_etaBE_re*expBE);
  }
  else{
    //printf("Round beam not implemented!\n");
    //exit(1);
    Ex = Ey = 1./0.;
  }

  if((x - Delta_x)<0) Ex=-Ex;
  if((y - Delta_y)<0) Ey=-Ey;

  (*Ex_out) = Ex;
  (*Ey_out) = Ey;
}

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

#ifndef NOFIELDMAP

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

#endif
