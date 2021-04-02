#ifndef FIELDS_BIGAUSSIAN 
#define FIELDS_BIGAUSSIAN 

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


/*gpukern*/
void get_Ex_Ey_Gx_Gy_gauss(
                 const int n_points,
    /*gpuglmem*/ const double* x_ptr, 
    /*gpuglmem*/ const double* y_ptr,
                 const double  sigma_x, 
                 const double  sigma_y,
                 const double  min_sigma_diff, 
                 const int     skip_Gs,
    /*gpuglmem*/       double* Ex_ptr,
    /*gpuglmem*/       double* Ey_ptr,
    /*gpuglmem*/       double* Gx_ptr,
    /*gpuglmem*/       double* Gy_ptr){

    for (int ip=0; ip<n_points; ip++){  //vectorize_over ip n_points
    	double x, y, Ex, Ey, Gx, Gy;

    	x = x_ptr[ip];
    	y = y_ptr[ip];

    	if (fabs(sigma_x-sigma_y)< min_sigma_diff){
    	    double sigma = 0.5*(sigma_x+sigma_y);
    	    	get_transv_field_gauss_round(sigma, 0., 0., x, y, &Ex, &Ey);

    	    	if(skip_Gs){
    	    	  Gx = 0.;
    	    	  Gy = 0.;
    	    	}
    	    	else{
    	    	  Gx = 1/(2.*(x*x+y*y))*(y*Ey-x*Ex+1./(2*PI*EPSILON_0*sigma*sigma)
    	    	                    *x*x*exp(-(x*x+y*y)/(2.*sigma*sigma)));
    	    	  Gy = 1./(2*(x*x+y*y))*(x*Ex-y*Ey+1./(2*PI*EPSILON_0*sigma*sigma)
    	    	                    *y*y*exp(-(x*x+y*y)/(2.*sigma*sigma)));
    	    	}
    	}
    	else{

    	    get_transv_field_gauss_ellip(
    	            sigma_x, sigma_y, 0., 0., x, y, &Ex, &Ey);

    	    double Sig_11 = sigma_x*sigma_x;
    	    double Sig_33 = sigma_y*sigma_y;

    	    if(skip_Gs){
    	      Gx = 0.;
    	      Gy = 0.;
    	    }
    	    else{
    	      Gx =-1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*\
    	                  (sigma_y/sigma_x*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
    	      Gy =1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*\
    	                  (sigma_x/sigma_y*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
    	    }
    	}
    	Ex_ptr[ip] = Ex;
    	Ey_ptr[ip] = Ey;
    	Gx_ptr[ip] = Gx;
    	Gy_ptr[ip] = Gy;
    }//end_vectorize
}


#endif
