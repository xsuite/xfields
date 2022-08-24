// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIEDLS_BIGUASSIIAN_H
#define XFIEDLS_BIGUASSIIAN_H


#if !defined(profiler_path)
    #define profiler_path "/Users/pkicsiny/phd/cern/PySBC/outputs" 
#endif


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

  clock_t tt;
  double time_taken;

  if (sigmax>sigmay){
    S = sqrt(2.*(sigmax*sigmax-sigmay*sigmay));
    factBE = 1./(2.*EPSILON_0*SQRT_PI*S);

    etaBE_re = sigmay/sigmax*abx;
    etaBE_im = sigmax/sigmay*aby;

    zetaBE_re = abx;
    zetaBE_im = aby;


//    char profiler_file_first[1024];
//    sprintf(profiler_file_first, "%s/%s", profiler_path, "profiler_sbc6d_full_first.txt");
//     char profiler_file_second[1024];
//    sprintf(profiler_file_second, "%s/%s", profiler_path, "profiler_sbc6d_full_second.txt");
//    FILE *f1 = fopen(profiler_file_first, "a");
//    FILE *f2 = fopen(profiler_file_second, "a");

//    tt = clock();
//    for(int l=0;l<100000;l++){
    //w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re/S, zetaBE_im/S)
        cerrf(zetaBE_re/S, zetaBE_im/S , &(w_zetaBE_re), &(w_zetaBE_im));
//    }
//    tt = clock() - tt;
//    time_taken = ((double)tt)/CLOCKS_PER_SEC;
//    fprintf(f1, "%.4e\n", time_taken);

//    tt = clock();
//    for(int l=0;l<100000;l++){
    //w_etaBE_re, w_etaBE_im = wfun(etaBE_re/S, etaBE_im/S)
        cerrf(etaBE_re/S, etaBE_im/S , &(w_etaBE_re), &(w_etaBE_im));
//    }
//    tt = clock() - tt;
//    time_taken = ((double)tt)/CLOCKS_PER_SEC;
//    fprintf(f2, "%.4e\n", time_taken);

//    fclose(f1);
//    fclose(f2);

    expBE = exp(-abx*abx/(2*sigmax*sigmax)-aby*aby/(2*sigmay*sigmay));

    Ex = factBE*(w_zetaBE_im - w_etaBE_im*expBE);
    Ey = factBE*(w_zetaBE_re - w_etaBE_re*expBE);

    //FILE *f1 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_ellip.txt", "a");
    //fprintf(f1, "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", S, factBE, etaBE_re, etaBE_im, zetaBE_re, zetaBE_im, expBE, w_zetaBE_re, w_zetaBE_im, w_etaBE_re, w_etaBE_im, sigmax, sigmay, Ex, Ey);
    //fclose(f1);

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
    Ex = Ey = 0.;
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

        // round beam
	if (fabs(sigma_x-sigma_y)< min_sigma_diff){
            //printf("Using round beam\n");
	    double sigma = 0.5*(sigma_x+sigma_y);
	    	get_transv_field_gauss_round(sigma, 0., 0., x, y, Ex_ptr, Ey_ptr);
	}
       
        // elliptical beam
	else{
            //printf("Using elliptical beam\nsigma_x: %.20f\nsigma_y: %.20f\nx: %.20f\ny: %.20f\n", sigma_x, sigma_y, x, y);
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
