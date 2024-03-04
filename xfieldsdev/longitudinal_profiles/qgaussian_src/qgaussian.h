// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_QGAUSSIAN 
#define XFIELDS_QGAUSSIAN 

/*gpufun*/
double LongitudinalProfileQGaussian_line_density_scalar(
		LongitudinalProfileQGaussianData prof, double z){

    const double number_of_particles = 
	    LongitudinalProfileQGaussianData_get_number_of_particles(prof);
    const double q = LongitudinalProfileQGaussianData_get__q_param(prof);
    const double cq = LongitudinalProfileQGaussianData_get__cq_param(prof);
    const double q_tol = LongitudinalProfileQGaussianData_get__q_tol(prof);
    const double z0 = LongitudinalProfileQGaussianData_get__z0(prof);
    const double beta_param = LongitudinalProfileQGaussianData_get__beta_param(prof);
    const double sqrt_beta_param = 
	    LongitudinalProfileQGaussianData_get__sqrt_beta_param(prof);
    const double z_min = LongitudinalProfileQGaussianData_get__support_min(prof);
    const double z_max = LongitudinalProfileQGaussianData_get__support_max(prof);

    const double factor = number_of_particles*sqrt_beta_param/cq; 


    if (fabs(q-1.) < q_tol){
	if (z<z_max && z>z_min){
	    double z_m_z0 = z - z0;
		return factor*exp(-beta_param*z_m_z0*z_m_z0 );
	}
	else{
		return 0; 
	}
    }
    else{
    	double exponent = 1./(1.-q);
	if (z<z_max && z>z_min){
	    double z_m_z0 = z - z0;
    		double q_exp_arg =  -(beta_param*z_m_z0*z_m_z0 );
    		double q_exp_res = pow(
	    	 (1.+(1.-q)*q_exp_arg), exponent );
    		return factor*q_exp_res;
	}
	else{
		return 0; 
	}
    }
}



/*gpukern*/
void line_density_qgauss(LongitudinalProfileQGaussianData prof,
		               const int64_t n,
		  /*gpuglmem*/ const double* z, 
		  /*gpuglmem*/       double* res){

   #pragma omp parallel for //only_for_context cpu_openmp 
   for(int ii; ii<n; ii++){ //vectorize_over ii n 

       res[ii] = LongitudinalProfileQGaussian_line_density_scalar(prof, z[ii]);
  
   }//end_vectorize
}

#endif
