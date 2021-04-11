#include <math.h> //only_for_context none



void q_gaussian_profile(
		    const int n,
		    const double* z,
		    const double z0,
		    const double z_min,
		    const double z_max,
		    const double beta, 
		    const double q, 
		    const double q_tol,
		    const double factor,
		          double* res){
    if (fabs(q-1.) < q_tol){
    	for(int ii; ii<n; ii++){ //vectorize_over ii n 
    	    double zi = z[ii];
	    if (zi<z_max && zi>z_min){
		double zi_m_z0 = zi - z0;
	    	res[ii] = factor*exp(-beta*zi_m_z0*zi_m_z0 );
	    }
	    else{
	    	res[ii] = 0; 
	    }
    	}//end_vectorize
    }
    else{
    	double exponent = 1./(1.-q);
    	for(int ii; ii<n; ii++){ //vectorize_over ii n
    	    double zi = z[ii];
	    if (zi<z_max && zi>z_min){
		double zi_m_z0 = zi - z0;
    	    	double q_exp_arg =  -(beta*zi_m_z0*zi_m_z0 );
    	    	double q_exp_res = pow(q_exp_arg, exponent );
    	    	res[ii] = factor*q_exp_res;
	    }
	    else{
	    	res[ii] = 0; 
	    }
    	}//end_vectorize
    }
}	
