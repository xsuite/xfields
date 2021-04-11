#include <math.h> //only_for_context none



void q_gaussian_profile(
		    const int n,
		    const double* z,
		    const double beta, 
		    const double q, 
		    const double q_tol,
		    const double factor,
		          double* res){
    if (fabs(q-1.) < q_tol){

    }
    else {
    	double exponent = 1./(1.-q);
    	for(int ii; ii<n; ii++){ //vectorize_over ii 
    	    double zi = z[ii]
    	    double q_exp_arg =  -( beta * zi * zi );
    	    double q_exp_res = pow(q_exp_arg, exponent );
    	    res[ii] = factor*q_exp_res;
    	}//end_vectorize
    }
}	
