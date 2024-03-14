#include <stdio.h>
//#include "const.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_histogram2d.h>

//to find particle coordinates:

///Users/chenying/miniforge3/envs/xsuite-dev/include/


int compute_distribution_histogram(gsl_histogram2d* h1, ParticlesData particles,int npart){
    int countOutsideOfDomain = 0;
    int histOut;
    for(int i = 0;i<npart;++i) {
        histOut = gsl_histogram2d_increment(h1, ParticlesData_get_x(particles, i), ParticlesData_get_y(particles, i));
        if(histOut==GSL_EDOM){
            countOutsideOfDomain++;
        }       
    }
    return countOutsideOfDomain;
}
