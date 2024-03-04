#include <stdio.h>
//#include "const.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_histogram2d.h>

//to find particle coordinates:

///Users/chenying/miniforge3/envs/xsuite-dev/include/

int fillHistogram(gsl_histogram2d* h1, double* particleCoordinates,int npart){
    int countOutsideOfDomain = 0;
    int histOut;
    for(int i = 0;i<npart;++i) {
        histOut = gsl_histogram2d_increment(h1, particleCoordinates[i*7+0], particleCoordinates[i*7+2]);
        if(histOut==GSL_EDOM){
            countOutsideOfDomain++;
        }       
    }
    return countOutsideOfDomain;
}

void lumicalc(gsl_histogram2d* h1,gsl_histogram2d* h2,double intensity1,double intensity2,double *lumicombi) {
         double sum1= gsl_histogram2d_sum(h1);
         double sum2= gsl_histogram2d_sum(h2);
         float dx=( gsl_histogram2d_xmax(h1)- gsl_histogram2d_xmin(h1))/gsl_histogram2d_nx(h1);
         float dy=( gsl_histogram2d_ymax(h1)- gsl_histogram2d_ymin(h1))/gsl_histogram2d_ny(h1);
         double scale1=1.0/(sum1*dx*dy);
         double scale2=1.0/(sum2*dx*dy);
         gsl_histogram2d_scale(h1,scale1);
         gsl_histogram2d_scale(h2,scale2);
         gsl_histogram2d_mul(h1,h2); //                                                                     
         long double integral=0.25*dx*dy*(gsl_histogram2d_get(h1, 0, 0)+gsl_histogram2d_get(h1, gsl_histogram2d_nx(h1)-1, 0)+gsl_histogram2d_get(h1, 0, gsl_histogram2d_ny(h1)-1)+gsl_histogram2d_get(h1, gsl_histogram2d_nx(h1)-1, gsl_histogram2d_ny(h1)-1));
         double secondPart=0.0;
         double thirdPart=0.0;
         double fourthPart=0.0;
         for(int i = 0;i<gsl_histogram2d_nx(h1)-1;i++) {
           for(int j = 0;j<gsl_histogram2d_ny(h1)-1;j++) {
             secondPart+= gsl_histogram2d_get(h1,i+1,j+1);
           }
         }
         for(int i = 0;i<gsl_histogram2d_nx(h1)-1;i++) {

           thirdPart+= gsl_histogram2d_get(h1,i+1,0)+gsl_histogram2d_get(h1,i+1,gsl_histogram2d_ny(h1)-1);
         }
         for(int i = 0;i<gsl_histogram2d_nx(h1)-1;i++) {

           fourthPart+= gsl_histogram2d_get(h1,0,i+1)+gsl_histogram2d_get(h1,gsl_histogram2d_nx(h1)-1,i+1);
         }
         double integralf= integral + 0.25*dx*dy*(4*secondPart+2*thirdPart+2*fourthPart);
         *lumicombi =  intensity1*intensity2*integralf;
}

