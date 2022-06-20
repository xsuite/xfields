// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_H
#define XFIELDS_BEAMBEAM3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

/*gpufun*/
void Sigmas_propagate(
	Sigmas sigmas_0,
        double const S,
	double const threshold_singular,
        int64_t const handle_singularities,
        double* Sig_11_hat_ptr,
        double* Sig_33_hat_ptr,
        double* costheta_ptr,
        double* sintheta_ptr,
        double* dS_Sig_11_hat_ptr,
        double* dS_Sig_33_hat_ptr,
        double* dS_costheta_ptr,
        double* dS_sintheta_ptr)
{
    double const Sig_11_0 = Sigmas_get_Sig_11(sigmas_0);
    double const Sig_12_0 = Sigmas_get_Sig_12(sigmas_0);
    double const Sig_13_0 = Sigmas_get_Sig_13(sigmas_0);
    double const Sig_14_0 = Sigmas_get_Sig_14(sigmas_0);
    double const Sig_22_0 = Sigmas_get_Sig_22(sigmas_0);
    double const Sig_23_0 = Sigmas_get_Sig_23(sigmas_0);
    double const Sig_24_0 = Sigmas_get_Sig_24(sigmas_0);
    double const Sig_33_0 = Sigmas_get_Sig_33(sigmas_0);
    double const Sig_34_0 = Sigmas_get_Sig_34(sigmas_0);
    double const Sig_44_0 = Sigmas_get_Sig_44(sigmas_0);

    // Propagate sigma matrix
    double const Sig_11 = Sig_11_0 + 2.*Sig_12_0*S+Sig_22_0*S*S;
    double const Sig_33 = Sig_33_0 + 2.*Sig_34_0*S+Sig_44_0*S*S;
    double const Sig_13 = Sig_13_0 + (Sig_14_0+Sig_23_0)*S+Sig_24_0*S*S;
    double const Sig_12 = Sig_12_0 + Sig_22_0*S;
    double const Sig_14 = Sig_14_0 + Sig_24_0*S;
    double const Sig_22 = Sig_22_0 + 0.*S;
    double const Sig_23 = Sig_23_0 + Sig_24_0*S;
    double const Sig_24 = Sig_24_0 + 0.*S;
    double const Sig_34 = Sig_34_0 + Sig_44_0*S;
    double const Sig_44 = Sig_44_0 + 0.*S;

/*   
    printf("sigma_xx at CP: %.20f\n",   Sig_11);
    printf("sigma_xpx at CP: %.20f\n",  Sig_12);
    printf("sigma_xy at CP: %.20f\n",   Sig_13);
    printf("sigma_xpy at CP: %.20f\n",  Sig_14);
    printf("sigma_pxpx at CP: %.20f\n", Sig_22);
    printf("sigma_pxy at CP: %.20f\n",  Sig_23);
    printf("sigma_pxpy at CP: %.20f\n", Sig_24);
    printf("sigma_yy at CP: %.20f\n",   Sig_33);
    printf("sigma_ypy at CP: %.20f\n",  Sig_34);
    printf("sigma_pypy at CP: %.20f\n", Sig_44);
*/ 

    double const R = Sig_11 - Sig_33;
    double const W = Sig_11 + Sig_33;
    double const T = R*R + 4*Sig_13*Sig_13;

    //evaluate derivatives
    double const dS_R = 2.*(Sig_12_0-Sig_34_0)+2*S*(Sig_22_0-Sig_44_0);
    double const dS_W = 2.*(Sig_12_0+Sig_34_0)+2*S*(Sig_22_0+Sig_44_0);
    double const dS_Sig_13 = Sig_14_0 + Sig_23_0 + 2*Sig_24_0*S;
    double const dS_T = 2*R*dS_R+8.*Sig_13*dS_Sig_13;

    double Sig_11_hat, Sig_33_hat, costheta, sintheta, dS_Sig_11_hat,
           dS_Sig_33_hat, dS_costheta, dS_sintheta, cos2theta, dS_cos2theta;

    double const signR = mysign(R);

/*
    printf("sigma R: %.20f\n", R);
    printf("sigma W: %.20f\n", W);
    printf("sigma T: %.20f\n", T);
    printf("sigma sig_xx_cp: %.20f\n", Sig_11);
    printf("sigma sig_yy_cp: %.20f\n", Sig_33);
    printf("sigma sig_xy_cp: %.20f\n", Sig_13);
    printf("Sz_i: %.20f\n", S);
    printf("sigma sig_xx_ip: %.20f\n"  , Sig_11_0);
    printf("sigma sig_xpx_ip: %.20f\n" , Sig_12_0);
    printf("sigma sig_xy_ip: %.20f\n"  , Sig_13_0);
    printf("sigma sig_xpy_ip: %.20f\n" , Sig_14_0);
    printf("sigma sig_pxpx_ip: %.20f\n", Sig_22_0);
    printf("sigma sig_pxy_ip: %.20f\n" , Sig_23_0);
    printf("sigma sig_pxpy_ip: %.20f\n", Sig_24_0);
    printf("sigma sig_yy_ip: %.20f\n"  , Sig_33_0);
    printf("sigma sig_ypy_ip: %.20f\n" , Sig_34_0);
    printf("sigma sig_pypy_ip: %.20f\n", Sig_44_0);
    printf("dS_R: %.20f\n", dS_R); 
    printf("dS_W: %.20f\n", dS_W); 
    printf("dS_Sig_13: %.20f\n", dS_Sig_13); 
    printf("dS_T: %.20f\n", dS_T); 
*/
    if (T<threshold_singular && handle_singularities){
         
        double const a = Sig_12-Sig_34;
        double const b = Sig_22-Sig_44;
        double const c = Sig_14+Sig_23;
        double const d = Sig_24;

        double sqrt_a2_c2 = sqrt(a*a+c*c);

        if (sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2 < threshold_singular){
        //equivalent to: if np.abs(c)<threshold_singular and np.abs(a)<threshold_singular:
            //printf("first\n");

            if (fabs(d)> threshold_singular){
                cos2theta = fabs(b)/sqrt(b*b+4*d*d);
                }
            else{
                cos2theta = 1.;
                } // Decoupled beam

            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(b)*mysign(d)*sqrt(0.5*(1.-cos2theta));

            dS_costheta = 0.;
            dS_sintheta = 0.;

            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;

            dS_Sig_11_hat = 0.5*dS_W;
            dS_Sig_33_hat = 0.5*dS_W;
        }
        else{
            //~ printf("I am here\n");
            //~ printf("a=%.2e c=%.2e\n", a, c);
            //printf("second\n");

            sqrt_a2_c2 = sqrt(a*a+c*c); //repeated?
            cos2theta = fabs(2.*a)/(2*sqrt_a2_c2);
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(a)*mysign(c)*sqrt(0.5*(1.-cos2theta));

            dS_cos2theta = mysign(a)*(0.5*b/sqrt_a2_c2-a*(a*b+2.*c*d)/(2.*sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2));

            dS_costheta = 1./(4.*costheta)*dS_cos2theta;
            if (fabs(sintheta)>threshold_singular){
            //equivalent to: if np.abs(c)>threshold_singular:
                dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
            }
            else{
                dS_sintheta = d/(2.*a);
            }

            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;

            dS_Sig_11_hat = 0.5*dS_W + mysign(a)*sqrt_a2_c2;
            dS_Sig_33_hat = 0.5*dS_W - mysign(a)*sqrt_a2_c2;
        }
    }
    else{
        //printf("third\n");

        double const sqrtT = sqrt(T);
        cos2theta = signR*R/sqrtT;
        costheta = sqrt(0.5*(1.+cos2theta));
        sintheta = signR*mysign(Sig_13)*sqrt(0.5*(1.-cos2theta));

/*
        printf("R: %.20f\n", R);
        printf("signR: %.20f\n", signR);
        printf("sqrtT: %.20f\n", sqrtT);
        printf("cos2theta: %.20f\n", cos2theta);
        printf("sintheta: %.20f\n", sintheta);
        printf("dS_cos2theta: %.20f\n", dS_cos2theta);
        printf("dS_W: %.20f\n", dS_W);
        printf("dS_T: %.20f\n", dS_T);
*/

        //in sixtrack this line seems to be different different
        // sintheta = -mysign((Sig_11-Sig_33))*np.sqrt(0.5*(1.-cos2theta))

        Sig_11_hat = 0.5*(W+signR*sqrtT);
        Sig_33_hat = 0.5*(W-signR*sqrtT);

        dS_cos2theta = signR*(dS_R/sqrtT - R/(2*sqrtT*sqrtT*sqrtT)*dS_T);
        dS_costheta = 1./(4.*costheta)*dS_cos2theta;

        if (fabs(sintheta)<threshold_singular && handle_singularities){
        //equivalent to to np.abs(Sig_13)<threshold_singular
           // printf("this");
            dS_sintheta = (Sig_14+Sig_23)/R;
        }
        else{
           // printf("that");
            dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
        }

        dS_Sig_11_hat = 0.5*(dS_W + signR*0.5/sqrtT*dS_T);
        dS_Sig_33_hat = 0.5*(dS_W - signR*0.5/sqrtT*dS_T);
    }
    *Sig_11_hat_ptr = Sig_11_hat;
    *Sig_33_hat_ptr = Sig_33_hat;
    *costheta_ptr = costheta;
    *sintheta_ptr = sintheta;
    *dS_Sig_11_hat_ptr = dS_Sig_11_hat;
    *dS_Sig_33_hat_ptr = dS_Sig_33_hat;
    *dS_costheta_ptr = dS_costheta;
    *dS_sintheta_ptr = dS_sintheta;

}

/*gpufun*/
void BoostParameters_boost_coordinates(
    const BoostParameters bp,
    double* x_star,
    double* px_star,
    double* y_star,
    double* py_star,
    double* sigma_star,
    double* delta_star){

    double const sphi = BoostParameters_get_sphi(bp);
    double const cphi = BoostParameters_get_cphi(bp);
    double const tphi = BoostParameters_get_tphi(bp);
    double const salpha = BoostParameters_get_salpha(bp);
    double const calpha = BoostParameters_get_calpha(bp);

    double const x = *x_star;
    double const px = *px_star;
    double const y = *y_star;
    double const py = *py_star ;
    double const sigma = *sigma_star;
    double const delta = *delta_star ;

    double const h = delta + 1. - sqrt((1.+delta)*(1.+delta)-px*px-py*py);


    double const px_st = px/cphi-h*calpha*tphi/cphi;
    double const py_st = py/cphi-h*salpha*tphi/cphi;
    double const delta_st = delta -px*calpha*tphi-py*salpha*tphi+h*tphi*tphi;

    double const pz_st =
        sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);

    double const hx_st = px_st/pz_st;
    double const hy_st = py_st/pz_st;
    double const hsigma_st = 1.-(delta_st+1)/pz_st;

    double const L11 = 1.+hx_st*calpha*sphi;
    double const L12 = hx_st*salpha*sphi;
    double const L13 = calpha*tphi;

    double const L21 = hy_st*calpha*sphi;
    double const L22 = 1.+hy_st*salpha*sphi;
    double const L23 = salpha*tphi;

    double const L31 = hsigma_st*calpha*sphi;
    double const L32 = hsigma_st*salpha*sphi;
    double const L33 = 1./cphi;

    double const x_st = L11*x + L12*y + L13*sigma;
    double const y_st = L21*x + L22*y + L23*sigma;
    double const sigma_st = L31*x + L32*y + L33*sigma;

    *x_star = x_st;
    *px_star = px_st;
    *y_star = y_st;
    *py_star = py_st;
    *sigma_star = sigma_st;
    *delta_star = delta_st;

}

/*gpufun*/
void BoostParameters_boost_coordinates_inv(
    	const BoostParameters bp,
    	double* x,
    	double* px,
    	double* y,
    	double* py,
    	double* sigma,
    	double* delta){

    double const sphi = BoostParameters_get_sphi(bp);
    double const cphi = BoostParameters_get_cphi(bp);
    double const tphi = BoostParameters_get_tphi(bp);
    double const salpha = BoostParameters_get_salpha(bp);
    double const calpha = BoostParameters_get_calpha(bp);

    double const x_st = *x;
    double const px_st = *px;
    double const y_st = *y;
    double const py_st = *py ;
    double const sigma_st = *sigma;
    double const delta_st = *delta ;

    double const pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    double const hx_st = px_st/pz_st;
    double const hy_st = py_st/pz_st;
    double const hsigma_st = 1.-(delta_st+1)/pz_st;

    double const Det_L =
        1./cphi + (hx_st*calpha + hy_st*salpha-hsigma_st*sphi)*tphi;

    double const Linv_11 =
        (1./cphi + salpha*tphi*(hy_st-hsigma_st*salpha*sphi))/Det_L;

    double const Linv_12 =
        (salpha*tphi*(hsigma_st*calpha*sphi-hx_st))/Det_L;

    double const Linv_13 =
        -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    double const Linv_21 =
        (calpha*tphi*(-hy_st + hsigma_st*salpha*sphi))/Det_L;

    double const Linv_22 =
        (1./cphi + calpha*tphi*(hx_st-hsigma_st*calpha*sphi))/Det_L;

    double const Linv_23 =
        -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    double const Linv_31 = -hsigma_st*calpha*sphi/Det_L;
    double const Linv_32 = -hsigma_st*salpha*sphi/Det_L;
    double const Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    double const x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*sigma_st;
    double const y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*sigma_st;
    double const sigma_i = Linv_31*x_st + Linv_32*y_st + Linv_33*sigma_st;

    double const h = (delta_st+1.-pz_st)*cphi*cphi;

    double const px_i = px_st*cphi+h*calpha*tphi;
    double const py_i = py_st*cphi+h*salpha*tphi;

    double const delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;


    *x = x_i;
    *px = px_i;
    *y = y_i;
    *py = py_i;
    *sigma = sigma_i;
    *delta = delta_i;

}

/*gpufun*/ void compute_Gx_Gy(
             const double  x, 
             const double  y,
             const double  sigma_x, 
             const double  sigma_y,
             const double  min_sigma_diff, 
	     const double  Ex,
	     const double  Ey,
                   double* Gx_ptr,
                   double* Gy_ptr){

    double Gx, Gy;

    if (fabs(sigma_x-sigma_y) < min_sigma_diff){

        const double sigma = 0.5*(sigma_x+sigma_y);
        Gx = 1/(2.*(x*x+y*y))*(y*Ey-x*Ex+1./(2*PI*EPSILON_0*sigma*sigma)
                            *x*x*exp(-(x*x+y*y)/(2.*sigma*sigma)));
        Gy = 1./(2*(x*x+y*y))*(x*Ex-y*Ey+1./(2*PI*EPSILON_0*sigma*sigma)
                            *y*y*exp(-(x*x+y*y)/(2.*sigma*sigma)));
    }
    else{

        const double Sig_11 = sigma_x*sigma_x;
        const double Sig_33 = sigma_y*sigma_y;

	//printf("Inside Sig_11=%.10e\n", Sig_11);
	//printf("Inside Sig_33=%.10e\n", Sig_33);

        Gx =-1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)   
                   *(sigma_y/sigma_x*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
        Gy =1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*
                      (sigma_x/sigma_y*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));

	//printf("Inside Gx=%.10e\n", Gx);
	//printf("Inside Gy=%.10e\n", Gy);
    }

    *Gx_ptr = Gx;
    *Gy_ptr = Gy;
}

/*gpufun*/
void BeamBeamBiGaussian3D_track_local_particle(BeamBeamBiGaussian3DData el, 
		 	   LocalParticle* part0){

    // Get data from memory
    const double do_beamstrahlung  = BeamBeamBiGaussian3DData_get_do_beamstrahlung(el);   
    const double q0_bb  = BeamBeamBiGaussian3DData_get_q0(el);     
    const BoostParameters bpar = BeamBeamBiGaussian3DData_getp_boost_parameters(el);
    const Sigmas Sigmas_0_star = BeamBeamBiGaussian3DData_getp_Sigmas_0_star(el);
    const double min_sigma_diff = BeamBeamBiGaussian3DData_get_min_sigma_diff(el);
    const double threshold_singular = 
	    BeamBeamBiGaussian3DData_get_threshold_singular(el);
    const int N_slices = BeamBeamBiGaussian3DData_get_num_slices(el);
    const double delta_x = BeamBeamBiGaussian3DData_get_delta_x(el);
    const double delta_y = BeamBeamBiGaussian3DData_get_delta_y(el);
    const double delta_px = BeamBeamBiGaussian3DData_get_delta_px(el);
    const double delta_py = BeamBeamBiGaussian3DData_get_delta_py(el);
    const double x_CO  = BeamBeamBiGaussian3DData_get_x_CO(el);     
    const double px_CO = BeamBeamBiGaussian3DData_get_px_CO(el);
    const double y_CO = BeamBeamBiGaussian3DData_get_y_CO(el);
    const double py_CO = BeamBeamBiGaussian3DData_get_py_CO(el);
    const double sigma_CO = BeamBeamBiGaussian3DData_get_sigma_CO(el);
    const double delta_CO = BeamBeamBiGaussian3DData_get_delta_CO(el);
    const double Dx_sub = BeamBeamBiGaussian3DData_get_Dx_sub(el); 
    const double Dpx_sub = BeamBeamBiGaussian3DData_get_Dpx_sub(el);
    const double Dy_sub =BeamBeamBiGaussian3DData_get_Dy_sub(el);
    const double Dpy_sub =BeamBeamBiGaussian3DData_get_Dpy_sub(el);
    const double Dsigma_sub =BeamBeamBiGaussian3DData_get_Dsigma_sub(el);
    const double Ddelta_sub =BeamBeamBiGaussian3DData_get_Ddelta_sub(el);
    /*gpuglmem*/ const double* N_part_per_slice_arr = 
	    BeamBeamBiGaussian3DData_getp1_N_part_per_slice(el, 0);
    /*gpuglmem*/ const double* x_slices_star_arr = 
	    BeamBeamBiGaussian3DData_getp1_x_slices_star(el, 0);
    /*gpuglmem*/ const double* y_slices_star_arr = 
	    BeamBeamBiGaussian3DData_getp1_y_slices_star(el, 0);
    /*gpuglmem*/ const double* sigma_slices_star_arr = 
	    BeamBeamBiGaussian3DData_getp1_sigma_slices_star(el, 0);

    double energy_loss;
    //start_per_particle_block (part0->part)
    	double x = LocalParticle_get_x(part);
    	double px = LocalParticle_get_px(part);
    	double y = LocalParticle_get_y(part);
    	double py = LocalParticle_get_py(part);
    	double zeta = LocalParticle_get_zeta(part);
    	double pzeta = LocalParticle_get_pzeta(part);

    	const double q0 = LocalParticle_get_q0(part); 
    	const double p0c = LocalParticle_get_p0c(part); // eV

    	const double P0 = p0c/C_LIGHT*QELEM;

    	// Change reference frame
    	double x_star =     x     - x_CO    - delta_x;
    	double px_star =    px    - px_CO;
    	double y_star =     y     - y_CO    - delta_y;
    	double py_star =    py    - py_CO;
    	double sigma_star = zeta  - sigma_CO;
    	double pzeta_star = pzeta - delta_CO; // TODO: could be fixed, in any case we assume beta=beta0=1
	                                      //       in the synchrobeam

/*
        printf("[beambeam3d] [%d] before boost:\n", part->ipart);
    	printf("\t_star=%.10e\n", x_star);
	printf("\tpx_star=%.10e\n", px_star);
	printf("\ty_star=%.10e\n", y_star);
	printf("\ty_star=%.10e\n", py_star);
    	printf("\tsigma_star=%.10e\n", sigma_star);
	printf("\tdelta_star=%.20e\n", delta_star);
*/

    	// Boost coordinates of the weak beam
	BoostParameters_boost_coordinates(bpar, &x_star, &px_star, &y_star, &py_star, &sigma_star, &delta_star);
    	LocalParticle_update_delta(part, delta_star);  // this updates energy variables, which are used in beamstrahlung generation

/*
        printf("[beambeam3d] [%d] after boost:\n", part->ipart);
    	printf("\tx_star=%.10e\n", x_star);
	printf("\tpx_star=%.10e\n", px_star);
	printf("\ty_star=%.10e\n", y_star);
	printf("\tpy_star=%.10e\n", py_star);
    	printf("\tsigma_star=%.10e\n", sigma_star);
	printf("\tdelta_star=%.20e\n", delta_star);
*/

    	// Synchro beam, WS because coords are not updaded by default after eahc slice interaction
    	for (int i_slice=0; i_slice<N_slices; i_slice++)
    	{

            // new: reload boosted delta after each slice kick to compare with sbc6d; these are boosted
       	    delta_star = LocalParticle_get_delta(part);

/*
            printf("[beambeam3d] [%d] at ip:\n", part->ipart);
    	    printf("\tx=%.10e\n", x_star);
	    printf("\ty=%.10e\n", y_star);
    	    printf("\tz=%.10e\n", sigma_star);
            printf("\tpx=%.10e\n", px_star);
            printf("\tpy=%.10e\n", py_star); 
            printf("\tdelta=%.20e\n", delta_star);
*/

    	    const double sigma_slice_star = sigma_slices_star_arr[i_slice];
    	    const double x_slice_star = x_slices_star_arr[i_slice];
    	    const double y_slice_star = y_slices_star_arr[i_slice];
//            printf("slice star xyz: %.12f, %.12f, %.12f", x_slice_star, y_slice_star, sigma_slice_star);
    	    //Compute force scaling factor
    	    const double Ksl = N_part_per_slice_arr[i_slice]*QELEM*q0_bb
		               *QELEM*q0/(P0 * C_LIGHT);

    	    //Identify the Collision Point (CP)
    	    const double S = 0.5*(sigma_star - sigma_slice_star);
    	    
            // Propagate sigma matrix
    	    double Sig_11_hat_star, Sig_33_hat_star, costheta, sintheta;
    	    double dS_Sig_11_hat_star, dS_Sig_33_hat_star, dS_costheta, dS_sintheta;

    	    // Get strong beam shape at the CP
	    Sigmas_propagate(Sigmas_0_star, S, threshold_singular, 1,
    	        &Sig_11_hat_star, &Sig_33_hat_star,
    	        &costheta, &sintheta,
    	        &dS_Sig_11_hat_star, &dS_Sig_33_hat_star,
    	        &dS_costheta, &dS_sintheta);

    	    // Evaluate transverse coordinates of the weake baem w.r.t. the strong beam centroid at the CP
    	    const double x_bar_star = x_star + px_star*S - x_slice_star;
    	    const double y_bar_star = y_star + py_star*S - y_slice_star;

    	    // Move to the uncoupled reference frame
    	    const double x_bar_hat_star = x_bar_star*costheta +y_bar_star*sintheta;
    	    const double y_bar_hat_star = -x_bar_star*sintheta +y_bar_star*costheta;

/*
            printf("[beambeam3d] [%d] at cp:\n", part->ipart);
    	    printf("\tx=%.10e\n", x_bar_star);
	    printf("\ty=%.10e\n", y_bar_star);
    	    printf("\tz=%.10e\n", S);
            printf("\tpx=%.10e\n", px_star);
            printf("\tpy=%.10e\n", py_star); 
            printf("\tdelta=%.20e\n", delta_star);
            printf("\tksl=%.10e, q0: %.12f, q0_bb: %.12f, n_bb: %.12f\n", Ksl, q0, q0_bb, N_part_per_slice_arr[i_slice]);
            printf("\tenergy0: %.20f, ptau: %.20f, p0c: %.20f, beta0: %.20f\n", LocalParticle_get_energy0(part), LocalParticle_get_ptau(part), LocalParticle_get_p0c(part), LocalParticle_get_beta0(part));


            printf("\tsig_11: %.20f, sig_12: %.20f, sig_13: %.20f, sig_14: %.20f, sig_22: %.20f, sig_23: %.20f, sig_24: %.20f, sig_33: %.20f, sig_34: %.20f, sig_44: %.20f\n", Sigmas_get_Sig_11(Sigmas_0_star), Sigmas_get_Sig_12(Sigmas_0_star), Sigmas_get_Sig_13(Sigmas_0_star), Sigmas_get_Sig_14(Sigmas_0_star), Sigmas_get_Sig_22(Sigmas_0_star), Sigmas_get_Sig_23(Sigmas_0_star), Sigmas_get_Sig_24(Sigmas_0_star), Sigmas_get_Sig_33(Sigmas_0_star), Sigmas_get_Sig_34(Sigmas_0_star), Sigmas_get_Sig_44(Sigmas_0_star)); 


            printf("\tSig_11_hat_star: %.20f, Sig_33_hat_star: %.20f, costheta: %.12f, sintheta: %.12f\n", Sig_11_hat_star, Sig_33_hat_star, costheta, sintheta);
            printf("\tdS_Sig_11_hat_stat: %.20f, dS_Sig_33_hat_star: %.20f, dS_costheta: %.12f, dS_sintheta: %.12f\n", dS_Sig_11_hat_star, dS_Sig_33_hat_star, dS_costheta, dS_sintheta);

            printf("[%d] uncoupled:\n", part->ipart);
    	    printf("\tx=%.10e\n", x_bar_hat_star);
	    printf("\ty=%.10e\n", y_bar_hat_star);
*/

    	    // Compute derivatives of the transformation
    	    const double dS_x_bar_hat_star = x_bar_star*dS_costheta +y_bar_star*dS_sintheta;
    	    const double dS_y_bar_hat_star = -x_bar_star*dS_sintheta +y_bar_star*dS_costheta;

    	    // Get transverse fieds
    	    double Ex, Ey;
    	    get_Ex_Ey_gauss(x_bar_hat_star, y_bar_hat_star,
    	        sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star),
		min_sigma_diff,
    	        &Ex, &Ey);

	    //printf("\tEx=%.10e\n", Ex);
	    //printf("\tEy=%.10e\n", Ey);
	
	    //compute Gs
	    double Gx, Gy;
	    compute_Gx_Gy(x_bar_hat_star, y_bar_hat_star,
			  sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), 
                          min_sigma_diff, Ex, Ey, &Gx, &Gy);
	    
	    //printf("\tGx=%.10e\n", Gx);
	    //printf("\tGy=%.10e\n", Gy);

            //printf("\tKsl=%.10e\n", Ksl);

    	    // Compute kicks
    	    double Fx_hat_star = Ksl*Ex;
    	    double Fy_hat_star = Ksl*Ey;
    	    double Gx_hat_star = Ksl*Gx;
    	    double Gy_hat_star = Ksl*Gy;

    	    // Move kisks to coupled reference frame
    	    double Fx_star = Fx_hat_star*costheta - Fy_hat_star*sintheta;
    	    double Fy_star = Fx_hat_star*sintheta + Fy_hat_star*costheta;

    	    // Compute longitudinal kick
    	    double Fz_star = 0.5*(Fx_hat_star*dS_x_bar_hat_star  + Fy_hat_star*dS_y_bar_hat_star+
    	                   Gx_hat_star*dS_Sig_11_hat_star + Gy_hat_star*dS_Sig_33_hat_star);

            // emit beamstrahlung photons from single macropart
            if (do_beamstrahlung==1){
              
                // total kick
                //printf("\trpp=%.20e\n", LocalParticle_get_rpp(part));
                //printf("\tFx_star=%.20e\n", Fx_star); 
                //printf("\tFy_star=%.20e\n", Fy_star); 

                double const Fr = hypot(Fx_star, Fy_star) * LocalParticle_get_rpp(part); // rad
    
                // bending radius is over this distance (half slice length)
                /*gpuglmem*/ const double* dz_arr = BeamBeamBiGaussian3DData_getp1_dz(el, 0);
                const double dz = dz_arr[i_slice]/2.;
                    
                double initial_energy = LocalParticle_get_energy0(part) + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part); 
                energy_loss = synrad(part, Fr, dz);
                //printf("\tFr: %.20e, eloss: %.20e\n", Fr, energy_loss); 

                // BS rescales these, so load again before kick 
                delta_star = LocalParticle_get_delta(part);  
            }
            else if(do_beamstrahlung==2){
               double var_z_bb = 0.00345;
               energy_loss = synrad_avg(part, N_part_per_slice_arr[i_slice], sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), var_z_bb);  // slice intensity and RMS slice sizes
               //printf("n_bb: %.20e, sigma_11: %.20e, sigma_33: %.20e, energy_loss: %.20e\n", N_part_per_slice_arr[i_slice], sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), energy_loss);
 
               delta_star = LocalParticle_get_delta(part);  
           }
 
    	    // Apply the kicks (Hirata's synchro-beam)

            //printf("[beambeam3d] [%d] before delta kick of slice %d\n", part->ipart, i_slice);
            //printf("\tdelta_star=%.20e\n", delta_star);  
    	   delta_star = delta_star + Fz_star+0.5*(
    	                Fx_star*(px_star+0.5*Fx_star)+
    	                Fy_star*(py_star+0.5*Fy_star));

/*
            printf("[beambeam3d] [%d] after kick of slice %d:\n", part->ipart, i_slice);
	          printf("\tdelta_star=%.20e\n", delta_star);
            printf("\tFx_star=%.20e\n", Fx_star);
            printf("\tFy_star=%.20e\n", Fy_star);
            printf("\tFz_star=%.20e\n", Fz_star);
            printf("\tpx_star=%.20e\n", px_star);
            printf("\tpy_star=%.20e\n", py_star);
*/ 

	        x_star = x_star - S*Fx_star;
    	    px_star = px_star + Fx_star;
    	    y_star = y_star - S*Fy_star;
    	    py_star = py_star + Fy_star;

/*
            printf("[beambeam3d] [%d] after kick of slice %d:\n", part->ipart, i_slice);
    	    printf("\tx_star=%.10e\n", x_star);
	    printf("\tpx_star=%.10e\n", px_star);
	    printf("\ty_star=%.10e\n", y_star);
 	    printf("\tpy_star=%.10e\n", py_star);
    	    printf("\tsigma_star=%.10e\n", sigma_star);
   	    printf("\tdelta_star=%.20e\n", delta_star);
*/            
            // new: update boosted delta after each ss interaction, like in sbc6d_full; this updates energy vars, like rpp
            LocalParticle_update_delta(part, delta_star);




    	}

    	// Inverse boost on the coordinates of the weak beam

	BoostParameters_boost_coordinates_inv(bpar, &x_star, &px_star, &y_star, &py_star, &sigma_star, &delta_star);


	//printf("pzeta_ret=%.10e\n", pzeta_star);

    	// Go back to original reference frame and remove dipolar effect
    	x =     x_star     + x_CO   + delta_x - Dx_sub;
    	px =    px_star    + px_CO            - Dpx_sub;
    	y =     y_star     + y_CO   + delta_y - Dy_sub;
    	py =    py_star    + py_CO            - Dpy_sub;
    	zeta =  sigma_star + sigma_CO         - Dsigma_sub;
    	pzeta = pzeta_star + delta_CO         - Ddelta_sub;

/*
        printf("[%d] after inverse boost:\n", part->ipart);
    	printf("x_star=%.10e\n", x);
	printf("px_star=%.10e\n", px);
	printf("y_star=%.10e\n", y);
	printf("py_star=%.10e\n", py);
    	printf("sigma_star=%.10e\n", zeta);
	printf("delta_star=%.10e\n\n", delta);

        printf("-----------------------------------------------\n");
*/
    	LocalParticle_set_x(part, x);
    	LocalParticle_set_px(part, px);
    	LocalParticle_set_y(part, y);
    	LocalParticle_set_py(part, py);
    	LocalParticle_set_zeta(part, zeta);
    	LocalParticle_update_pzeta(part, pzeta);
	
    //end_per_particle_block

}


#endif
