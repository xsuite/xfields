#ifndef XFIELDS_STRONGSTRONG3D_H
#define XFIELDS_STRONGSTRONG3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

/*gpufun*/
void uncouple_xy_plane(
	Sigmas sigma_matrix_boosted_ip,
        Sigmas sigma_matrix_boosted_cp,
        double const Sz_i, // macropart z coord at CP
	double const threshold_singular,
        int64_t const handle_singularities, // flag
        double* Sig_11_uncoupled_ptr, // return values: sigma_x/y at uncoupled frame
        double* Sig_33_uncoupled_ptr,
        double* costheta_ptr,
        double* sintheta_ptr,
        double* dS_Sig_11_uncoupled_ptr,
        double* dS_Sig_33_uncoupled_ptr,
        double* dS_costheta_ptr,
        double* dS_sintheta_ptr)
{
   
    // sigma at IP
    //double const Sig_11_ip = Sigmas_get_Sig_11(sigma_matrix_boosted_ip);
    double const Sig_12_ip = Sigmas_get_Sig_12(sigma_matrix_boosted_ip);
    double const Sig_13_ip = Sigmas_get_Sig_13(sigma_matrix_boosted_ip);
    double const Sig_14_ip = Sigmas_get_Sig_14(sigma_matrix_boosted_ip);
    double const Sig_22_ip = Sigmas_get_Sig_22(sigma_matrix_boosted_ip);
    double const Sig_23_ip = Sigmas_get_Sig_23(sigma_matrix_boosted_ip);
    double const Sig_24_ip = Sigmas_get_Sig_24(sigma_matrix_boosted_ip);
    //double const Sig_33_ip = Sigmas_get_Sig_33(sigma_matrix_boosted_ip);
    double const Sig_34_ip = Sigmas_get_Sig_34(sigma_matrix_boosted_ip);
    double const Sig_44_ip = Sigmas_get_Sig_44(sigma_matrix_boosted_ip);

    // sigma at CP
    double const Sig_11_cp = Sigmas_get_Sig_11(sigma_matrix_boosted_cp);
    double const Sig_12_cp = Sigmas_get_Sig_12(sigma_matrix_boosted_cp);
    double const Sig_13_cp = Sigmas_get_Sig_13(sigma_matrix_boosted_cp);
    double const Sig_14_cp = Sigmas_get_Sig_14(sigma_matrix_boosted_cp);
    double const Sig_22_cp = Sigmas_get_Sig_22(sigma_matrix_boosted_cp);
    double const Sig_23_cp = Sigmas_get_Sig_23(sigma_matrix_boosted_cp);
    double const Sig_24_cp = Sigmas_get_Sig_24(sigma_matrix_boosted_cp);
    double const Sig_33_cp = Sigmas_get_Sig_33(sigma_matrix_boosted_cp);
    double const Sig_34_cp = Sigmas_get_Sig_34(sigma_matrix_boosted_cp);
    double const Sig_44_cp = Sigmas_get_Sig_44(sigma_matrix_boosted_cp);

  
 
// sigmas are (co)-variances 
    double const R = Sig_11_cp - Sig_33_cp;
    double const W = Sig_11_cp + Sig_33_cp;
    double const T = R*R + 4*Sig_13_cp*Sig_13_cp;

/*
    printf("sigma R: %.20f\n", R);
    printf("sigma W: %.20f\n", W);
    printf("sigma T: %.20f\n", T);
    printf("sigma sig_x_cp: %.20f\n", Sig_11_cp);
    printf("sigma sig_y_cp: %.20f\n", Sig_33_cp);
*/

    //evaluate derivatives
    double const dS_R = 2.*(Sig_12_ip - Sig_34_ip) + 2*Sz_i*(Sig_22_ip - Sig_44_ip);
    double const dS_W = 2.*(Sig_12_ip + Sig_34_ip) + 2*Sz_i*(Sig_22_ip + Sig_44_ip);
    double const dS_Sig_13 = Sig_14_ip + Sig_23_ip + 2*Sz_i*Sig_24_ip;
    double const dS_T = 2*R*dS_R + 8.*Sig_13_ip*dS_Sig_13;

    double Sig_11_uncoupled, Sig_33_uncoupled, costheta, sintheta, dS_Sig_11_uncoupled,
           dS_Sig_33_uncoupled, dS_costheta, dS_sintheta, cos2theta, dS_cos2theta;

    double const signR = mysign(R);


    if (T<threshold_singular && handle_singularities){

        double const a = Sig_12_cp - Sig_34_cp;
        double const b = Sig_22_cp - Sig_44_cp;
        double const c = Sig_14_cp + Sig_23_cp;
        double const d = Sig_24_cp;

        double sqrt_a2_c2 = sqrt(a*a+c*c);

        if (sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2 < threshold_singular){
        //equivalent to: if np.abs(c)<threshold_singular and np.abs(a)<threshold_singular:
//            printf("first");

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

            Sig_11_uncoupled = 0.5*W;
            Sig_33_uncoupled = 0.5*W;

            dS_Sig_11_uncoupled = 0.5*dS_W;
            dS_Sig_33_uncoupled = 0.5*dS_W;
        }
        else{
//            printf("second");

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

            Sig_11_uncoupled = 0.5*W;
            Sig_33_uncoupled = 0.5*W;

            dS_Sig_11_uncoupled = 0.5*dS_W + mysign(a)*sqrt_a2_c2;
            dS_Sig_33_uncoupled = 0.5*dS_W - mysign(a)*sqrt_a2_c2;
        }
    }
    else{
//        printf("third\n");
 
        double const sqrtT = sqrt(T);
        cos2theta = signR*R/sqrtT;
        costheta = sqrt(0.5*(1.+cos2theta));
        sintheta = signR*mysign(Sig_13_cp)*sqrt(0.5*(1.-cos2theta));

        //in sixtrack this line seems to be different different
        // sintheta = -mysign((Sig_11-Sig_33))*np.sqrt(0.5*(1.-cos2theta))

        Sig_11_uncoupled = 0.5*(W+signR*sqrtT);
        Sig_33_uncoupled = 0.5*(W-signR*sqrtT); // this is 0 if N=2 macroparts in slice

/*
        printf("W: %.20f\n", W);
        printf("signR: %.20f\n", signR);
        printf("sqrtT: %.20f\n", sqrtT);
*/


        dS_cos2theta = signR*(dS_R/sqrtT - R/(2*sqrtT*sqrtT*sqrtT)*dS_T);
        dS_costheta = 1./(4.*costheta)*dS_cos2theta;

        if (fabs(sintheta)<threshold_singular && handle_singularities){
        //equivalent to to np.abs(Sig_13)<threshold_singular
            dS_sintheta = (Sig_14_cp + Sig_23_cp)/R;
        }
        else{
            dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
        }

        dS_Sig_11_uncoupled = 0.5*(dS_W + signR*0.5/sqrtT*dS_T);
        dS_Sig_33_uncoupled = 0.5*(dS_W - signR*0.5/sqrtT*dS_T);
    }

/*
    printf("sigma x: %.20f\n", Sig_11_uncoupled);
    printf("sigma y: %.20f\n", Sig_33_uncoupled);
*/

    *Sig_11_uncoupled_ptr = Sig_11_uncoupled;
    *Sig_33_uncoupled_ptr = Sig_33_uncoupled;
    *costheta_ptr = costheta;
    *sintheta_ptr = sintheta;
    *dS_Sig_11_uncoupled_ptr = dS_Sig_11_uncoupled;
    *dS_Sig_33_uncoupled_ptr = dS_Sig_33_uncoupled;
    *dS_costheta_ptr = dS_costheta;
    *dS_sintheta_ptr = dS_sintheta;

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
void StrongStrong3D_track_local_particle(StrongStrong3DData el, LocalParticle* part){

    // part= single macropart
    // el= beambeam element (other beam)	
    // Get data from memory

    // star = boosted frame (_boosted)
    // hat = at collision point (_cp)
    // bar = uncoupled frame (_uncoupled)

    // slice 2
    const double q0_bb              = StrongStrong3DData_get_q0_bb(el); // charge of slice 2 (the whole slice as a thin lens)    
    const double n_macroparts_bb    = StrongStrong3DData_get_n_macroparts_bb(el);
    const double min_sigma_diff     = StrongStrong3DData_get_min_sigma_diff(el);
    const double threshold_singular = StrongStrong3DData_get_threshold_singular(el);
    const int64_t slice_id          = StrongStrong3DData_get_slice_id(el);
    const int64_t is_sliced         = StrongStrong3DData_get_is_sliced(el);

    // sigma matrix is passed in the call
    const Sigmas sigma_matrix_boosted_ip = StrongStrong3DData_getp_sigma_matrix_ip(el); // centroid sigmas at IP
    const Sigmas sigma_matrix_boosted_cp = StrongStrong3DData_getp_sigma_matrix_cp(el); // centroid sigmas at CP
   
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp

    // loop over slice 1 macroparts    
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

        // get macropart properties, boosted and at CP (px, py, delta are the same at ip and cp)
    	double Sx_i          = LocalParticle_get_x(part);
    	double px_boosted    = LocalParticle_get_px(part);
    	double Sy_i          = LocalParticle_get_y(part);
    	double py_boosted    = LocalParticle_get_py(part);
    	double Sz_i          = LocalParticle_get_zeta(part);
    	double delta_boosted = LocalParticle_get_delta(part);

        // macropart state: 0=dead, 1=alive, 100X: part of slice X
        int64_t state = LocalParticle_get_state(part);
        state -= 1000;

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

       	    const double q0  = LocalParticle_get_q0(part); // charge of single macropart
    	    const double p0c = LocalParticle_get_p0c(part); // eV
    	    const double P0  = p0c/C_LIGHT*QELEM;

            //Compute force scaling factor: Q_bb * Q_macropart / pc
    	    const double Ksl = n_macroparts_bb*QELEM*q0_bb * QELEM*q0 / (P0 * C_LIGHT);

            // get thetas from the sigma matrix
    	    double Sig_11_boosted_cp_uncoupled, Sig_33_boosted_cp_uncoupled, costheta, sintheta;
    	    double dS_Sig_11_boosted_cp_uncoupled, dS_Sig_33_boosted_cp_uncoupled, dS_costheta, dS_sintheta;

            uncouple_xy_plane(sigma_matrix_boosted_ip, sigma_matrix_boosted_cp, Sz_i, threshold_singular, 1,
            &Sig_11_boosted_cp_uncoupled, &Sig_33_boosted_cp_uncoupled, &costheta, &sintheta,
            &dS_Sig_11_boosted_cp_uncoupled, &dS_Sig_33_boosted_cp_uncoupled, &dS_costheta, &dS_sintheta);

            // Move to the uncoupled reference frame and the derivatives of the tranformation
    	    const double Sx_i_uncoupled =  Sx_i*costheta + Sy_i*sintheta;
            const double Sy_i_uncoupled = -Sx_i*sintheta + Sy_i*costheta;
    	    const double dS_x_boosted_cp_uncoupled =  Sx_i*dS_costheta  + Sy_i*dS_sintheta;
            const double dS_y_boosted_cp_uncoupled = -Sx_i*dS_sintheta  + Sy_i*dS_costheta;

/*
            printf("step 4: uncoupled\n");
            printf("Macropart %d: x: %.10f\n", ii, Sx_i_uncoupled);
            printf("Macropart %d: y: %.10f\n", ii, Sy_i_uncoupled);
            printf("Macropart %d: sintheta: %.10f\n", ii, sintheta);
            printf("Macropart %d: costheta: %.10f\n", ii, costheta);
            printf("Macropart %d: Ksl: %.10f\n", ii, Ksl);
            printf("Macropart %d: num_particles_bb: %.10f\n", ii, n_macroparts_bb);
            printf("Macropart %d: q_bb: %.10f\n", ii, q0_bb);
            printf("Macropart %d: q_mp: %.10f\n", ii, q0);
            printf("Macropart %d: p0: %.30f\n", ii, P0);
            printf("Macropart %d: p0c: %.10f\n", ii, p0c);
            printf("Macropart %d: c: %.10f\n", ii, C_LIGHT);
            printf("Macropart %d: qelem: %.20f\n", ii, QELEM);
            printf("Macropart %d: sigma x: %.20f\n", ii, sqrt(Sig_11_boosted_cp_uncoupled));
            printf("Macropart %d: sigma y: %.20f\n", ii, sqrt(Sig_33_boosted_cp_uncoupled));
*/



    	    // Get transverse fields using soft Gausiian slice 2
    	    double Ex, Ey;
        
    	    get_Ex_Ey_gauss(Sx_i_uncoupled, Sy_i_uncoupled, // coord of slice 1 macropart at CP
    	        sqrt(Sig_11_boosted_cp_uncoupled), sqrt(Sig_33_boosted_cp_uncoupled), // beam size of slice 2 at CP
		min_sigma_diff,
    	        &Ex, &Ey);

	    //compute Gs
	    double Gx, Gy;
	    compute_Gx_Gy(Sx_i_uncoupled, Sy_i_uncoupled, sqrt(Sig_11_boosted_cp_uncoupled), sqrt(Sig_33_boosted_cp_uncoupled), min_sigma_diff, Ex, Ey, &Gx, &Gy);
	    
            // Compute kicks
            double Fx_boosted_cp_uncoupled = Ksl*Ex;
    	    double Fy_boosted_cp_uncoupled = Ksl*Ey;
    	    double Gx_boosted_cp_uncoupled = Ksl*Gx;
    	    double Gy_boosted_cp_uncoupled = Ksl*Gy;

/*
            printf("Macropart %d: fx: %.20f\n", ii, Ex);
            printf("Macropart %d: fy: %.20f\n", ii, Ey);
            printf("Macropart %d: gx: %.20f\n", ii, Gx);
            printf("Macropart %d: gx: %.20f\n", ii, Gy);
*/

            // Move kisks to coupled reference frame
    	    double Fx_boosted = Fx_boosted_cp_uncoupled*costheta - Fy_boosted_cp_uncoupled*sintheta;
    	    double Fy_boosted = Fx_boosted_cp_uncoupled*sintheta + Fy_boosted_cp_uncoupled*costheta;

    	    // Compute longitudinal kick
    	    double Fz_boosted = 0.5*(Fx_boosted_cp_uncoupled*dS_x_boosted_cp_uncoupled      + Fy_boosted_cp_uncoupled*dS_y_boosted_cp_uncoupled +
    	                         Gx_boosted_cp_uncoupled*dS_Sig_11_boosted_cp_uncoupled + Gy_boosted_cp_uncoupled*dS_Sig_33_boosted_cp_uncoupled);

    	    // Apply the kicks (Hirata's synchro-beam)
    	    delta_boosted +=      Fz_boosted + 0.5*(Fx_boosted*(px_boosted + 0.5*Fx_boosted) + Fy_boosted*(py_boosted + 0.5*Fy_boosted)); 
    	    px_boosted    +=      Fx_boosted;
    	    py_boosted    +=      Fy_boosted;

//          printf("Macropart %d:", ii+1);
//          printf(" Fx: %.10f,", Fx_boosted); 
//          printf(" Fy: %.10f,", Fy_boosted); 
//          printf(" Fz: %.30f\n", Fz_boosted); 
     
            // only momenta are updated, coords will be drifted back to IP with new momentum
    	    LocalParticle_set_x(part, Sx_i);
    	    LocalParticle_set_px(part, px_boosted);
    	    LocalParticle_set_y(part, Sy_i);
    	    LocalParticle_set_py(part, py_boosted);
    	    LocalParticle_set_zeta(part, Sz_i);
    	    LocalParticle_update_delta(part, delta_boosted);
	
        }
    } //only_for_context cpu_serial cpu_openmp
}


#endif
