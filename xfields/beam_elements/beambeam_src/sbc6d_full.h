#ifndef XFIELDS_SBC6D_FULL_H
#define XFIELDS_SBC6D_FULL_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

/*gpufun*/
void uncouple_xy_plane(
        double const Sig_11_ip,
        double const Sig_12_ip,
        double const Sig_13_ip,
        double const Sig_14_ip,
        double const Sig_22_ip,
        double const Sig_23_ip,
        double const Sig_24_ip,
        double const Sig_33_ip,
        double const Sig_34_ip,
        double const Sig_44_ip,
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
    //printf("sig_11: %.10e\n", Sig_11_ip);
    //printf("sig_33: %.10e\n", Sig_33_ip);    
    // sigma at CP
    // Propagate sigma matrix
    double const Sig_11_cp = Sig_11_ip + 2.*Sig_12_ip*Sz_i            + Sig_22_ip*Sz_i*Sz_i;
    double const Sig_33_cp = Sig_33_ip + 2.*Sig_34_ip*Sz_i            + Sig_44_ip*Sz_i*Sz_i;
    double const Sig_13_cp = Sig_13_ip + (Sig_14_ip + Sig_23_ip)*Sz_i + Sig_24_ip*Sz_i*Sz_i;
    double const Sig_12_cp = Sig_12_ip + Sig_22_ip*Sz_i;
    double const Sig_14_cp = Sig_14_ip + Sig_24_ip*Sz_i;
    double const Sig_22_cp = Sig_22_ip;
    double const Sig_23_cp = Sig_23_ip + Sig_24_ip*Sz_i;
    double const Sig_24_cp = Sig_24_ip;
    double const Sig_34_cp = Sig_34_ip + Sig_44_ip*Sz_i;
    double const Sig_44_cp = Sig_44_ip;

/*
    printf("sigma_xx at IP: %.20f\n", Sig_11_ip);
    printf("sigma_xpx at IP: %.20f\n", Sig_12_ip);
    printf("sigma_pxpx at IP: %.20f\n", Sig_22_ip);
    printf("S: %.20f\n", Sz_i);

    printf("sigma_xx at CP: %.20f\n", Sig_11_cp);
    printf("sigma_xpx at CP: %.20f\n", Sig_12_cp);
    printf("sigma_xy at CP: %.20f\n", Sig_13_cp);
    printf("sigma_xpy at CP: %.20f\n", Sig_14_cp);
    printf("sigma_pxpx at CP: %.20f\n", Sig_22_cp);
    printf("sigma_pxy at CP: %.20f\n", Sig_23_cp);
    printf("sigma_pxpy at CP: %.20f\n", Sig_24_cp);
    printf("sigma_yy at CP: %.20f\n", Sig_33_cp);
    printf("sigma_ypy at CP: %.20f\n", Sig_34_cp);
    printf("sigma_pypy at CP: %.20f\n", Sig_44_cp);
*/
 
// sigmas are (co)-variances 
    double const R = Sig_11_cp - Sig_33_cp;
    double const W = Sig_11_cp + Sig_33_cp;
    double const T = R*R + 4*Sig_13_cp*Sig_13_cp;

/*
    printf("sigma R: %.20f\n", R);
    printf("sigma W: %.20f\n", W);
    printf("sigma T: %.20f\n", T);
    printf("sigma sig_xx_cp: %.20f\n", Sig_11_cp);
    printf("sigma sig_yy_cp: %.20f\n", Sig_33_cp);
    printf("sigma sig_xy_cp: %.20f\n", Sig_13_cp);
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
        printf("R: %.20f\n", R);
        printf("signR: %.20f\n", signR);
        printf("sqrtT: %.20f\n", sqrtT);
        printf("cos2theta: %.20f\n", cos2theta);
        printf("sintheta: %.20f\n", sintheta);
        printf("dS_cos2theta: %.20f\n", dS_cos2theta);
*/
        dS_cos2theta = signR*(dS_R/sqrtT - R/(2*sqrtT*sqrtT*sqrtT)*dS_T);
        dS_costheta = 1./(4.*costheta)*dS_cos2theta;

        if (fabs(sintheta)<threshold_singular && handle_singularities){
        //equivalent to to np.abs(Sig_13)<threshold_singular
         //   printf("this");
            dS_sintheta = (Sig_14_cp + Sig_23_cp)/R;
        }
        else{
           // printf("that");
            dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
        }
        dS_Sig_11_uncoupled = 0.5*(dS_W + signR*0.5/sqrtT*dS_T);
        dS_Sig_33_uncoupled = 0.5*(dS_W - signR*0.5/sqrtT*dS_T);
    }
/*
    printf("dS_sintheta: %.20f\n", dS_sintheta);
    printf("sigma x: %.20f\n", Sig_11_uncoupled);
    printf("sigma y: %.20f\n", Sig_33_uncoupled);
*/
    //printf("sig_11_uncoupled: %.10e\n", Sig_11_uncoupled);
    //printf("sig_33_uncoupled: %.10e\n", Sig_33_uncoupled);
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


void MacropartToCP(const unsigned int use_strongstrong,
                   double x1, double y1, const double z1, double px1, double py1,
                   const double x2_c, const double y2_c, const double z2_c, const double px2_c, const double py2_c,// boosted centroid of other slice at IP (= vector pointing from my full beam/slice centoid to other beam slice centroid)
                   const double x2_bc, const double y2_bc, // boosted centroid of other full beam (or slice, when ref frames are transformed slice by slice, so same as x_c) at IP (= vector pointing from my full beam/slice centroid to other full beam/slice centroid), w.r.t other full beam centroid
                   double* Sx_i, double* Sy_i, double* Sz_i){

    // in order to move macroparts of slice 1 to their CP need to have centroid of slice 2 in advance
    // for this slice 2 is collapsed onto the Z of the centroid as a thin lens
    // each macropart of slice 1 will interact with this thin lens at CP
    (*Sz_i) = 0.5*(z1 - z2_c); // CP for macropart

    // weakstrong model: center of ref. is full beam 2 centroid: x^cp_weak_macropart + xc_strong_fullbeam - xc^cp_strong_slice = Sx_i w.r.t. strong slice centroid at CP  
    if (use_strongstrong == 0){

        // if beams have 1 slice, then x2_bc = x2_c
        (*Sx_i) = (x1 + px1*(*Sz_i)) + x2_bc - (x2_c - px2_c*(*Sz_i)); 
        (*Sy_i) = (y1 + py1*(*Sz_i)) + y2_bc - (y2_c - py2_c*(*Sz_i));

    // strongstorng model: center of ref. is the barycentric frame with common origin: x^cp_beam1_macropart - xc^cp_beam2_slice = Sx_i w.r.t. strong slice centroid at CP  
    }else if (use_strongstrong == 1){
        (*Sx_i) = (x1 + px1*(*Sz_i)) - (x2_c - px2_c*(*Sz_i)); 
        (*Sy_i) = (y1 + py1*(*Sz_i)) - (y2_c - py2_c*(*Sz_i));
    }
}


void MacropartToIP(const unsigned int use_strongstrong,
                   double Sx_i, double Sy_i, double Sz_i, double px1, double py1,
                   const double x2_c, const double y2_c, const double z2_c, const double px2_c, const double py2_c,
                   const double x2_bc, const double y2_bc,
                   double* x1, double* y1, double* z1){

    // IP for macropart, here we get back the centroid of beam 1
    (*z1) = 2*Sz_i + z2_c; 

    // weakstrong model: center of ref. is full beam 2 centroid: x^cp_weak_macropart + xc_strong_fullbeam - xc^cp_strong_slice = Sx_i w.r.t. strong slice centroid at CP   
    if (use_strongstrong == 0){        

        // if beams have 1 slice, then x2_bc = x2_c
        (*x1) = (Sx_i - px1*Sz_i) - x2_bc + (x2_c - px2_c*Sz_i); 
        (*y1) = (Sy_i - py1*Sz_i) - y2_bc + (y2_c - py2_c*Sz_i);

    // strongstorng model: center of ref. is the barycentric frame with common origin: x^cp_beam1_macropart - xc^cp_beam2_slice = Sx_i w.r.t. strong slice centroid at CP  
    }else if (use_strongstrong == 1){

        (*x1) = (Sx_i - px1*Sz_i) + (x2_c - px2_c*Sz_i); 
        (*y1) = (Sy_i - py1*Sz_i) + (y2_c - py2_c*Sz_i);
    }
}



/*gpufun*/
void Sbc6D_full_track_local_particle(Sbc6D_fullData el, LocalParticle* part0){
//    clock_t tt;
//    tt = clock(); 
	
    // part0 = single macropart
    // el    = beambeam element (other beam)	
    // star = boosted frame (_boosted)
    // hat = at collision point (_cp)
    // bar = uncoupled frame (_uncoupled)

    // slice 2
    const double min_sigma_diff            = Sbc6D_fullData_get_min_sigma_diff(el);
    const double threshold_singular        = Sbc6D_fullData_get_threshold_singular(el);
    const int64_t do_beamstrahlung         = Sbc6D_fullData_get_do_beamstrahlung(el);
    const int64_t verbose_info             = Sbc6D_fullData_get_verbose_info(el); // print turn number
    const unsigned int use_strongstrong    = Sbc6D_fullData_get_use_strongstrong(el);
    const double q0_bb                     = Sbc6D_fullData_get_q0_bb(el);  // +/-1

    const int64_t timestep = Sbc6D_fullData_get_timestep(el);
    const int64_t n_slices = Sbc6D_fullData_get_n_slices(el);

    double energy_loss;

    // arrays of boosted slice moments and intensities of other beam
    /*gpuglmem*/ const double* n_bb_per_slice_arr      = Sbc6D_fullData_getp1_n_bb(el, 0);
    /*gpuglmem*/ const double* mean_x_per_slice_arr    = Sbc6D_fullData_getp1_mean_x(el, 0); 
    /*gpuglmem*/ const double* mean_xp_per_slice_arr   = Sbc6D_fullData_getp1_mean_xp(el, 0);
    /*gpuglmem*/ const double* mean_y_per_slice_arr    = Sbc6D_fullData_getp1_mean_y(el, 0);
    /*gpuglmem*/ const double* mean_yp_per_slice_arr   = Sbc6D_fullData_getp1_mean_yp(el, 0);
    /*gpuglmem*/ const double* mean_z_per_slice_arr    = Sbc6D_fullData_getp1_mean_z(el, 0);
    /*gpuglmem*/ const double* var_x_per_slice_arr     = Sbc6D_fullData_getp1_var_x(el, 0);
    /*gpuglmem*/ const double* cov_x_xp_per_slice_arr  = Sbc6D_fullData_getp1_cov_x_xp(el, 0);
    /*gpuglmem*/ const double* cov_x_y_per_slice_arr   = Sbc6D_fullData_getp1_cov_x_y(el, 0);
    /*gpuglmem*/ const double* cov_x_yp_per_slice_arr  = Sbc6D_fullData_getp1_cov_x_yp(el, 0);
    /*gpuglmem*/ const double* var_xp_per_slice_arr    = Sbc6D_fullData_getp1_var_xp(el, 0);
    /*gpuglmem*/ const double* cov_xp_y_per_slice_arr  = Sbc6D_fullData_getp1_cov_xp_y(el, 0);
    /*gpuglmem*/ const double* cov_xp_yp_per_slice_arr = Sbc6D_fullData_getp1_cov_xp_yp(el, 0);
    /*gpuglmem*/ const double* var_y_per_slice_arr     = Sbc6D_fullData_getp1_var_y(el, 0);
    /*gpuglmem*/ const double* cov_y_yp_per_slice_arr  = Sbc6D_fullData_getp1_cov_y_yp(el, 0);
    /*gpuglmem*/ const double* var_yp_per_slice_arr    = Sbc6D_fullData_getp1_var_yp(el, 0);
    /*gpuglmem*/ const double* var_z_per_slice_arr     = Sbc6D_fullData_getp1_var_z(el, 0);

    const double* x_full_bb_centroid  = Sbc6D_fullData_getp_x_full_bb_centroid(el);
    const double* y_full_bb_centroid  = Sbc6D_fullData_getp_y_full_bb_centroid(el);

    int64_t count = 0;
    //printf("[sbc6d_full]");

    //start_per_particle_block (part0->part)

        // macropart state: 0=dead, 1=alive
        int64_t state       = LocalParticle_get_state(part);
        int64_t slice_id    = LocalParticle_get_slice_id(part);
        int64_t slice_id_bb = timestep - slice_id;  // need to use sigma of this slice

        //printf("n_slices: %d\nslice_id: %d\nslice_id_bb: %d\n", n_slices, slice_id, slice_id_bb);
        // code is executed only if macropart is alive and interacts with a valid slice 
        if(slice_id_bb>=0 && slice_id_bb<n_slices && state!=0){

            count += 1;

            // get moments of colliding slice
            const double n_bb         =      n_bb_per_slice_arr[slice_id_bb];
            const double mean_x_bb    =    mean_x_per_slice_arr[slice_id_bb]; 
            const double mean_xp_bb   =   mean_xp_per_slice_arr[slice_id_bb];
            const double mean_y_bb    =    mean_y_per_slice_arr[slice_id_bb];
            const double mean_yp_bb   =   mean_yp_per_slice_arr[slice_id_bb];
            const double mean_z_bb    =    mean_z_per_slice_arr[slice_id_bb];
            const double var_x_bb     =     var_x_per_slice_arr[slice_id_bb];
            const double cov_x_xp_bb  =  cov_x_xp_per_slice_arr[slice_id_bb];
            const double cov_x_y_bb   =   cov_x_y_per_slice_arr[slice_id_bb];
            const double cov_x_yp_bb  =  cov_x_yp_per_slice_arr[slice_id_bb];
            const double var_xp_bb    =    var_xp_per_slice_arr[slice_id_bb];
            const double cov_xp_y_bb  =  cov_xp_y_per_slice_arr[slice_id_bb];
            const double cov_xp_yp_bb = cov_xp_yp_per_slice_arr[slice_id_bb];
            const double var_y_bb     =     var_y_per_slice_arr[slice_id_bb];
            const double cov_y_yp_bb  =  cov_y_yp_per_slice_arr[slice_id_bb];
            const double var_yp_bb    =    var_yp_per_slice_arr[slice_id_bb];
            const double var_z_bb     =     var_z_per_slice_arr[slice_id_bb];

            //printf("q0_bb: %f\n", q0_bb);
            //printf("mean_x_bb: %f\n", mean_x_bb);
            //printf("mean_y_bb: %f\n", mean_y_bb);
            //printf("var_x_bb: %f\n", var_x_bb);
            //printf("cov_y_yp_bb: %f\n", cov_y_yp_bb);

            // get macropart properties, boosted and at IP 
       	    double x     = LocalParticle_get_x(part);
    	    double px    = LocalParticle_get_px(part);
    	    double y     = LocalParticle_get_y(part);
    	    double py    = LocalParticle_get_py(part);
    	    double z     = LocalParticle_get_zeta(part);
    	    double delta = LocalParticle_get_delta(part);

            // transport IP to CP (px, py, delta are the same at ip and cp)
            double Sx_i, Sy_i, Sz_i;
            MacropartToCP(use_strongstrong, x, y, z, px, py,
                      mean_x_bb, mean_y_bb, mean_z_bb, mean_xp_bb, mean_yp_bb,
                      *x_full_bb_centroid, *y_full_bb_centroid,
                      &Sx_i, &Sy_i, &Sz_i);
  
            //Compute force scaling factor: Q_bb * Q_macropart / pc [C^2 C-1 m s-1] 
       	    const double q0  = LocalParticle_get_q0(part);  // +/-1
    	    const double p0c = LocalParticle_get_p0c(part); // ref. energy [eV]
    	    const double P0  = p0c/C_LIGHT*QELEM;  // ref. momentum, electronvolt to joule conversion
   	    const double Ksl = (QELEM*q0_bb*n_bb)*(QELEM*q0) / (P0 * C_LIGHT);
            //printf("QELEM: %.10e, p0c: %.10e, P0: %.10e, C_LIGHT: %.10e, Ksl: %.10e, q0: %.10e, q0_bb: %.10e\n", QELEM, p0c, P0, C_LIGHT, Ksl, q0, q0_bb);

            // get thetas from the sigma matrix
    	    double Sig_11_boosted_cp_uncoupled, Sig_33_boosted_cp_uncoupled, costheta, sintheta;
    	    double dS_Sig_11_boosted_cp_uncoupled, dS_Sig_33_boosted_cp_uncoupled, dS_costheta, dS_sintheta;

            uncouple_xy_plane(var_x_bb, cov_x_xp_bb, cov_x_y_bb, cov_x_yp_bb, var_xp_bb, cov_xp_y_bb, cov_xp_yp_bb, var_y_bb, cov_y_yp_bb, var_yp_bb,
            Sz_i, threshold_singular, 1,
            &Sig_11_boosted_cp_uncoupled, &Sig_33_boosted_cp_uncoupled, &costheta, &sintheta,
            &dS_Sig_11_boosted_cp_uncoupled, &dS_Sig_33_boosted_cp_uncoupled, &dS_costheta, &dS_sintheta);

            // Move to the uncoupled reference frame and the derivatives of the tranformation
    	    const double Sx_i_uncoupled =  Sx_i*costheta + Sy_i*sintheta;
            const double Sy_i_uncoupled = -Sx_i*sintheta + Sy_i*costheta;
    	    const double dS_x_boosted_cp_uncoupled =  Sx_i*dS_costheta  + Sy_i*dS_sintheta;
            const double dS_y_boosted_cp_uncoupled = -Sx_i*dS_sintheta  + Sy_i*dS_costheta;

    	    // Get transverse fields using soft Gausiian slice 2
    	    double Ex, Ey;
    	    get_Ex_Ey_gauss(Sx_i_uncoupled, Sy_i_uncoupled, // coord of slice 1 macropart at CP
    	        sqrt(Sig_11_boosted_cp_uncoupled), sqrt(Sig_33_boosted_cp_uncoupled), // beam size of slice 2 at CP
		min_sigma_diff,
    	        &Ex, &Ey);

	    //compute Gs
	    double Gx, Gy;
	    compute_Gx_Gy(Sx_i_uncoupled, Sy_i_uncoupled,
                sqrt(Sig_11_boosted_cp_uncoupled), sqrt(Sig_33_boosted_cp_uncoupled), min_sigma_diff,
                Ex, Ey, &Gx, &Gy);
	    
            // Compute kicks
            double Fx_boosted_cp_uncoupled = Ksl*Ex;
    	    double Fy_boosted_cp_uncoupled = Ksl*Ey;
    	    double Gx_boosted_cp_uncoupled = Ksl*Gx;
    	    double Gy_boosted_cp_uncoupled = Ksl*Gy;

            // Move disks to coupled reference frame
    	    double Fx_boosted = Fx_boosted_cp_uncoupled*costheta - Fy_boosted_cp_uncoupled*sintheta;
    	    double Fy_boosted = Fx_boosted_cp_uncoupled*sintheta + Fy_boosted_cp_uncoupled*costheta;
    	    double Fz_boosted = 0.5*(Fx_boosted_cp_uncoupled*dS_x_boosted_cp_uncoupled      + Fy_boosted_cp_uncoupled*dS_y_boosted_cp_uncoupled +
    	                         Gx_boosted_cp_uncoupled*dS_Sig_11_boosted_cp_uncoupled + Gy_boosted_cp_uncoupled*dS_Sig_33_boosted_cp_uncoupled);

            // bs is a consequence of kick: emit beamstrahlung photons from single macropart
            if (do_beamstrahlung==1){
          
                // total kick 
                double const Fr = hypot(Fx_boosted, Fy_boosted) * LocalParticle_get_rpp(part); // rad

                // longitudinal distance
                /*gpuglmem*/ const double* dz_arr = Sbc6D_fullData_getp1_dz(el, 0);
                //const double dz = 0.0726;
                const double dz = dz_arr[slice_id_bb]/2.;
                double initial_energy = (LocalParticle_get_energy0(part) + LocalParticle_get_psigma(part)*LocalParticle_get_p0c(part)*LocalParticle_get_beta0(part)); 
                FILE *f1 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_force.txt", "a");
                fprintf(f1, "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", x, y, z, Sx_i_uncoupled, Sy_i_uncoupled, Sz_i, Sig_11_boosted_cp_uncoupled, Sig_33_boosted_cp_uncoupled, Ksl, Fx_boosted, Fy_boosted, px, py, px+Fx_boosted, py+Fy_boosted, dz, initial_energy, Fx_boosted_cp_uncoupled, Fy_boosted_cp_uncoupled, sintheta, costheta, n_bb);
                fclose(f1);
                energy_loss = synrad(part, Fr, dz);
 
                // BS rescales these, so load again before kick 
    	        delta = LocalParticle_get_delta(part);  
            }
            else if(do_beamstrahlung==2){
                synrad_avg(part, n_bb, sqrt(Sig_11_boosted_cp_uncoupled), sqrt(Sig_33_boosted_cp_uncoupled), sqrt(var_z_bb));  // slice intensity and RMS slice sizes
                delta = LocalParticle_get_delta(part);  
            }

            delta += Fz_boosted + 0.5*(Fx_boosted*(px + 0.5*Fx_boosted) + Fy_boosted*(py + 0.5*Fy_boosted));
     	    LocalParticle_add_to_px(part, Fx_boosted);
     	    LocalParticle_add_to_py(part, Fy_boosted);
     	    LocalParticle_update_delta(part, delta);
     	    LocalParticle_add_to_x(part, -Sz_i*Fx_boosted);
     	    LocalParticle_add_to_y(part, -Sz_i*Fy_boosted);

      	    // Apply the kicks (Hirata's synchro-beam)
    	    //delta +=      Fz_boosted + 0.5*(Fx_boosted*(px + 0.5*Fx_boosted) + Fy_boosted*(py + 0.5*Fy_boosted)); 
    	    //px    +=      Fx_boosted;
    	    //py    +=      Fy_boosted;
 
            //x_new -= Sz_i*Fx_boosted;
            //y_new -= Sz_i*Fy_boosted;

            //double x_new, y_new, z_new;
            //MacropartToIP(use_strongstrong, Sx_i, Sy_i, Sz_i, px, py,
            //      mean_x_bb, mean_y_bb, mean_z_bb, mean_xp_bb, mean_yp_bb, 
            //      *x_full_bb_centroid, *y_full_bb_centroid,
            //      &x_new, &y_new, &z_new);

            // update due to kick, coords will be drifted back to IP with new momentum
    	    //LocalParticle_set_x(part, x_new);
    	    //LocalParticle_set_px(part, px);
    	    //LocalParticle_set_y(part, y_new);
    	    //LocalParticle_set_py(part, py);
    	    //LocalParticle_set_zeta(part, z_new);  // z gets rescaled in BS
    	    //LocalParticle_update_delta(part, delta);

        }
    //end_per_particle_block
    //printf("count: %d\n", count);

   

}

#endif
