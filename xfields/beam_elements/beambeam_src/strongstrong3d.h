#ifndef XFIELDS_STRONGSTRONG3D_H
#define XFIELDS_STRONGSTRONG3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

#define rndm_synrad() rndm7()

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
    double const Sig_11_ip = Sigmas_get_Sig_11(sigma_matrix_boosted_ip);
    double const Sig_12_ip = Sigmas_get_Sig_12(sigma_matrix_boosted_ip);
    double const Sig_13_ip = Sigmas_get_Sig_13(sigma_matrix_boosted_ip);
    double const Sig_14_ip = Sigmas_get_Sig_14(sigma_matrix_boosted_ip);
    double const Sig_22_ip = Sigmas_get_Sig_22(sigma_matrix_boosted_ip);
    double const Sig_23_ip = Sigmas_get_Sig_23(sigma_matrix_boosted_ip);
    double const Sig_24_ip = Sigmas_get_Sig_24(sigma_matrix_boosted_ip);
    double const Sig_33_ip = Sigmas_get_Sig_33(sigma_matrix_boosted_ip);
    double const Sig_34_ip = Sigmas_get_Sig_34(sigma_matrix_boosted_ip);
    double const Sig_44_ip = Sigmas_get_Sig_44(sigma_matrix_boosted_ip);

/*
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
*/
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
int synrad_0(double e_macropart,
             double dz,
             double rho_inv,  // [1/m] changes after each photon emission
             double* e_photon){ 


    /*
    return 0: no photon
    return 1: emit 1 photon, energy stored in photon_energy
    */

    // constants for approximating synrad spectrum
    const double g1_a[5] = {1.0, -0.8432885317, 0.1835132767, -0.0527949659, 0.0156489316};
    const double g2_a[5] = {0.4999456517, -0.5853467515, 0.3657833336, -0.0695055284, 0.019180386};
    const double g1_b[7] = {2.066603927, -0.5718025331, 0.04243170587, -0.9691386396, 5.651947051, -0.6903991322, 1.0};
    const double g2_b[7] = {1.8852203645, -0.5176616313, 0.03812218492, -0.49158806, 6.1800441958, -0.6524469236, 1.0};
    const double g1_c[4] = {1.0174394594, 0.5831679349, 0.9949036186, 1.0};
    const double g2_c[4] = {0.2847316689, 0.58306846, 0.3915531539, 1.0};

    double c1 = 1.5*HBAR_GEVS / pow(MELECTRON_GEVPERC, 3) * C_LIGHT;  // [c^4/Gev^2] 2.22e-6 = 1.5*hbar*cst.c/e0**3 
    double xcrit = c1 * pow(e_macropart, 2) * rho_inv;  // [1] ecrit/E magnitude of quantum correction, in guineapig: xcrit(C)=ξ(doc)=upsbar(C++)
    double ecrit = xcrit * e_macropart;  // [GeV]
    double omega_crit = ecrit/HBAR_GEVS;  // [1/s] = 1.5 * gamma**3 * cst.c / rho
    double upsilon = 2/3 * ecrit / e_macropart;  // [1] beamstrahlung parameter for single macropart


    double p0 = 25.4 * e_macropart * dz * rho_inv;  // [1]  Fr * dz, specific for 1 macropart

    // eliminate region A in p0*g-v plane (=normalize with p0 = reject 1-p0 (p0<1) fraction of cases = y axis of p0*g-v plane is now spanning 0--p0=1
    if (rndm_synrad() > p0){return 0;}
    
    // 2 random numbers to calculate g(v, xcrit)
    double p = rndm_synrad();  // if this is 1, then it corresponds to p0 on original p0*g-v plane
    double v;
    while((v=rndm_synrad())==0.0) ; // draw a nonzero random number, variable of the beamstrahlung spectrum
    
    double v2 = v*v;
    double v3 = v2*v;
    double y = v3 / (1.0 - v3);
    double denom = 1 - ( 1 - xcrit ) * v3;
    
    // calculate synrad spectrum coefficients, depending the value of y
    double g1, g2;
    if (y <= 1.54){
        g1 = pow(y, -2/3) * (g1_a[0] + g1_a[1]*pow(y, 2/3) + g1_a[2]*pow(y, 2) + g1_a[3]*pow(y, 10/3) + g1_a[4]*pow(y, 4));
        g2 = pow(y, -2/3) * (g2_a[0] + g2_a[1]*pow(y, 4/3) + g2_a[2]*pow(y, 2) + g2_a[3]*pow(y, 10/3) + g2_a[4]*pow(y, 4));
    }else if (y <= 4.48){
        g1 = ( g1_b[0] + g1_b[1]*y + g1_b[2]*pow(y, 2) ) / ( g1_b[3] + g1_b[4]*y + g1_b[5]*pow(y, 2) + g1_b[6]*pow(y, 3) );
        g2 = ( g2_b[0] + g2_b[1]*y + g2_b[2]*pow(y, 2) ) / ( g2_b[3] + g2_b[4]*y + g2_b[5]*pow(y, 2) + g2_b[6]*pow(y, 3) );
    }else if (y <= 165){
        g1 = exp(-y)/sqrt(y) * ( g1_c[0] + g1_c[1]*y ) / ( g1_c[2] + g1_c[3]*y );
        g2 = exp(-y)/sqrt(y) * ( g2_c[0] + g2_c[1]*y ) / ( g2_c[2] + g2_c[3]*y );
    }else{
        // no radiation, y too high
        return 0;
    }
        
    // g normalized (g(v=0, xcrit)=1=p0), g(v, xcrit) gives the no. of emitted photons in a fiven delta v interval
    double g = v2 / pow(denom, 2) * ( g1 + ( pow(xcrit, 2) * pow(y, 2) ) / ( 1 + xcrit * y ) * g2 );  // g (w.o. normalization above) splits the unit rectangle p0*g-v to A,B,C regions
    
    // region C (emit photon) if p<p0*g, region B (no photon) if p>=p0*g, p0=1 bc. of normalization above
    if (p<g){
        *e_photon = ecrit * v3 / denom;
        return 1;
    }else{
        *e_photon = 0.0;
        return 0;
    }
}



/*gpufun*/
void synrad(double e_macropart, // [GeV]
            double gamma,  // [1] 
       	    double Fx_boosted,  // [1] (px' - px)/Dt, Dt=1
            double Fy_boosted,  // [1] (py' - py)/Dt, Dt=1
	    double dz,  // [m] z step between 2 slices ((z_max - z_min) / 2)
	    double* e_photon_array // [GeV] array to store energy of emitted photons
){

    double r = pow(QELEM, 2)/(4* PI * EPSILON_0 * MELECTRON_KG * pow(C_LIGHT, 2));  // [m] electron radius
    
    double ax = dz/(e_macropart) * Fx_boosted;
    double ay = dz/(e_macropart) * Fy_boosted;
    double rho_inv = sqrt(ax*ax + ay*ay) / dz;  // [m] (Fr/E) macropart bending radius from bb kick: dz/rho = dz/E*sqrt((px' - px)**2 + (py' - py)**2) = Fr * dz / E

    double tmp = 25.4 * e_macropart * dz * rho_inv;  // [1]  Fr * dz, specific for 1 macropart
    int max_photons = (int)(tmp*10.0)+1;

    // photons are emitted uniformly in space along dz (between 2 slice interactions)
    dz /= max_photons;
    
    int j = 0;
    // emit photons in a loop
    for (int i=0; i<max_photons; i++){
    
        // see if photon emitted

        double e_photon;
        if (synrad_0(e_macropart, dz, rho_inv, &e_photon)){
            e_photon_array[j] = e_photon;
            
            // update bending radius
            rho_inv *= e_macropart/(e_macropart - e_photon);

            // update macropart energy
            e_macropart -= e_photon;

            // one more photon
            j++;
           
            // some error handling
            if (e_photon_array[j]<=0.0){
		printf("photon emitted with negative energy: E_photon=%g, E_macropart=%g, photon ID: %d, limit: %d\n", e_photon, e_macropart, j, max_photons);
       	    }
       	    if (j>=1000){
		printf("too many photons produced by one particle (photon ID: %d)\n", j);
		exit(-1);
	    }

        }
    }
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
    const int64_t do_beamstrahlung  = StrongStrong3DData_get_do_beamstrahlung(el);
    const int64_t verbose_info  = StrongStrong3DData_get_verbose_info(el); // print turn number

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
        // state -= 1000;

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

       	    const double q0  = LocalParticle_get_q0(part); // charge of single macropart [elementary charge]
    	    const double p0c = LocalParticle_get_p0c(part); // [eV]
    	    const double P0  = p0c/C_LIGHT*QELEM;  // [C]

            //Compute force scaling factor: Q_bb * Q_macropart / pc [C^2 C-1 m s-1]
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

            // Move disks to coupled reference frame
    	    double Fx_boosted = Fx_boosted_cp_uncoupled*costheta - Fy_boosted_cp_uncoupled*sintheta;
    	    double Fy_boosted = Fx_boosted_cp_uncoupled*sintheta + Fy_boosted_cp_uncoupled*costheta;

    	    // Compute longitudinal kick
    	    double Fz_boosted = 0.5*(Fx_boosted_cp_uncoupled*dS_x_boosted_cp_uncoupled      + Fy_boosted_cp_uncoupled*dS_y_boosted_cp_uncoupled +
    	                         Gx_boosted_cp_uncoupled*dS_Sig_11_boosted_cp_uncoupled + Gy_boosted_cp_uncoupled*dS_Sig_33_boosted_cp_uncoupled);
            if (ii == -1){
                printf("Turn: %d macropart %d Sx_i: %.10f\n",                           verbose_info, ii,                              Sx_i);
                printf("Turn: %d macropart %d px_i: %.10f\n",                           verbose_info, ii,                        px_boosted);
                printf("Turn: %d macropart %d Sy_i: %.10f\n",                           verbose_info, ii,                              Sy_i);
                printf("Turn: %d macropart %d py_i: %.10f\n",                           verbose_info, ii,                        py_boosted);
                printf("Turn: %d macropart %d Sz_i: %.10f\n",                           verbose_info, ii,                              Sz_i);
                printf("Turn: %d macropart %d delta_i_: %.10f\n",                       verbose_info, ii,                     delta_boosted); 
                printf("Turn: %d macropart %d Sx_i_uncoupled: %.10f\n",                 verbose_info, ii,                    Sx_i_uncoupled);
                printf("Turn: %d macropart %d Sy_i_uncoupled: %.10f\n",                 verbose_info, ii,                    Sy_i_uncoupled);
                printf("Turn: %d macropart %d fx: %.20f\n",                             verbose_info, ii,                                Ex);
                printf("Turn: %d macropart %d fy: %.20f\n",                             verbose_info, ii,                                Ey);
                printf("Turn: %d macropart %d gx: %.20f\n",                             verbose_info, ii,                                Gx);
                printf("Turn: %d macropart %d gy: %.20f\n",                             verbose_info, ii,                                Gy);
                printf("Turn: %d macropart %d sintheta: %.10f\n",                       verbose_info, ii,                          sintheta);
                printf("Turn: %d macropart %d costheta: %.10f\n",                       verbose_info, ii,                          costheta);
                printf("Turn: %d macropart %d dS_sintheta: %.10f\n",                    verbose_info, ii,                       dS_sintheta);
                printf("Turn: %d macropart %d dS_costheta: %.10f\n",                    verbose_info, ii,                       dS_costheta);
                printf("Turn: %d macropart %d Ksl: %.10f\n",                            verbose_info, ii,                               Ksl);
                printf("Turn: %d macropart %d num_particles_bb: %.10f\n",               verbose_info, ii,                   n_macroparts_bb);
                printf("Turn: %d macropart %d q_bb: %.10f\n",                           verbose_info, ii,                             q0_bb);
                printf("Turn: %d macropart %d q_mp: %.10f\n",                           verbose_info, ii,                                q0);
                printf("Turn: %d macropart %d p0: %.30f\n",                             verbose_info, ii,                                P0);
                printf("Turn: %d macropart %d p0c: %.10f\n",                            verbose_info, ii,                               p0c);
                printf("Turn: %d macropart %d c: %.10f\n",                              verbose_info, ii,                           C_LIGHT);
                printf("Turn: %d macropart %d qelem: %.20f\n",                          verbose_info, ii,                             QELEM);
                printf("Turn: %d macropart %d Sig_11_boosted_cp_uncoupled: %.20f\n",    verbose_info, ii, sqrt(Sig_11_boosted_cp_uncoupled));
                printf("Turn: %d macropart %d Sig_33_boosted_cp_uncoupled: %.20f\n",    verbose_info, ii, sqrt(Sig_33_boosted_cp_uncoupled));
                printf("Turn: %d macropart %d Fx_boosted: %.10f\n",                     verbose_info, ii,                        Fx_boosted);
                printf("Turn: %d macropart %d Fy_boosted: %.10f\n",                     verbose_info, ii,                        Fy_boosted);
                printf("Turn: %d macropart %d Fz_boosted: %.10f\n",                     verbose_info, ii,                        Fz_boosted);
            	printf("Turn: %d macropart %d dS_x_boosted_cp_uncoupled: %.10f\n",      verbose_info, ii,         dS_x_boosted_cp_uncoupled);
                printf("Turn: %d macropart %d dS_y_boosted_cp_uncoupled: %.10f\n",      verbose_info, ii,         dS_y_boosted_cp_uncoupled);
                printf("Turn: %d macropart %d dS_Sig_11_boosted_cp_uncoupled: %.10f\n", verbose_info, ii,    dS_Sig_11_boosted_cp_uncoupled);
                printf("Turn: %d macropart %d dS_Sig_33_boosted_cp_uncoupled: %.10f\n", verbose_info, ii,    dS_Sig_33_boosted_cp_uncoupled);
            }
    	    // Apply the kicks (Hirata's synchro-beam)
    	    delta_boosted +=      Fz_boosted + 0.5*(Fx_boosted*(px_boosted + 0.5*Fx_boosted) + Fy_boosted*(py_boosted + 0.5*Fy_boosted)); 
    	    px_boosted    +=      Fx_boosted;
    	    py_boosted    +=      Fy_boosted;
    
            // emit beamstrahlung photons from single macropart
         //   if (do_beamstrahlung){

	//	synrad(e_macropart, gamma, Fx_boosted, Fy_boosted, dz, &e_photon_array);


          //  }
 
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
