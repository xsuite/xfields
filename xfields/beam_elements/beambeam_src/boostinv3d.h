#ifndef XFIELDS_BOOSTINV3D_H
#define XFIELDS_BOOSTINV3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

/*gpufun*/
void BoostParameters_boost_coordinates_inv(
    	const BoostParameters bp,
    	double* x,
    	double* px,
    	double* y,
    	double* py,
    	double* z,
    	double* delta){

    double const sphi = BoostParameters_get_sphi(bp);
    double const cphi = BoostParameters_get_cphi(bp);
    double const tphi = BoostParameters_get_tphi(bp);
    double const salpha = BoostParameters_get_salpha(bp);
    double const calpha = BoostParameters_get_calpha(bp);

    double const x_st     = *x;
    double const px_st    = *px;
    double const y_st     = *y;
    double const py_st    = *py ;
    double const z_st     = *z;
    double const delta_st = *delta ;

    double const pz_st = sqrt((1. + delta_st)*(1. + delta_st) - px_st*px_st - py_st*py_st);
    double const hx_st = px_st/pz_st;
    double const hy_st = py_st/pz_st;
    double const hz_st = 1. - (delta_st + 1.)/pz_st;

    double const Det_L = 1./cphi + (hx_st*calpha + hy_st*salpha - hz_st*sphi)*tphi;

    double const Linv_11 = (1./cphi + salpha*tphi*(hy_st - hz_st*salpha*sphi))/Det_L;
    double const Linv_12 = (salpha*tphi*(hz_st*calpha*sphi - hx_st))/Det_L;
    double const Linv_13 = -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    double const Linv_21 = (calpha*tphi*(-hy_st + hz_st*salpha*sphi))/Det_L;
    double const Linv_22 = (1./cphi + calpha*tphi*(hx_st-hz_st*calpha*sphi))/Det_L;
    double const Linv_23 = -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    double const Linv_31 = -hz_st*calpha*sphi/Det_L;
    double const Linv_32 = -hz_st*salpha*sphi/Det_L;
    double const Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    double const x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*z_st;
    double const y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*z_st;
    double const z_i = Linv_31*x_st + Linv_32*y_st + Linv_33*z_st;

    double const h = (delta_st + 1. - pz_st)*cphi*cphi;

    double const px_i = px_st*cphi + h*calpha*tphi;
    double const py_i = py_st*cphi + h*salpha*tphi;
    double const delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;
    
    // write inverse boosted coordinates to memory
    *x     =     x_i;
    *px    =    px_i;
    *y     =     y_i;
    *py    =    py_i;
    *z     =     z_i;
    *delta = delta_i;

}

/*gpufun*/
void BoostInv3D_track_local_particle(BoostInv3DData el, LocalParticle* part){

    // part= single macropart
    // el= beambeam element (other beam)	
    // Get data from memory
    const BoostParameters bpar      = BoostInv3DData_getp_boost_parameters(el);
    const double delta_x            = BoostInv3DData_get_delta_x(el);
    const double delta_y            = BoostInv3DData_get_delta_y(el);
    const double delta_px            = BoostInv3DData_get_delta_px(el);
    const double delta_py            = BoostInv3DData_get_delta_py(el);
    const double x_CO               = BoostInv3DData_get_x_CO(el);     
    const double px_CO              = BoostInv3DData_get_px_CO(el);
    const double y_CO               = BoostInv3DData_get_y_CO(el);
    const double py_CO              = BoostInv3DData_get_py_CO(el);
    const double z_CO               = BoostInv3DData_get_z_CO(el);
    const double delta_CO           = BoostInv3DData_get_delta_CO(el);
    const double Dx_sub             = BoostInv3DData_get_Dx_sub(el); 
    const double Dpx_sub            = BoostInv3DData_get_Dpx_sub(el);
    const double Dy_sub             = BoostInv3DData_get_Dy_sub(el);
    const double Dpy_sub            = BoostInv3DData_get_Dpy_sub(el);
    const double Dz_sub             = BoostInv3DData_get_Dz_sub(el);
    const double Ddelta_sub         = BoostInv3DData_get_Ddelta_sub(el);
    const unsigned int change_to_CO = BoostInv3DData_get_change_to_CO(el);

    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
        
        // point to macropart ii
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp
      
        // get macropart properties, one number each
    	double x_star     = LocalParticle_get_x(part);
    	double px_star    = LocalParticle_get_px(part);
    	double y_star     = LocalParticle_get_y(part);
    	double py_star    = LocalParticle_get_py(part);
    	double z_star     = LocalParticle_get_zeta(part);
    	double delta_star = LocalParticle_get_delta(part);

    	// Inverse boost coordinates
	BoostParameters_boost_coordinates_inv(bpar, &x_star, &px_star, &y_star, &py_star, &z_star, &delta_star);

        // Go back to original reference frame and remove dipolar effect
        double x, px, y, py, z, delta;
        if(change_to_CO == 1){
    
    	    x     = x_star     + x_CO     + delta_x  - Dx_sub;
            px    = px_star    + px_CO    + delta_px - Dpx_sub;
    	    y     = y_star     + y_CO     + delta_y  - Dy_sub;
      	    py    = py_star    + py_CO    + delta_py - Dpy_sub;
    	    z     = z_star     + z_CO                - Dz_sub;
    	    delta = delta_star + delta_CO            - Ddelta_sub;
        }else{

            x     = x_star;
            px    = px_star;
            y     = y_star;
            py    = py_star;
            z     = z_star;
            delta = delta_star;
        }
   	
        LocalParticle_set_x(part, x);
    	LocalParticle_set_px(part, px);
    	LocalParticle_set_y(part, y);
    	LocalParticle_set_py(part, py);
    	LocalParticle_set_zeta(part, z);
    	LocalParticle_update_delta(part, delta);
	

    } //only_for_context cpu_serial cpu_openmp
}


#endif
