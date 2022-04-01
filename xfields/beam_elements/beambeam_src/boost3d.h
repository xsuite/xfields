#ifndef XFIELDS_BOOST3D_H
#define XFIELDS_BOOST3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

#include <stdio.h>

/*gpufun*/
void BoostParameters_boost_coordinates(const BoostParameters bp, 
    double* x_star,
    double* px_star,
    double* y_star,
    double* py_star,
    double* z_star,
    double* delta_star){
  
    double x     = *x_star;
    double px    = *px_star;
    double y     = *y_star;
    double py    = *py_star;
    double z     = *z_star;
    double delta = *delta_star;

    // boost
    double const sphi = BoostParameters_get_sphi(bp);
    double const cphi = BoostParameters_get_cphi(bp);
    double const tphi = BoostParameters_get_tphi(bp);
    double const salpha = BoostParameters_get_salpha(bp);
    double const calpha = BoostParameters_get_calpha(bp);

    double const h = delta + 1. - sqrt((1. + delta)*(1. + delta) - px*px - py*py);

    double const px_st    = px/cphi - h*calpha*tphi/cphi;
    double const py_st    = py/cphi - h*salpha*tphi/cphi;
    double const delta_st = delta   - px*calpha*tphi - py*salpha*tphi + h*tphi*tphi;

    double const pz_st = sqrt((1. + delta_st)*(1. + delta_st) - px_st*px_st - py_st*py_st);

    double const hx_st = px_st/pz_st;
    double const hy_st = py_st/pz_st;
    double const hz_st = 1. - (delta_st + 1)/pz_st;

    double const L11 = 1. + hx_st*calpha*sphi;
    double const L12 = hx_st*salpha*sphi;
    double const L13 = calpha*tphi;

    double const L21 = hy_st*calpha*sphi;
    double const L22 = 1. + hy_st*salpha*sphi;
    double const L23 = salpha*tphi;

    double const L31 = hz_st*calpha*sphi;
    double const L32 = hz_st*salpha*sphi;
    double const L33 = 1./cphi;
    
    double const x_st = L11*x + L12*y + L13*z;
    double const y_st = L21*x + L22*y + L23*z;
    double const z_st = L31*x + L32*y + L33*z;

    // write boosted coordinates to memory
    *x_star     =     x_st;
    *px_star    =    px_st;
    *y_star     =     y_st;
    *py_star    =    py_st;
    *z_star     =     z_st;
    *delta_star = delta_st;

}

/*gpufun*/
void Boost3D_track_local_particle(Boost3DData el, LocalParticle* part0){

    // Get data from memory
    const BoostParameters bpar = Boost3DData_getp_boost_parameters(el);
    const double x2_CO         = Boost3DData_get_x2_CO(el);
    const double y2_CO         = Boost3DData_get_y2_CO(el);
    const double px2_CO        = Boost3DData_get_px2_CO(el);
    const double py2_CO        = Boost3DData_get_py2_CO(el);
    const double delta_x       = Boost3DData_get_delta_x(el);
    const double delta_y       = Boost3DData_get_delta_y(el);
    const int64_t swap_x       = Boost3DData_get_swap_x(el);

    //start_per_particle_block (part0->part)
    	double x = LocalParticle_get_x(part);
    	double px = LocalParticle_get_px(part);
    	double y = LocalParticle_get_y(part);
    	double py = LocalParticle_get_py(part);
    	double zeta = LocalParticle_get_zeta(part);
    	double delta = LocalParticle_get_delta(part);
 

        // swap x and px
        if(swap_x == 1){
            x *= -1.0;
            px *= -1.0;
        }

        double x_star     = x  - x2_CO   - delta_x;
        double px_star    = px - px2_CO;
        double y_star     = y  - y2_CO   - delta_y;
        double py_star    = py - py2_CO;
        double z_star     = zeta;
        double delta_star = delta;
  
        // Boost coordinates
	BoostParameters_boost_coordinates(bpar, &x_star, &px_star, &y_star, &py_star, &z_star, &delta_star);
    	
        LocalParticle_set_x(part, x_star);
    	LocalParticle_set_px(part, px_star);
    	LocalParticle_set_y(part, y_star);
    	LocalParticle_set_py(part, py_star);
    	LocalParticle_set_zeta(part, z_star);
    	LocalParticle_update_delta(part, delta_star);

    //end_per_particle_block
}

#endif
