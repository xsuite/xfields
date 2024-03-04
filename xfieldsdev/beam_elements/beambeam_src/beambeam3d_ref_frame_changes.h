// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_REF_FRAME_CHANGES_H
#define XFIELDS_BEAMBEAM3D_REF_FRAME_CHANGES_H


/*gpufun*/
void boost_coordinates(
        double const sphi,
        double const cphi,
        double const tphi,
        double const salpha,
        double const calpha,
        double* x_star,
        double* px_star,
        double* y_star,
        double* py_star,
        double* sigma_star,
        double* delta_star){


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
void boost_coordinates_inv(
        double const sphi,
        double const cphi,
        double const tphi,
        double const salpha,
        double const calpha,
        double* x,
        double* px,
        double* y,
        double* py,
        double* sigma,
        double* delta){

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

/*gpufun*/
void change_ref_frame_coordinates(
        double* x, double* px, double* y, double* py, double* zeta, double* pzeta,
        double const shift_x, double const shift_px,
        double const shift_y, double const shift_py,
        double const shift_zeta, double const shift_pzeta,
        double const sin_phi, double const cos_phi, double const tan_phi,
        double const sin_alpha, double const cos_alpha){

    // Change reference frame
    double x_star =     *x     - shift_x;
    double px_star =    *px    - shift_px;
    double y_star =     *y     - shift_y;
    double py_star =    *py    - shift_py;
    double zeta_star =  *zeta  - shift_zeta;
    double pzeta_star = *pzeta - shift_pzeta;

    // Boost coordinates of the weak beam
    boost_coordinates(
        sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha,
        &x_star, &px_star, &y_star, &py_star,
        &zeta_star, &pzeta_star);

    *x = x_star;
    *px = px_star;
    *y = y_star;
    *py = py_star;
    *zeta = zeta_star;
    *pzeta = pzeta_star;
    }

/*gpufun*/
void change_back_ref_frame_and_subtract_dipolar_coordinates(
        double* x, double* px,
        double* y, double* py,
        double* zeta, double* pzeta,
        double const shift_x, double const shift_px,
        double const shift_y, double const shift_py,
        double const shift_zeta, double const shift_pzeta,
        double const post_subtract_x, double const post_subtract_px,
        double const post_subtract_y, double const post_subtract_py,
        double const post_subtract_zeta, double const post_subtract_pzeta,
        double const sin_phi, double const cos_phi, double const tan_phi,
        double const sin_alpha, double const cos_alpha){

    // Inverse boost on the coordinates of the weak beam
    boost_coordinates_inv(
        sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha,
        x, px, y, py, zeta, pzeta);

    // Go back to original reference frame and remove dipolar effect
    *x =     *x     + shift_x     - post_subtract_x;
    *px =    *px    + shift_px    - post_subtract_px;
    *y =     *y     + shift_y     - post_subtract_y;
    *py =    *py    + shift_py    - post_subtract_py;
    *zeta =  *zeta  + shift_zeta  - post_subtract_zeta;
    *pzeta = *pzeta + shift_pzeta - post_subtract_pzeta;

    }

#endif
