// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_FADDEEVA_CERNLIB_H
#define XFIELDS_FADDEEVA_CERNLIB_H

/** \file complex_error_function.h
  * \note always include headers/constants.h, headers/power_n.h, and
  *       headers/sincos.h first! */

#include <stdbool.h>  //only_for_context cpu_serial cpu_openmp
#include <math.h>     //only_for_context cpu_serial cpu_openmp

/* These parameters correspond to the original algorithm developed by Gautschi
 * with a target accuracy of < 0.5 x 10^{-10} in the *absolute* error. Upstream
 * cernlib had switched to more aggressive parameters targeting approx. a
 * *relative* error > 0.5 x 10^{-14} instead -> see the values in comments
 * after the defines. In order to properly use them, you also have to define
 *
 * FADDEEVA_NO_GZ_WEIGHT_FN
 *
 * as compile parameter / compiler flag, this sets g(z) = 1 for all z. */

#if !defined( FADDEEVA_X_LIMIT )
    #define FADDEEVA_X_LIMIT 5.33           /* CERNLib value: 8.3 */
#endif /* !defined( FADDEEVA_X_LIMIT ) */

#if !defined( FADDEEVA_Y_LIMIT )
    #define FADDEEVA_Y_LIMIT 4.29           /* CERNLib value: 7.4 */
#endif /* !defined( FADDEEVA_Y_LIMIT ) */

#if !defined( FADDEEVA_H0 )
    #define FADDEEVA_H0 1.6                 /* CERNLib value: 1.6, unchanged */
#endif /* !defined( FADDEEVA_H0 ) */

#if !defined( FADDEEVA_NU_0 )
    #define FADDEEVA_NU_0 10                /* CERNLib value: 10, unchanged */
#endif /* !defined( FADDEEVA_NU_0 ) */

#if !defined( FADDEEVA_NU_1 )
    #define FADDEEVA_NU_1 21                /* CERNLib value: 26 */
#endif /* !defined( FADDEEVA_NU_1 ) */

#if !defined( FADDEEVA_N0 )
    #define FADDEEVA_N0 7                   /* CERNLib value: 7, unchanged */
#endif /* !defined( FADDEEVA_N0 ) */

#if !defined( FADDEEVA_N1 )
    #define FADDEEVA_N1 23                  /* CERNLib value: 16 */
#endif /* !defined( FADDEEVA_N1 ) */

#if !defined( FADDEEVA_CONT_FRAC_K )
    #define FADDEEVA_CONT_FRAC_K 9          /* CERNLib value: 9, unchanged */
#endif /* !defined( FADDEEVA_CONT_FRAC_K ) */

/* ************************************************************************* */

/** \fn void faddeeva_w_q1( double const, double const, double*, double* )
 *  \brief calculates the Faddeeva function w(z) for z = x + i * y in Q1
 *
 *  \param[in] x real component of argument z
 *  \param[in] y imaginary component of argument z
 *  \param[out] out_x pointer to real component of result
 *  \param[out] out_y pointer to imanginary component of result
 *
 *  \warning This function assumes that x and y are > 0 i.e., that z is
 *           from the first quadrant Q1 of the complex plane. Use faddeeva_w if
 *           you need a more general function
 *
 *  \note    Based upon the algorithm developed by W. Gautschi 1970,
 *           "Efficient Computation of the Complex Error Function",
 *           SIAM Journal on Numerical Analysis, Vol. 7, Issue 1. 1970,
 *           pages 187-198, https://epubs.siam.org/doi/10.1137/0707012
 */

/*gpufun*/ void faddeeva_w_q1(
    double const x, double const y,
    double* /*restrict*/ out_x,
    double* /*restrict*/ out_y )
{
    /* This implementation corresponds closely to the previously used
     * "CERNLib C" version, translated from the FORTRAN function written at
     * CERN by K. Koelbig, Program C335, 1970. The main difference to
     * Gautschi's formulation is a split in the main loop and the introduction
     * of arrays to store the intermediate results as a consequence of this.
     * The version implemented here should perform roughly equally well or even
     * slightly better on modern out-of-order super-scalar CPUs but has
     * drastically improved performance on GPUs and GPU-like systems.
     *
     * See also M. Bassetti and G.A. Erskine,
     * "Closed expression for the electric field of a two-dimensional Gaussian
     *  charge density", CERN-ISR-TH/80-06; */

    double inv_h2   = ( double )1.0;
    double y_plus_h = y;
    double temp, Rx, Ry, Sx, Sy, Wx, Wy, h2_n, nn;

    int nu = ( int )FADDEEVA_CONT_FRAC_K;
    int N  = 0;
    int n  = 0;

    bool use_taylor_sum;
    Ry = Sx = Sy = h2_n = ( double )0.0;

    /* R_0 ... rectangle with width FADDEEVA_X_LIMIT and
     *         height FADDEEVA_Y_LIMIT. Inside R_0, w(z) is calculated using
     *         a truncated Taylor expansion. Outside, a Gauss--Hermite
     *         quadrature in the guise of a continuos fraction is used */

    use_taylor_sum = ( ( y < ( double )FADDEEVA_Y_LIMIT ) &&
                       ( x < ( double )FADDEEVA_X_LIMIT ) );

	if( use_taylor_sum )
    {
        #if !defined( FADDEEVA_NO_GZ_WEIGHT_FN )
        /* calculate g(z) = sqrt( 1 - (x/x0)^2 ) * ( 1 - y/y0 ) */
        temp  = x * ( ( double )1. / ( double )FADDEEVA_X_LIMIT );
        temp  = ( ( double )1.0 +  temp ) * ( ( double )1.0 - temp );
        temp  = sqrt( temp );
        temp *= ( double )1. - y * ( ( double )1. / ( double )FADDEEVA_Y_LIMIT );
        /*now: temp = g(z) */
        #else /* !defined( FADDEEVA_NO_GZ_WEIGHT_FN ) */
        temp = ( double )1.;
        #endif /* defined( FADDEEVA_NO_GZ_WEIGHT_FN ) */

        nu   = ( int )FADDEEVA_NU_0 + ( int )( ( double )FADDEEVA_NU_1 * temp );

        N         = ( int )FADDEEVA_N0 + ( int )( ( double )FADDEEVA_N1 * temp );
        h2_n      = ( double )FADDEEVA_H0 * temp; /* h(z) = h_0 * g(z) */
        y_plus_h += h2_n; /* y_plus_h = y + h(z) */
        h2_n     *= ( double )2.; /* now: h2_n = 2 * h(z) */
        inv_h2    = ( double )1. / h2_n;
        h2_n      = power_n( h2_n, N - 1 ); /* finally: h2_n = (2*h(z))^(N-1) */
    }

    /* If h(z) is so close to 0 that it is practically 0, there is no
     * point in doing the extra work for the Taylor series -> in that
     * very unlikely case, use the continuos fraction & verify result! */
    use_taylor_sum &= ( h2_n > ( double )REAL_EPSILON );

    Rx = 0;
    #ifdef FADDEEVA_SPECIAL_Y_0
    Rx = ( y > ( double )REAL_EPSILON )
       ? ( double )0.0 : exp( -x * x ) / ( double )TWO_OVER_SQRT_PI;
    #endif

    n = nu;
    nn = ( double )n;

    /* z outside of R_0: continuos fraction / Gauss - Hermite quadrature
     * z inside  of R_0: first iterations of recursion until n == N */
    for( ; n > N ; --n, nn -= ( double )1.0 )
    {
        Wx     = y_plus_h + nn * Rx;
        Wy     = x - nn * Ry;
        temp   = ( Wx * Wx ) + ( Wy * Wy );
        Rx     = ( double )0.5 * Wx;
        Ry     = ( double )0.5 * Wy;
        temp   = ( double )1.0 / temp;
        Rx    *= temp;
        Ry    *= temp;
    }

    /* loop rejects everything if z is not in R_0 because then n == 0 already;
     * otherwise, N iterations until taylor expansion is summed up */
    for( ; n > 0 ; --n, nn -= ( double )1.0 )
    {
        Wx     = y_plus_h + nn * Rx;
        Wy     = x - nn * Ry;
        temp   = ( Wx * Wx ) + ( Wy * Wy );
        Rx     = ( double )0.5 * Wx;
        Ry     = ( double )0.5 * Wy;
        temp   = ( double )1.0 / temp;
        Rx    *= temp;
        Ry    *= temp;

        Wx     = h2_n + Sx;
        h2_n  *= inv_h2;
        Sx     = Rx * Wx - Ry * Sy;
        Sy     = Ry * Wx + Rx * Sy;
    }

    if( use_taylor_sum )
    {
        Wx = ( double )TWO_OVER_SQRT_PI * Sx;
        Wy = ( double )TWO_OVER_SQRT_PI * Sy;
    }
    else
    {
        Wx = ( double )TWO_OVER_SQRT_PI * Rx;
        Wy = ( double )TWO_OVER_SQRT_PI * Ry;
    }

    *out_x = Wx;
    *out_y = Wy;
}

/** \fn void faddeeva_w( double const x, double const y, double* out_x, double* out_y )
 *  \brief calculates the Faddeeva function w(z) for general z = x + i * y
 *
 *   Calls faddeeva_w_q1 internally for |x| and |y| on quadrant Q1 and
 *   transforms the result to Q2, Q3, and Q4 before returning them via
 *   out_x and out_y.
 *
 *  \param[in] x real component of argument z
 *  \param[in] y imaginary component of argument z
 *  \param[out] out_x pointer to real component of result
 *  \param[out] out_y pointer to imanginary component of result
 *
 */

/*gpufun*/ void faddeeva_w( double x, double y,
    double* /*restrict*/ out_x, double* /*restrict*/ out_y )
{
    double const sign_x = ( double )( ( x >= ( double )0. ) - ( x < ( double )0. ) );
    double const sign_y = ( double )( ( y >= ( double )0. ) - ( y < ( double )0. ) );
    double Wx, Wy;

    x *= sign_x;
    y *= sign_y;

    faddeeva_w_q1( x, y, &Wx, &Wy );

    if( sign_y < ( double )0.0 )  /* Quadrants Q3 and Q4 */
    {
        double const exp_arg  = ( y - x ) * ( y + x );
        double const trig_arg = ( double )2. * x * y;
        double const exp_factor = ( double )2. * exp( exp_arg );
        double sin_arg, cos_arg;

        xsuite_sincos( trig_arg, &sin_arg, &cos_arg );
        Wx = exp_factor * cos_arg - Wx;
        Wy = exp_factor * sin_arg + Wy;
    }

    *out_x = Wx;
    *out_y = sign_x * Wy; /* Takes care of Quadrants Q2 and Q3 */
}

#endif /* XFIELDS_FADDEEVA_CERNLIB_H */

