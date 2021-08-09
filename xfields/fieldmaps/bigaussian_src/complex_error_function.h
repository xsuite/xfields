#ifndef XFIELDS_COMPLEX_ERROR_FUNCTION_H
#define XFIELDS_COMPLEX_ERROR_FUNCTION_H

#include <stdbool.h> //only_for_context cpu_serial cpu_openmp
#include <math.h> //only_for_context cpu_serial cpu_openmp

#include "constants.h"

/*gpufun*/
void cerrf( double x, double y,
    double* /*restrict*/ out_x,
    double* /*restrict*/ out_y )
{
    typedef double real_type;

    /**
     * Based upon the algorithm developed by W. Gautschi 1970,
     * "Efficient Computation of the Complex Error Function",
     * SIAM Journal on Numerical Analysis, Vol. 7, Issue 1. 1970, pages 187-198
     * https://epubs.siam.org/doi/10.1137/0707012
     *
     * It corresponds closely to the previously used "CERNLib C" version,
     * translated from the FORTRAN function written at CERN by K. Koelbig,
     * Program C335, 1970. The main difference to Gautschi's formulation is
     * a split in the main loop and the introduction of arrays to store the
     * intermediate results as a consequence of this. The version implemented
     * here should perform roughly equally well or even slightly better on
     * modern out-of-order super-scalar CPUs but has drastically improved
     * performance on GPUs and GPU-like systems.
     *
     * See also M. Bassetti and G.A. Erskine,
     * "Closed expression for the electric field of a two-dimensional Gaussian
     *  charge density", CERN-ISR-TH/80-06; */

    real_type const sign_x = ( real_type )( x >= ( real_type )0.0 ) -
                             ( real_type )( x <  ( real_type )0.0 );

    real_type const sign_y = ( real_type )( y >= ( real_type )0.0 ) -
                             ( real_type )( y <  ( real_type )0.0 );

    real_type inv_h2   = ( real_type )1.0;
    real_type temp     = y * sign_y;
    real_type y_plus_h = temp;

    real_type Rx, Ry, Sx, Sy, Wx, Wy, h2_n, nn;

    int nu = ( int )FADDEEVA_GAUSS_HERMITE_NU;
    int N  = 0;
    int n  = 0;

    bool z_is_in_r0, use_taylor_sum;
    Ry = Sx = Sy = h2_n = ( real_type )0.0;

    y  = temp;
	x *= sign_x;

    /* R_0 ... rectangle with width FADDEEVA_X_LIMIT and
     *         height FADDEEVA_Y_LIMIT. Inside R_0, w(z) is calculated using
     *         a truncated Taylor expansion. Outside, a Gauss--Hermite
     *         quadrature in the guise of a continuos fraction is used */

    z_is_in_r0 = ( ( y < ( real_type )FADDEEVA_Y_LIMIT ) &&
                   ( x < ( real_type )FADDEEVA_X_LIMIT ) );

	if( z_is_in_r0 )
    {
        temp = x / ( real_type )FADDEEVA_X_LIMIT;
        temp = ( ( real_type )1.0 - y / ( real_type )FADDEEVA_Y_LIMIT ) *
               sqrt( ( real_type )1.0 - temp * temp );


        nu   = ( y > ( real_type )REAL_EPS )
             ? ( int )FADDEEVA_NU_0 + ( int )( ( real_type )FADDEEVA_NU_1 * temp )
             : ( int )0;

        N    = ( int )FADDEEVA_N0 + ( int )( ( real_type )FADDEEVA_N1 * temp );
        h2_n = ( real_type )2.0 * ( real_type )FADDEEVA_H0 * temp;
        inv_h2 = ( real_type )1.0 / h2_n;
        y_plus_h += ( real_type )0.5 * h2_n;
        h2_n = power_n( h2_n, N - 1 );
    }

    Rx = ( y > ( real_type )REAL_EPS )
       ? ( real_type )0.0
       : exp( -x * x ) / ( real_type )TWO_OVER_SQRT_PI;

    use_taylor_sum = ( z_is_in_r0 ) && ( h2_n > ( real_type )REAL_EPS );

    n = nu;
    nn = ( real_type )n;

    /* z outside of R_0: continuos fraction / Gauss - Hermite quadrature
     * z inside  of R_0: first iterations of recursion until n == N */
    for( ; n > N ; --n, nn -= ( real_type )1.0 )
    {
        Wx     = y_plus_h + nn * Rx;
        Wy     = x - nn * Ry;
        temp   = ( Wx * Wx + Wy * Wy );
        Rx     = ( real_type )0.5 * Wx;
        Ry     = ( real_type )0.5 * Wy;
        temp   = ( real_type )1.0 / temp;
        Rx    *= temp;
        Ry    *= temp;
    }

    /* loop rejects everything if z is not in R_0 because then n == 0 already;
     * otherwise, N iterations until taylor expansion is summed up */
    for( ; n > 0 ; --n, nn -= ( real_type )1.0 )
    {
        Wx     = y_plus_h + nn * Rx;
        Wy     = x - nn * Ry;
        temp   = ( Wx * Wx + Wy * Wy );
        Rx     = ( real_type )0.5 * Wx;
        Ry     = ( real_type )0.5 * Wy;
        temp   = ( real_type )1.0 / temp;
        Rx    *= temp;
        Ry    *= temp;

        Wx     = h2_n + Sx;
        h2_n  *= inv_h2;
        Sx     = Rx * Wx - Ry * Sy;
        Sy     = Ry * Wx + Rx * Sy;
    }

    /* Wx, Wy ... result for z|Q1 = |x| + i |y| ... in first quadrant! */
    Wx = ( real_type )TWO_OVER_SQRT_PI * ( ( use_taylor_sum ) ? Sx : Rx );
    Wy = ( real_type )TWO_OVER_SQRT_PI * ( ( use_taylor_sum ) ? Sy : Ry );

    if( sign_y < ( real_type )0.0 )  /* Quadrants 3 & 4 */
    {
        real_type const arg = ( real_type )2.0 * x * y;

        #if defined( __OPENCL_VERSION__ ) || \
                   ( ( defined( XFIELDS_USE_SINCOS ) ) && \
                     ( XFIELDS_USE_SINCOS == 1 ) )
        real_type exp_cos;
        real_type exp_sin = sincos( arg, &exp_cos );
        #else
        real_type exp_cos = cos( arg );
        real_type exp_sin = sin( arg );
        #endif

        temp = ( y - x ) * ( x + y );
        temp = exp( temp );
        exp_cos *= temp;
        exp_sin *= temp;

        Wx  = exp_cos - Wx;
        Wy += exp_sin;
    }

    *out_x = Wx;
    *out_y = sign_x * Wy;
}

#endif /* XFIELDS_COMPLEX_ERROR_FUNCTION_H */

