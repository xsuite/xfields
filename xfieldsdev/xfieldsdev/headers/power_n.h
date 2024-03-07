// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_HEADERS_POWER_H_H__
#define XFIELDS_HEADERS_POWER_H_H__

/** \file power_n.h
 *  \note always include constants first!!! */

/*gpufun*/ double power_n( double x, unsigned int n )
{
    #if defined( __OPENCL_VERSION__ )
    return pown( x, n );
    #else

    double x_n = x;

    unsigned int const n_div_16 = n >> 4u;
    unsigned int const n_mod_16 = n - ( n_div_16 << 4u );

    switch( n_mod_16 )
    {
        case  0u: { x_n = ( double )1.0; break; }
        case  1u: { break; }
        case  2u: { x_n *= x;                                       break; }
        case  3u: { x_n *= x * x;                                   break; }
        case  4u: { x_n *= x;     x_n *= x_n;                       break; }
        case  5u: { x_n *= x;     x_n *= x_n * x;                   break; }
        case  6u: { x_n *= x * x; x_n *= x_n;                       break; }
        case  7u: { x_n *= x * x; x_n *= x_n * x;                   break; }
        case  8u: { x_n *= x;     x_n *= x_n;     x_n *= x_n;       break; }
        case  9u: { x_n *= x * x; x_n *= x_n * x_n;                 break; }
        case 10u: { x_n *= x * x; x_n *= x_n * x_n * x;             break; }
        case 11u: { x_n *= x;     x_n *= x_n * x; x_n *= x_n * x;   break; }
        case 12u: { x_n *= x * x; x_n *= x_n;     x_n *= x_n;       break; }
        case 13u: { x_n *= x * x; x_n *= x_n;     x_n *= x_n * x;   break; }
        case 14u: { x_n *= x * x; x_n *= x_n * x; x_n *= x_n;       break; }
        case 15u: { x_n *= x;     x_n *= x_n * x; x_n *= x_n * x_n; break; }
        default:  { x_n = ( double )0.0; }
    };

    if( n_div_16 > 0u ){ x *= x; x *= x; x *= x; x *= x; }

    switch( n_div_16 )
    {
        case  0u: { x_n  = ( n_mod_16 != 0u ) ? x_n : ( double )1.0; break; }
        case  1u: { x_n *= x;                                           break; }
        case  2u: { x   *= x; x_n *= x;                                 break; }
        case  3u: { x_n *= x * x * x;                                   break; }
        case  4u: { x   *= x; x *= x; x_n *= x;                         break; }
        case  5u: { x_n *= x; x *= x; x *= x; x_n *= x;                 break; }
        case  6u: { x   *= x * x; x *= x; x_n *= x;                     break; }
        case  7u: { x_n *= x; x *= x * x; x *= x; x_n *= x;             break; }
        case  8u: { x *= x; x *= x; x*= x; x_n *= x;                    break; }
        case  9u: { x *= x * x; x *= x * x; x_n *= x;                   break; }
        case 10u: { x_n *= x; x *= x * x; x *= x * x; x_n *= x;         break; }
        case 11u: { x_n *= x * x; x *= x * x; x *= x * x; x_n *= x;     break; }
        case 12u: { x *= x; x *= x; x_n *= x; x *= x; x_n *= x;         break; }
        case 13u: { x_n *= x; x *= x; x *= x; x_n *= x; x *= x;
                    x_n *= x; break; }

        case 14u: { x_n *= x * x; x *= x; x *= x; x_n *= x; x *= x;
                    x_n *= x; break; }

        case 15u: { x *= x * x; x_n *= x * x; x *= x * x; x_n *= x;    break; }

        default:
        {
            unsigned int ii = 0u;
            unsigned int nn = n_div_16 % 16u;

            for( ; ii < nn ; ++ii ) x_n *= x;

            x *= x; x *= x; x *= x; x *= x;
            nn = ( n_div_16 - nn ) >> 4u;

            for( ii = 0u ; ii < nn ; ++ii ) x_n *= x;
        }
    };

    return x_n;
    #endif /* defined( __OPENCL_VERSION__ ) */
}

#endif /* XFIELDS_HEADERS_POWER_H_H__ */
