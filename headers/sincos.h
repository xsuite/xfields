// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_SINCOS_H__
#define XFIELDS_SINCOS_H__

#include <math.h> //only_for_context cpu_serial cpu_openmp

/* Define XSUITE_NO_SINCOS as a compiler parameter to never use the inline
 * wrapper function below (the compiler may still decide to use sincos - like
 * intrinsic, but usually enforcing that math functions have to update
 * errno should put an end to this.
 *
 * If XSUITE_NO_SINCOS is not set, then a best-effort attempt is made to
 * use it on platforms that support it (i.e. gnu compiler, no ANSI standards,
 * relaxed errno handling for math functions.
 */

#if ( !defined( XSUITE_NO_SINCOS ) ) && ( defined( __OPENCL_C_VERSION__ ) )
/*gpufun*/ void xsuite_sincos( double const arg,
    double* /*restrict*/ sin_result, double* /*restrict*/ cos_result ) {
    *sin_result = sincos( arg, cos_result ); }

#elif ( !defined( XSUITE_NO_SINCOS ) ) && \
      ( ( defined( __CUDA_ARCH__ ) ) || \
        ( defined( __GNUC__ ) && !defined( __clang__ ) && \
         !defined( __STRICT_ANSI__ ) && !defined( __INTEL_COMPILER ) && \
         defined( __NO_MATH_ERRNO__ ) ) )
/*gpufun*/ void xsuite_sincos( double const arg,
    double* /*restrict*/ sin_result, double* /*restrict*/ cos_result ) {
    sincos( arg, sin_result, cos_result ); }

#else
/*gpufun*/ void xsuite_sincos( double const arg,
    double* /*restrict*/ sin_result, double* /*restrict*/ cos_result ) {
    *sin_result = sin( arg );
    *cos_result = cos( arg ); }

#endif /* XSUITE_NO_SINCOS */
#endif /* XFIELDS_SINCOS_H__ */
