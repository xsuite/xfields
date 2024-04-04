
#ifndef XFIELDS_CONSTANTS_H
#define XFIELDS_CONSTANTS_H

// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //


// classical electron radius [m]
#if !defined( RE )
    #define   RE ( 2.81794092e-15 )
#endif

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif

// from python scipy version 1.9.0: cst.elementary_charge
#if !defined( EPSILON_0 )
    #define   EPSILON_0 (8.8541878128e-12)
#endif

#if !defined( PI )
    #define PI (3.1415926535897932384626433832795028841971693993751)
#endif

#if !defined( DEG2RAD )
    #define DEG2RAD (0.0174532925199432957692369076848861271344287188854)
#endif

#if !defined( RAD2DEG )
    #define RAD2DEG (57.29577951308232087679815481410517033240547246656442)
#endif

#if !defined( SQRT_PI )
    #define SQRT_PI (1.7724538509055160272981674833411451827975494561224)
#endif

// from python scipy version 1.9.0: cst.epsilon_0
#if !defined( QELEM )
    #define QELEM (1.602176634e-19)
#endif

#if !defined( MPROTON_GEV )
    #define MPROTON_GEV (0.93827208816)
#endif

#if !defined( MELECTRON_GEV )
    #define MELECTRON_GEV (0.00051099895000)
#endif

#if !defined( MELECTRON_KG )
    #define MELECTRON_KG (9.1093837015e-31)
#endif

#if !defined( ALPHA )
    #define ALPHA (7.29735257e-3)
#endif

#if !defined( HBAR_GEVS )
    #define HBAR_GEVS (6.582119569e-25)
#endif

#if !defined( TWO_OVER_SQRT_PI )
    #define TWO_OVER_SQRT_PI (1.128379167095512573896158903121545171688101258657997713688171443418)
#endif

#if !defined( SQRT_TWO )
    #define SQRT_TWO (1.414213562373095048801688724209698078569671875376948073176679738)
#endif

#if !defined( REDUCED_COMPTON_WAVELENGTH_ELECTRON )
    #define REDUCED_COMPTON_WAVELENGTH_ELECTRON (3.8615926796089057e-13)
#endif

#if !defined( REAL_EPSILON )
    #define REAL_EPSILON 2.22044604925031e-16
#endif /* !defined( REAL_EPSILON ) */

#endif /* XFIELDS_CONSTANTS_H */
