// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2025.                   //
// ########################################### //

#ifndef XFIELDS_CONSTANTS_H
#define XFIELDS_CONSTANTS_H

#include "xtrack/headers/constants.h"

#if !defined( MPROTON_GEV )
    #define MPROTON_GEV (0.93827208816)
#endif

#if !defined( MELECTRON_GEV )
    #define MELECTRON_GEV (0.00051099895000)
#endif

#if !defined( MELECTRON_KG )
    #define MELECTRON_KG (9.1093837015e-31)
#endif

#if !defined( HBAR_GEVS )
    #define HBAR_GEVS (6.582119569e-25)
#endif

#if !defined( REDUCED_COMPTON_WAVELENGTH_ELECTRON )
    #define REDUCED_COMPTON_WAVELENGTH_ELECTRON (3.8615926796089057e-13)
#endif

#endif /* XFIELDS_CONSTANTS_H */
