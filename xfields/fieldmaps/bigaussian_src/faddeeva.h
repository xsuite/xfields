// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_FADDEEVA_H
#define XFIELDS_FADDEEVA_H

#ifdef XO_CONTEXT_CPU_SERIAL
    #include "xfields/fieldmaps/bigaussian_src/faddeeva_mit.h"
#else /* XO_CONTEXT_{CPU_OPENMP, CUDA, CL} */
    #include "xfields/fieldmaps/bigaussian_src/faddeeva_cernlib.h"
#endif

#endif /* XFIELDS_FADDEEVA_H */
