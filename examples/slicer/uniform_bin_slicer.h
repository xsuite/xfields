// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_UNIFORM_BIN_SLICER_H
#define XFIELDS_UNIFORM_BIN_SLICER_H

/*gpufun*/
void UniformBinSlicer_slice(
    UniformBinSlicerData slicer, LocalParticle* part0,
    /*gpuglmem*/ int64_t* i_slice_part){

    double const dzeta = UniformBinSlicerData_get_dzeta(slicer);
    double const z_min = UniformBinSlicerData_get_z_min(slicer);
    int64_t const num_slices = UniformBinSlicerData_get_num_slices(slicer);

    //start_per_particle_block (part0->part)
        double zeta = LocalParticle_get_zeta(part);
        int64_t i_slice = floor((zeta - z_min) / dzeta);

        if (i_slice >= 0 && i_slice < num_slices){
            i_slice_part[0] = i_slice;
        } else {
            i_slice_part[0] = -1;
        }
    //end_per_particle_block

    }
#endif