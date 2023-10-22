// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_UNIFORM_BIN_SLICER_H
#define XFIELDS_UNIFORM_BIN_SLICER_H


/*gpufun*/
void UniformBinSlicer_slice(UniformBinSlicerData el,
                LocalParticle* part0,
                /*gpuglmem*/ double* b){

    double const a = UniformBinSlicerData_get_a(el);

    //start_per_particle_block (part0->part)

        const int64_t ipart = part->ipart;
        double const val = b[ipart];

        LocalParticle_add_to_s(part, val + a);

    //end_per_particle_block
}

/*gpufun*/
void UniformBinSlicer_track_local_particle(UniformBinSlicerData el,
                LocalParticle* part0){

    double const a = UniformBinSlicerData_get_a(el);

    //start_per_particle_block (part0->part)

        LocalParticle_set_s(part, a);

    //end_per_particle_block
}

#endif

// /*gpufun*/
// void UniformBinSlicer_slice(
//     UniformBinSlicerData slicer, LocalParticle* part0,
//     /*gpuglmem*/ int64_t* i_slice_for_particles){

//     double const dzeta = UniformBinSlicerData_get_dzeta(slicer);
//     double const z_min = UniformBinSlicerData_get_z_min(slicer);
//     int64_t const num_slices = UniformBinSlicerData_get_num_slices(slicer);

//     /start_per_particle_block (part0->part)
//         double zeta = LocalParticle_get_zeta(part);
//         int64_t part_id = LocalParticle_get_particle_id(part);

//         int64_t i_slice = floor((zeta - z_min) / dzeta);

//         if (i_slice >= 0 && i_slice < num_slices){
//             i_slice_for_particles[part_id] = i_slice;
//         } else {
//             i_slice_for_particles[part_id] = -1;
//         }
//     /end_per_particle_block

//     }

// void UniformBinSlicer_track_local_particle(
//     UniformBinSlicerData slicer, LocalParticle* part0){

//     /start_per_particle_block (part0->part)
//         double zeta = LocalParticle_get_zeta(part);
//     /end_per_particle_block

//     }
