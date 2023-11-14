#ifndef XFIELDS_COPRESSED_PROFILE_H
#define XFIELDS_COPRESSED_PROFILE_H

void CompressedProfile_track_local_particle(CompressedProfileData el,
                                           LocalParticle *part0){

                                           }

void CompressedProfile_interp_result(
    CompressedProfileData el, LocalParticle *part0,
    /*gpuglmem*/ double* data,
    /*gpuglmem*/ int64_t* i_bunch_particles,
    /*gpuglmem*/ int64_t* i_slice_particles){

    //start_per_particle_block (part0->part)


    //end_per_particle_block

    }




#endif // XFIELDS_COPRESSED_PROFILE_H