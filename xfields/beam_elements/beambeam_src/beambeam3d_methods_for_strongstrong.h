// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_METHODS_FOR_STRONGSTRONG_H
#define XFIELDS_BEAMBEAM3D_METHODS_FOR_STRONGSTRONG_H


/*gpufun*/
void boost_local_particle(BeamBeamBiGaussian3DData el,
                LocalParticle* part0){

    // Get data from memory
    double const sin_phi = BeamBeamBiGaussian3DData_get_sin_phi(el);
    double const cos_phi = BeamBeamBiGaussian3DData_get_cos_phi(el);
    double const tan_phi = BeamBeamBiGaussian3DData_get_tan_phi(el);
    double const sin_alpha = BeamBeamBiGaussian3DData_get_sin_alpha(el);
    double const cos_alpha = BeamBeamBiGaussian3DData_get_cos_alpha(el);

    //start_per_particle_block (part0->part)

        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double zeta = LocalParticle_get_zeta(part);
        double pzeta = LocalParticle_get_pzeta(part);

        // Change reference frame
        double x_star =     x;
        double px_star =    px;
        double y_star =     y;
        double py_star =    py;
        double sigma_star = zeta;
        double pzeta_star = pzeta; // TODO: could be fixed, in any case we assume beta=beta0=1
                                          //       in the synchrobeam

        // Boost coordinates of the weak beam
        boost_coordinates(
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha,
            &x_star, &px_star, &y_star, &py_star,
            &sigma_star, &pzeta_star);

        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);
        LocalParticle_set_zeta(part, zeta);
        LocalParticle_update_pzeta(part, pzeta);

    //end_per_particle_block

}

/*gpufun*/
void BeamBeam3D_selective_apply_synchrobeam_kick(BeamBeamBiGaussian3DData el,
                LocalParticle* part0,
                const int64_t i_step,
                /*gpuglmem*/ int64_t* i_slice_for_particles){


    //start_per_particle_block (part0->part)

        const int64_t N_slices = BeamBeamBiGaussian3DData_get_num_slices_other_beam(el);
        const int64_t i_slice = i_step - i_slice_for_particles[part->ipart];

        if (i_slice >= 0 && i_slice < N_slices){

            double x_star = LocalParticle_get_x(part);
            double px_star = LocalParticle_get_px(part);
            double y_star = LocalParticle_get_y(part);
            double py_star = LocalParticle_get_py(part);
            double zeta_star = LocalParticle_get_zeta(part);
            double pzeta_star = LocalParticle_get_pzeta(part);

            const double q0 = LocalParticle_get_q0(part);
            const double p0c = LocalParticle_get_p0c(part); // eV

            synchrobeam_kick(
                el, i_slice,
                q0, p0c,
                &x_star,
                &px_star,
                &y_star,
                &py_star,
                &zeta_star,
                &pzeta_star);

            LocalParticle_set_x(part, x_star);
            LocalParticle_set_px(part, px_star);
            LocalParticle_set_y(part, y_star);
            LocalParticle_set_py(part, py_star);
            LocalParticle_set_zeta(part, zeta_star);
            LocalParticle_update_pzeta(part, pzeta_star);

        }

    //end_per_particle_block

}

#endif