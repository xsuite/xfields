// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_ELECTRONCLOUD_H
#define XFIELDS_ELECTRONCLOUD_H

#include "xtrack/headers/track.h"
#include "xfields/headers/particle_states.h"
#include "xfields/fieldmaps/interpolated_src/cubic_interpolators.h"


GPUFUN
void ElectronCloud_track_local_particle(
		 ElectronCloudData el, LocalParticle* part0){

    //TODO: kick_z flag needs to be inserted in xobject

    const double length = ElectronCloudData_get_length(el);

    const double x_shift = ElectronCloudData_get_x_shift(el);
    const double y_shift = ElectronCloudData_get_y_shift(el);
    const double zeta_shift = ElectronCloudData_get_zeta_shift(el);
    TriCubicInterpolatedFieldMapData fmap = ElectronCloudData_getp_fieldmap(el);
    START_PER_PARTICLE_BLOCK(part0, part);
        const double x = LocalParticle_get_x(part);
        const double y = LocalParticle_get_y(part);
        const double zeta = LocalParticle_get_zeta(part);

        double const beta0 = LocalParticle_get_beta0(part);

        double dphi_dx=0;
        double dphi_dy=0;
        double dphi_dzeta=0;

        if( TriCubicInterpolatedFieldMap_interpolate_grad(fmap,
            x - x_shift, y - y_shift, zeta - zeta_shift,
            &dphi_dx, &dphi_dy, &dphi_dzeta)
          ){
              LocalParticle_set_state(part, XF_OUTSIDE_INTERPOL); // Stop tracking particle if it escapes the interpolation grid.
          }

        const double px_kick = - dphi_dx * length - ElectronCloudData_get_dipolar_px_kick(el);
        const double py_kick = - dphi_dy * length - ElectronCloudData_get_dipolar_py_kick(el);
        const double pzeta_kick = - dphi_dzeta * length - ElectronCloudData_get_dipolar_pzeta_kick(el);

        // TODO: implement kicks for particles with different charge and or mass
        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);

        double const q = LocalParticle_get_q0(part);
        double const p0c = LocalParticle_get_p0c(part);
        double const energy_change = q * ( ( beta0 * p0c ) * pzeta_kick);
        LocalParticle_add_to_energy(part, energy_change, 1);
    END_PER_PARTICLE_BLOCK;
}

#endif
