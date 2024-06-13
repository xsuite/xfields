// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_SPACECHARGE3D_H
#define XFIELDS_SPACECHARGE3D_H

/*gpufun*/
void SpaceCharge3D_track_local_particle(
		 SpaceCharge3DData el, LocalParticle* part0){


    const double length = SpaceCharge3DData_get_length(el);
	const int64_t apply_z_kick = SpaceCharge3DData_get_apply_z_kick(el);

    /*gpuglmem*/ double* dphi_dx_map = SpaceCharge3DData_getp1_fieldmap_dphi_dx(el, 0);
    /*gpuglmem*/ double* dphi_dy_map = SpaceCharge3DData_getp1_fieldmap_dphi_dy(el, 0);
	/*gpuglmem*/ double* dphi_dz_map = SpaceCharge3DData_getp1_fieldmap_dphi_dz(el, 0);
    TriLinearInterpolatedFieldMapData fmap = SpaceCharge3DData_getp_fieldmap(el);

    //start_per_particle_block (part0->part)
		double const x = LocalParticle_get_x(part);
		double const y = LocalParticle_get_y(part);
		double const z = LocalParticle_get_zeta(part);

		double const q0 = LocalParticle_get_q0(part);
		double const mass0 = LocalParticle_get_mass0(part);
		double const chi = LocalParticle_get_chi(part);
		double const beta0 = LocalParticle_get_beta0(part);
		double const gamma0 = LocalParticle_get_gamma0(part);

		const IndicesAndWeights iw = 
			TriLinearInterpolatedFieldMap_compute_indeces_and_weights(fmap, x, y, z);

		const double dphi_dx = 
			TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(dphi_dx_map, iw);
		const double dphi_dy = 
			TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(dphi_dy_map, iw);

		const double charge_mass_ratio = 
						chi*QELEM*q0/(mass0*QELEM/(C_LIGHT*C_LIGHT));
		const double factor = -(charge_mass_ratio
								*length*(1.-beta0*beta0)
								/(gamma0*beta0*beta0*C_LIGHT*C_LIGHT));

		LocalParticle_add_to_px(part, factor*dphi_dx);
		LocalParticle_add_to_py(part, factor*dphi_dy);

		if (apply_z_kick > 0){
			const double dphi_dz =
				TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(dphi_dz_map, iw);
			LocalParticle_update_delta(part,
				LocalParticle_get_delta(part) + factor*dphi_dz);
		}

    //end_per_particle_block
}

#endif
