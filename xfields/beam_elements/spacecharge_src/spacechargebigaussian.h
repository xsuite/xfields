// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_SPACECHARGEBIGAUSSIAN_H
#define XFIELDS_SPACECHARGEBIGAUSSIAN_H

/*gpufun*/
void SpaceChargeBiGaussian_track_local_particle(
		 SpaceChargeBiGaussianData el, LocalParticle* part0){

    const double length = SpaceChargeBiGaussianData_get_length(el);
    BiGaussianFieldMapData fmap = SpaceChargeBiGaussianData_getp_fieldmap(el);
    LongitudinalProfileQGaussianData prof =
	    SpaceChargeBiGaussianData_getp_longitudinal_profile(el);

	const int64_t z_kick_num_integ_per_sigma =
		SpaceChargeBiGaussianData_get_z_kick_num_integ_per_sigma(el);

    //start_per_particle_block (part0->part)
		double const x = LocalParticle_get_x(part);
		double const y = LocalParticle_get_y(part);
		double const z = LocalParticle_get_zeta(part);

		double const q0 = LocalParticle_get_q0(part);
		double const mass0 = LocalParticle_get_mass0(part);
		double const chi = LocalParticle_get_chi(part);
		double const beta0 = LocalParticle_get_beta0(part);
		double const gamma0 = LocalParticle_get_gamma0(part);

		double dphi_dx, dphi_dy;
		BiGaussianFieldMap_get_dphi_dx_dphi_dy(fmap, x, y,
							&dphi_dx, &dphi_dy);

		const double lambda_z =
			LongitudinalProfileQGaussian_line_density_scalar(prof, z);

		const double charge_mass_ratio =
						chi*QELEM*q0/(mass0*QELEM/(C_LIGHT*C_LIGHT));
		const double factor = -(charge_mass_ratio*q0*QELEM
								*length*(1.-beta0*beta0)
								/(gamma0*beta0*beta0*C_LIGHT*C_LIGHT));

		LocalParticle_add_to_px(part, factor*lambda_z*dphi_dx);
		LocalParticle_add_to_py(part, factor*lambda_z*dphi_dy);

		if (z_kick_num_integ_per_sigma > 0){

			const int64_t x_sigma = (int64_t)(x / BiGaussianFieldMapData_get_sigma_x(fmap));
			const int64_t y_sigma = (int64_t)(y / BiGaussianFieldMapData_get_sigma_y(fmap));

			int64_t n_sigma;
			if (x_sigma > y_sigma){
				n_sigma = x_sigma;
			}
			else{
				n_sigma = y_sigma;
			}

			if (n_sigma < 1){
				n_sigma = 1;
			}
			const int64_t n_integ = n_sigma * z_kick_num_integ_per_sigma;

			const double dx = fabs(x / (n_integ - 1));
			const double dy = fabs(y / (n_integ - 1));

			double phi = 0.0;
			double dphi_integ = 0.0;
			for (int ii=0; ii<n_integ; ii++){
				double const x_integ = ii*dx;
				double const y_integ = ii*dy;
				double dphi_dx_integ, dphi_dy_integ;
				BiGaussianFieldMap_get_dphi_dx_dphi_dy(fmap, x_integ, y_integ,
									            &dphi_dx_integ, &dphi_dy_integ);

				dphi_integ = dphi_dx_integ*dx + dphi_dy_integ*dy;

				phi += dphi_integ;
			}
			phi -= 0.5 * dphi_integ;

			double const lam_prime =
				LongitudinalProfileQGaussian_line_density_derivative_scalar(prof, z);

			LocalParticle_update_delta(part,
				LocalParticle_get_delta(part) + factor*lam_prime*phi);

		}

    //end_per_particle_block
}


#endif
