#ifndef XFIELDS_ELECTRONCLOUD_H
#define XFIELDS_ELECTRONCLOUD_H

/*gpufun*/
void ElectronCloud_track_local_particle(
		 ElectronCloudData el, LocalParticle* part0){

    //TODO: kick_z flag needs to be inserted in xobject

    const double length = ElectronCloudData_get_length(el);
    ///*gpuglmem*/ double* dphi_dx_map = ElectronCloudData_getp1_fieldmap_dphi_dx(el, 0);
    ///*gpuglmem*/ double* dphi_dy_map = ElectronCloudData_getp1_fieldmap_dphi_dy(el, 0);
    //TriLinearInterpolatedFieldMapData fmap = ElectronCloudData_getp_fieldmap(el);

    //start_per_particle_block (part0->part)
	double const x = LocalParticle_get_x(part);
	double const y = LocalParticle_get_y(part);
	double const zeta = LocalParticle_get_zeta(part);

	double const q0 = LocalParticle_get_q0(part);
	double const mass0 = LocalParticle_get_mass0(part);
	double const chi = LocalParticle_get_chi(part);
	double const beta0 = LocalParticle_get_beta0(part);
	double const gamma0 = LocalParticle_get_gamma0(part);
	double const rvv = LocalParticle_get_rvv(part);

	double const tau = zeta / ( beta0 * rvv );

	double px_kick=0;
	double py_kick=0;
	double ptau_kick=0;
	
//	const IndicesAndWeights iw = 
//	    TriLinearInterpolatedFieldMap_compute_indeces_and_weights(fmap, x, y, z);
//
//   	const double dphi_dx = 
//	    TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(dphi_dx_map, iw);
//   	const double dphi_dy = 
//	    TriLinearInterpolatedFieldMap_interpolate_3d_map_scalar(dphi_dy_map, iw);
//
//        const double charge_mass_ratio = 
//		             chi*QELEM*q0/(mass0*QELEM/(C_LIGHT*C_LIGHT));
//        const double factor = -(charge_mass_ratio
//                                *length*(1.-beta0*beta0)
//                                /(gamma0*beta0*beta0*C_LIGHT*C_LIGHT));
//
//	LocalParticle_add_to_px(part, factor*dphi_dx);
//	LocalParticle_add_to_py(part, factor*dphi_dy);
//
    //end_per_particle_block
}

#endif
