// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_CHARGE_DEPOSITION_H
#define XFIELDS_CHARGE_DEPOSITION_H

/*gpufun*/ void p2m_rectmesh3d_one_particle(
        // INPUTS:
        const double x, 
	const double y, 
	const double z,
	  // particle weight
	const double pwei,
          // mesh origin
        const double x0, const double y0, const double z0,
          // mesh distances per cell
        const double dx, const double dy, const double dz,
          // mesh dimension (number of cells)
        const int nx, const int ny, const int nz,
        // OUTPUTS:
        /*gpuglmem*/ double *grid1d
) {

    double vol_m1 = 1/(dx*dy*dz);

    // indices
    int jx = floor((x - x0) / dx);
    int ix = floor((y - y0) / dy);
    int kx = floor((z - z0) / dz);

    // distances
    double dxi = x - (x0 + jx * dx);
    double dyi = y - (y0 + ix * dy);
    double dzi = z - (z0 + kx * dz);

    // weights
    double wijk =    pwei * vol_m1 * (1.-dxi/dx) * (1.-dyi/dy) * (1.-dzi/dz);
    double wi1jk =   pwei * vol_m1 * (1.-dxi/dx) * (dyi/dy)    * (1.-dzi/dz);
    double wij1k =   pwei * vol_m1 * (dxi/dx)    * (1.-dyi/dy) * (1.-dzi/dz);
    double wi1j1k =  pwei * vol_m1 * (dxi/dx)    * (dyi/dy)    * (1.-dzi/dz);
    double wijk1 =   pwei * vol_m1 * (1.-dxi/dx) * (1.-dyi/dy) * (dzi/dz);
    double wi1jk1 =  pwei * vol_m1 * (1.-dxi/dx) * (dyi/dy)    * (dzi/dz);
    double wij1k1 =  pwei * vol_m1 * (dxi/dx)    * (1.-dyi/dy) * (dzi/dz);
    double wi1j1k1 = pwei * vol_m1 * (dxi/dx)    * (dyi/dy)    * (dzi/dz);

    if (jx >= 0 && jx < nx - 1 && ix >= 0 && ix < ny - 1
        	    && kx >= 0 && kx < nz - 1)
    {
        atomicAdd(&grid1d[jx   + ix*nx     + kx*nx*ny],     wijk);
        atomicAdd(&grid1d[jx+1 + ix*nx     + kx*nx*ny],     wij1k);
        atomicAdd(&grid1d[jx   + (ix+1)*nx + kx*nx*ny],     wi1jk);
        atomicAdd(&grid1d[jx+1 + (ix+1)*nx + kx*nx*ny],     wi1j1k);
        atomicAdd(&grid1d[jx   + ix*nx     + (kx+1)*nx*ny], wijk1);
        atomicAdd(&grid1d[jx+1 + ix*nx     + (kx+1)*nx*ny], wij1k1);
        atomicAdd(&grid1d[jx   + (ix+1)*nx + (kx+1)*nx*ny], wi1jk1);
        atomicAdd(&grid1d[jx+1 + (ix+1)*nx + (kx+1)*nx*ny], wi1j1k1);
    }

}


/*gpukern*/ void p2m_rectmesh3d(
        // INPUTS:
          // length of x, y, z arrays
        const int nparticles,
          // particle positions
        /*gpuglmem*/ const double* x, 
	/*gpuglmem*/ const double* y, 
	/*gpuglmem*/ const double* z,
	  // particle weights and stat flags
	/*gpuglmem*/ const double* part_weights,
	/*gpuglmem*/ const int64_t* part_state,
          // mesh origin
        const double x0, const double y0, const double z0,
          // mesh distances per cell
        const double dx, const double dy, const double dz,
          // mesh dimension (number of cells)
        const int nx, const int ny, const int nz,
        // OUTPUTS:
        /*gpuglmem*/ int8_t*  grid1d_buffer,
	             int64_t  grid1d_offset){

    /*gpuglmem*/ double* grid1d = 
		(/*gpuglmem*/ double*)(grid1d_buffer + grid1d_offset);

    #pragma omp parallel for //only_for_context cpu_openmp 
    for (int pidx=0; pidx<nparticles; pidx++){ //vectorize_over pidx nparticles
        if (part_state[pidx] > 0){
    	    double pwei = part_weights[pidx];

            p2m_rectmesh3d_one_particle(x[pidx], y[pidx], z[pidx], pwei,
                                        x0, y0, z0, dx, dy, dz, nx, ny, nz,
                                        grid1d);
	}
    }//end_vectorize
}

/*gpukern*/ void p2m_rectmesh3d_xparticles(
        // INPUTS:
          // length of x, y, z arrays
        const int nparticles,
	ParticlesData particles,
          // mesh origin
        const double x0, const double y0, const double z0,
          // mesh distances per cell
        const double dx, const double dy, const double dz,
          // mesh dimension (number of cells)
        const int nx, const int ny, const int nz,
        // OUTPUTS:
        /*gpuglmem*/ int8_t*  grid1d_buffer,
	             int64_t  grid1d_offset){

    /*gpuglmem*/ double* grid1d = 
    	(/*gpuglmem*/ double*)(grid1d_buffer + grid1d_offset);
    
    /*gpuglmem*/ const double* x = ParticlesData_getp1_x(particles, 0); 
    /*gpuglmem*/ const double* y = ParticlesData_getp1_y(particles, 0); 
    /*gpuglmem*/ const double* z = ParticlesData_getp1_zeta(particles, 0);
    /*gpuglmem*/ const double* part_weights = ParticlesData_getp1_weight(
    		                                             particles, 0);
    /*gpuglmem*/ const int64_t* part_state = ParticlesData_getp1_state(
    		                                             particles, 0);
    // TODO I am forgetting about charge_ratio and mass_ratio
    const double q0_coulomb = QELEM * ParticlesData_get_q0(particles);

    #pragma omp parallel for //only_for_context cpu_openmp 
    for (int pidx=0; pidx<nparticles; pidx++){ //vectorize_over pidx nparticles
        if (part_state[pidx] > 0){
    	    double pwei = part_weights[pidx] * q0_coulomb;

            p2m_rectmesh3d_one_particle(x[pidx], y[pidx], z[pidx], pwei,
                                        x0, y0, z0, dx, dy, dz, nx, ny, nz,
                                        grid1d);
	}
    }//end_vectorize

}
#endif
