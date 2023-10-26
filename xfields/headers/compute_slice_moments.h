// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_COMPUTESLICEMOMENTS_CUDA
#define XFIELDS_COMPUTESLICEMOMENTS_CUDA //only_for_context cuda
#endif

#ifndef XFIELDS_COMPUTESLICEMOMENTS_CUDA

#ifndef XFIELDS_COMPUTESLICEMOMENTS_H__
#define XFIELDS_COMPUTESLICEMOMENTS_H__

void compute_slice_moments_cuda_sums_per_slice(ParticlesData particles, int64_t* particles_slice, double* moments, const int64_t num_macroparticles, const int64_t n_slices, const int64_t shared_mem_size_bytes){};
void compute_slice_moments_cuda_moments_from_sums(double* moments, const int64_t n_slices, const int64_t weight, const int64_t threshold_num_macroparticles){};

int64_t binary_search(const double* bins, int first, int last, const double x){
    // bins must be in descending order: bins[i-1] >= x > bins[i]. If bins in increasing order, change < to >.
    if (x <= bins[last])
        return last+1;

    while (first <= last)
    {
        int64_t middle = first + (last - first) / 2;
        if (x > bins[middle] && x <= bins[middle-1])
            return middle;
        if (bins[middle] >= x)
        {
            first = middle + 1;
        }
        else
            last = middle - 1;
    }
    return 0;
}

void digitize(ParticlesData particles, const double* particles_zeta, const double* bin_edges, int n_slices, int64_t* particles_slice){
  int n_part = ParticlesData_get__capacity(particles);
  int first = 0;
  int last = n_slices;
  #pragma omp parallel for //only_for_context cpu_openmp
  for (int i=0; i<n_part; i++) {
//      int tid = omp_get_thread_num(); //only_for_context cpu_openmp
//      printf("[digitize] omp thread %d\n", tid); //only_for_context cpu_openmp
      particles_slice[i] = binary_search(bin_edges, first, last, particles_zeta[i]);
  }
}

void compute_slice_moments(ParticlesData particles, int64_t* particles_slice, double* moments, int n_slices, int threshold_n_macroparticles) {
    int n_first_moments = 7;
    int n_second_moments = 10;
    int n_moments = n_first_moments+n_second_moments;
    for(int i = 0;i<n_slices*n_moments;++i) {
        moments[i] = 0.0;
    }
    int n_part = ParticlesData_get__capacity(particles);
    //////// zero and first order moments ///////////////
    #pragma omp parallel default(none) firstprivate(n_part,n_slices,n_first_moments) shared(particles,moments,particles_slice) //only_for_context cpu_openmp
    { //only_for_context cpu_openmp
//        int tid = omp_get_thread_num(); //only_for_context cpu_openmp
//        printf("[compute_slice_moments] omp thread %d\n", tid); //only_for_context cpu_openmp
        double tmpSliceM[n_slices*n_first_moments];
        for(int i = 0;i<n_slices*n_first_moments;++i) {
            tmpSliceM[i] = 0.0;
        }
        #pragma omp for //only_for_context cpu_openmp
        for(int i = 0;i<n_part;++i) {
            int i_slice = particles_slice[i];
            if(i_slice >= 0 && i_slice < n_slices && ParticlesData_get_state(particles,i)>0){
                tmpSliceM[i_slice] += 1.0;
                tmpSliceM[n_slices + i_slice] += ParticlesData_get_x(particles,i);
                tmpSliceM[2*n_slices + i_slice] += ParticlesData_get_px(particles,i);
                tmpSliceM[3*n_slices + i_slice] += ParticlesData_get_y(particles,i);
                tmpSliceM[4*n_slices + i_slice] += ParticlesData_get_py(particles,i);
                tmpSliceM[5*n_slices + i_slice] += ParticlesData_get_zeta(particles,i);
                tmpSliceM[6*n_slices + i_slice] += ParticlesData_get_delta(particles,i);
            }
        }
        //reduction
        #pragma omp critical //only_for_context cpu_openmp
        { //only_for_context cpu_openmp
            for (int i_slice = 0;i_slice<n_slices;++i_slice) {
                for(int j = 0;j<n_first_moments;++j){
                    moments[j*n_slices + i_slice] += tmpSliceM[j*n_slices + i_slice];
                }
            }
        } //only_for_context cpu_openmp
    } //only_for_context cpu_openmp
    for (int i_slice = 0;i_slice<n_slices;++i_slice) {
        if(moments[i_slice] > threshold_n_macroparticles){
            for(int j = 1;j<n_first_moments;++j){
                moments[j*n_slices+i_slice] /= moments[i_slice];
            }
        }else{
            for(int j = 0;j<n_first_moments;++j){
                moments[j*n_slices+i_slice] = 0.0;
            }
        }
    }
    //////// second order moments ///////////////
    #pragma omp parallel default(none) firstprivate(n_part,n_slices,n_first_moments,n_second_moments) shared(particles,moments,particles_slice) //only_for_context cpu_openmp
    { //only_for_context cpu_openmp
        double tmpSliceM2[n_slices*n_second_moments];
        for(int i = 0;i<n_slices*n_second_moments;++i) {
            tmpSliceM2[i] = 0.0;
        }
        #pragma omp for //only_for_context cpu_openmp
        for(int i = 0;i<n_part;++i) {
            int i_slice = particles_slice[i];
            if(i_slice >=0 && i_slice < n_slices && ParticlesData_get_state(particles,i)>0){
                tmpSliceM2[i_slice] += ParticlesData_get_x(particles,i)*ParticlesData_get_x(particles,i); //Sigma_11
                tmpSliceM2[n_slices + i_slice] += ParticlesData_get_x(particles,i)*ParticlesData_get_px(particles,i); //Sigma_12
                tmpSliceM2[2*n_slices + i_slice] += ParticlesData_get_x(particles,i)*ParticlesData_get_y(particles,i); //Sigma_13
                tmpSliceM2[3*n_slices + i_slice] += ParticlesData_get_x(particles,i)*ParticlesData_get_py(particles,i); //Sigma_14
                tmpSliceM2[4*n_slices + i_slice] += ParticlesData_get_px(particles,i)*ParticlesData_get_px(particles,i); //Sigma_22
                tmpSliceM2[5*n_slices + i_slice] += ParticlesData_get_px(particles,i)*ParticlesData_get_y(particles,i); //Sigma_23
                tmpSliceM2[6*n_slices + i_slice] += ParticlesData_get_px(particles,i)*ParticlesData_get_py(particles,i); //Sigma_24
                tmpSliceM2[7*n_slices + i_slice] += ParticlesData_get_y(particles,i)*ParticlesData_get_y(particles,i); //Sigma_33
                tmpSliceM2[8*n_slices + i_slice] += ParticlesData_get_y(particles,i)*ParticlesData_get_py(particles,i); //Sigma_34
                tmpSliceM2[9*n_slices + i_slice] += ParticlesData_get_py(particles,i)*ParticlesData_get_py(particles,i); //Sigma_44
            }
        }

        #pragma omp critical //only_for_context cpu_openmp
        { //only_for_context cpu_openmp
            for (int i_slice = 0;i_slice<n_slices;++i_slice) {
                for(int j = 0;j<n_second_moments;++j) {
                    moments[(n_first_moments+j)*n_slices+i_slice] += tmpSliceM2[j*n_slices+i_slice];
                }
            }
        } //only_for_context cpu_openmp
    } //only_for_context cpu_openmp
    for(int i_slice = 0;i_slice<n_slices;++i_slice) {
        if(moments[i_slice] > threshold_n_macroparticles){
            for(int j = n_first_moments;j<n_moments;++j){
                moments[j*n_slices+i_slice] /= moments[i_slice];
            }
            moments[7*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[n_slices+i_slice]; //Sigma_11
            moments[8*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[2*n_slices+i_slice]; //Sigma_12
            moments[9*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[3*n_slices+i_slice]; //Sigma_13
            moments[10*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_14
            moments[11*n_slices + i_slice] -= moments[2*n_slices+i_slice]*moments[2*n_slices+i_slice]; //Sigma_22
            moments[12*n_slices + i_slice] -= moments[2*n_slices+i_slice]*moments[3*n_slices+i_slice]; //Sigma_23
            moments[13*n_slices + i_slice] -= moments[2*n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_24
            moments[14*n_slices + i_slice] -= moments[3*n_slices+i_slice]*moments[3*n_slices+i_slice]; //Sigma_33
            moments[15*n_slices + i_slice] -= moments[3*n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_34
            moments[16*n_slices + i_slice] -= moments[4*n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_44
            moments[i_slice] *= ParticlesData_get_weight(particles,0);  // added to scale num_macroparts_per_slice to real charge
        }else{
            for(int j = n_first_moments;j<n_moments;++j){
                moments[j*n_slices+i_slice] = 0.0;
            }
        }
    }

}
#endif /* XFIELDS_COMPUTESLICEMOMENTS_H__ */
#endif /* XFIELDS_COMPUTESLICEMOMENTS_CUDA */


#ifdef XFIELDS_COMPUTESLICEMOMENTS_CUDA

#ifndef XFIELDS_COMPUTESLICEMOMENTS_CUH__
#define XFIELDS_COMPUTESLICEMOMENTS_CUH__
__global__ void digitize(ParticlesData particles, const double* particles_zeta, const double* bin_edges, int n_slices, int64_t* particles_slice){};
__global__ void compute_slice_moments(ParticlesData particles, int64_t* particles_slice, double* moments, int n_slices, int threshold_n_macroparticles){};

__global__ void compute_slice_moments_cuda_sums_per_slice(ParticlesData particles,
                        int64_t* particles_slice, double* moments, const int64_t num_macroparticles, const int64_t n_slices, const int64_t shared_mem_size_bytes) {

        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	// init shared memory used for partial sums
        extern __shared__ double sdata[];  // len n_slices * 17 (6 x sum(xi), 10 x sum(xi*xy), 1 x count)
        int full_pass = (int)(17*n_slices / blockDim.x);
        int residual = (17*n_slices)%blockDim.x;
        for (int i=0; i<full_pass; i++){
          sdata[i*blockDim.x + tid] = 0.0;
        }
	if (tid < residual){
          sdata[full_pass*blockDim.x+tid] = 0.0;
        }
        __syncthreads();
        if (gid < num_macroparticles){
          int64_t s_i    = particles_slice[gid];
	  if (s_i >= 0 && s_i < n_slices){
              double x_i     = ParticlesData_get_x(particles,gid);     
              double px_i    = ParticlesData_get_px(particles,gid);    
              double y_i     = ParticlesData_get_y(particles,gid);     
              double py_i    = ParticlesData_get_py(particles,gid);    
              double zeta_i  = ParticlesData_get_zeta(particles,gid);  
              double delta_i = ParticlesData_get_delta(particles,gid); 
    
              // count
              atomicAdd(&sdata[16*n_slices+s_i], 1);
    
              // sum(xi)
              atomicAdd(&sdata[           s_i],     x_i);
    
    	      atomicAdd(&sdata[  n_slices+s_i],    px_i);
    	      atomicAdd(&sdata[2*n_slices+s_i],     y_i);
              atomicAdd(&sdata[3*n_slices+s_i],    py_i);
              atomicAdd(&sdata[4*n_slices+s_i],  zeta_i);
              atomicAdd(&sdata[5*n_slices+s_i], delta_i);
    
              // sum(xi*xj)
              atomicAdd(&sdata[ 6*n_slices+s_i],     x_i*x_i);
              atomicAdd(&sdata[ 7*n_slices+s_i],    x_i*px_i);
              atomicAdd(&sdata[ 8*n_slices+s_i],     x_i*y_i);
              atomicAdd(&sdata[ 9*n_slices+s_i],    x_i*py_i);
              atomicAdd(&sdata[10*n_slices+s_i],   px_i*px_i);
              atomicAdd(&sdata[11*n_slices+s_i],    px_i*y_i);
              atomicAdd(&sdata[12*n_slices+s_i],   px_i*py_i);
              atomicAdd(&sdata[13*n_slices+s_i],     y_i*y_i);
              atomicAdd(&sdata[14*n_slices+s_i],    y_i*py_i);
              atomicAdd(&sdata[15*n_slices+s_i],   py_i*py_i);
	  }
        }
        __syncthreads();

        // write count and first and second order partial sums from shared to global mem
        for (int i=0; i<full_pass; i++){
              atomicAdd(&moments[i*blockDim.x + tid], sdata[i*blockDim.x + tid]);
        }
        if (tid < residual){
              atomicAdd(&moments[full_pass*blockDim.x + tid], sdata[full_pass*blockDim.x + tid]);
        }

	}

__global__ void compute_slice_moments_cuda_moments_from_sums(double* moments, const int64_t n_slices, const int64_t weight, const int64_t threshold_num_macroparticles) {

    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gid >=n_slices) return;  // in case blocksize > n_threads

    // for this n_slices threads are enough
    // one thread computes moments of one slice
    // compute first and second moments in global memory (6 1st order sums + 10 2nd order sums + 1 count + 6 1st order moments + 10 2nd order moments = 33*n_slices)
    if (moments[16*n_slices + gid] > threshold_num_macroparticles){
 
        // first order moments E(xi)
        moments[17*n_slices + gid] = moments[             gid] / moments[16*n_slices + gid];  // E(x)
        moments[18*n_slices + gid] = moments[  n_slices + gid] / moments[16*n_slices + gid];  // E(px)
        moments[19*n_slices + gid] = moments[2*n_slices + gid] / moments[16*n_slices + gid];  // E(y)
        moments[20*n_slices + gid] = moments[3*n_slices + gid] / moments[16*n_slices + gid];  // E(py)
        moments[21*n_slices + gid] = moments[4*n_slices + gid] / moments[16*n_slices + gid];  // E(zeta)
        moments[22*n_slices + gid] = moments[5*n_slices + gid] / moments[16*n_slices + gid];  // E(delta)

        // second order momemnts E(xi*xj)-E(xi)E(xj)
        moments[23*n_slices + gid] = moments[ 6*n_slices + gid] / moments[16*n_slices + gid] - moments[17*n_slices + gid] * moments[17*n_slices + gid];  // Sigma xx
        moments[24*n_slices + gid] = moments[ 7*n_slices + gid] / moments[16*n_slices + gid] - moments[17*n_slices + gid] * moments[18*n_slices + gid];  // Cov xpx
        moments[25*n_slices + gid] = moments[ 8*n_slices + gid] / moments[16*n_slices + gid] - moments[17*n_slices + gid] * moments[19*n_slices + gid];  // Cov xy
        moments[26*n_slices + gid] = moments[ 9*n_slices + gid] / moments[16*n_slices + gid] - moments[17*n_slices + gid] * moments[20*n_slices + gid];  // Cov xpy
        moments[27*n_slices + gid] = moments[10*n_slices + gid] / moments[16*n_slices + gid] - moments[18*n_slices + gid] * moments[18*n_slices + gid];  // Sigma pxpx
        moments[28*n_slices + gid] = moments[11*n_slices + gid] / moments[16*n_slices + gid] - moments[18*n_slices + gid] * moments[19*n_slices + gid];  // Cov pxy
        moments[29*n_slices + gid] = moments[12*n_slices + gid] / moments[16*n_slices + gid] - moments[18*n_slices + gid] * moments[20*n_slices + gid];  // Cov pxpy
        moments[30*n_slices + gid] = moments[13*n_slices + gid] / moments[16*n_slices + gid] - moments[19*n_slices + gid] * moments[19*n_slices + gid];  // Sigma yy
        moments[31*n_slices + gid] = moments[14*n_slices + gid] / moments[16*n_slices + gid] - moments[19*n_slices + gid] * moments[20*n_slices + gid];  // Cov ypy
        moments[32*n_slices + gid] = moments[15*n_slices + gid] / moments[16*n_slices + gid] - moments[20*n_slices + gid] * moments[20*n_slices + gid];  // Sigma pypy

        moments[16*n_slices + gid] *= weight;  // scale from macroparticle to real charge

    }else{ // if not enough macroparticles in slice 0 out everything
        for (int j=0; j<33; j++){
          moments[j*n_slices + gid] = 0.0;
        }
    }
}
#endif /* XFIELDS_COMPUTESLICEMOMENTS_CUH__ */
#endif /* XFIELDS_COMPUTESLICEMOMENTS_CUDA */
