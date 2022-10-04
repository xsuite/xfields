# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp
from matplotlib import pyplot as plt

def stat_emittance(beam, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0):
    """
    compute statistical emittances. First normalize coordinates by using (263) then (130) from
    https://arxiv.org/pdf/2107.02614.pdf
    """
        
    x_norm     = beam.x / np.sqrt(beta_x)
    y_norm     = beam.y / np.sqrt(beta_y)
    px_norm    = alpha_x / beta_x * beam.x + beta_x * beam.px
    py_norm    = alpha_y / beta_y * beam.y + beta_y * beam.py   
    
    emit_x = np.sqrt(np.mean(( x_norm -  np.mean(x_norm))**2) *\
                     np.mean((px_norm - np.mean(px_norm))**2) -\
                     np.mean(( x_norm -  np.mean(x_norm)) *\
                             (px_norm - np.mean(px_norm)))**2)
        
    emit_y = np.sqrt(np.mean(( y_norm -  np.mean(y_norm))**2) *\
                     np.mean((py_norm - np.mean(py_norm))**2) -\
                     np.mean(( y_norm -  np.mean(y_norm)) *\
                             (py_norm - np.mean(py_norm)))**2)
        
    emit_s = np.sqrt(np.mean(( beam.zeta -  np.mean(beam.zeta))**2) *\
                     np.mean((beam.delta - np.mean(beam.delta))**2) -\
                     np.mean(( beam.zeta -  np.mean(beam.zeta)) *\
                             (beam.delta - np.mean(beam.delta)))**2)
        
    return emit_x, emit_y, emit_s

context = xo.ContextCpu(omp_num_threads=0)

###########
# ttbar 2 #
###########
bunch_intensity     = 2.3e11  # [1]
energy              = 182.5  # [GeV]
p0c                 = 182.5e9  # [eV]
mass0               = .511e6  # [eV]
phi                 = 15e-3  # [rad] half xing
u_sr                = 9.2  # [GeV]
u_bs                = .0114  # [GeV]
k2_factor           = .4  # [1]
qx                  = .554  # [1] half arc
qy                  = .588  # [1]
qs                  = .0436  # [1]
physemit_x          = 1.46e-09  # [m]
physemit_y          = 2.9e-12  # [m]
beta_x              = 1  # [m]
beta_y              = .0016  # [m]
sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
sigma_z             = .00194  # [m] sr
sigma_z_tot         = .00254  # [m] sr+bs
sigma_delta         = .0015  # [m]
sigma_delta_tot     = .00192  # [m]
beta_s              = sigma_z/sigma_delta  # [m]
physemit_s          = sigma_z*sigma_delta  # [m]
physemit_s_tot      = sigma_z_tot*sigma_delta_tot  # [m]
n_macroparticles_b1 = int(1e5)
n_macroparticles_b2 = int(1e6)

#############
# particles #
#############

#e-
particles_b1 = xp.Particles(
            _context = context, 
            q0        = -1,
            p0c       = p0c,
            mass0     = mass0,
            x         = sigma_x        *np.random.randn(n_macroparticles_b1),
            y         = sigma_y        *np.random.randn(n_macroparticles_b1),
            zeta      = sigma_z        *np.random.randn(n_macroparticles_b1),
            px        = sigma_px       *np.random.randn(n_macroparticles_b1),
            py        = sigma_py       *np.random.randn(n_macroparticles_b1),
            delta     = sigma_delta    *np.random.randn(n_macroparticles_b1),
            )

# e+
particles_b2 = xp.Particles(
            _context = context, 
            q0        = 1,
            p0c       = p0c,
            mass0     = mass0,
            x         = sigma_x        *np.random.randn(n_macroparticles_b2),
            y         = sigma_y        *np.random.randn(n_macroparticles_b2),
            zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b2),
            px        = sigma_px       *np.random.randn(n_macroparticles_b2),
            py        = sigma_py       *np.random.randn(n_macroparticles_b2),
            delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b2),
            )

particles_b1.name = "b1"
particles_b2.name = "b2"

particles_b1._init_random_number_generator()
particles_b2._init_random_number_generator()

n_turns = 1
n_slices = 100

##########################################
# weakstrong beambeam with beamstrahlung #
##########################################

bin_edges = sigma_z_tot*np.linspace(-3.0,3.0,n_slices+1)
slicer = xf.TempSlicer(bin_edges=bin_edges)

# slice intensity [num. real particles] n_slices inferred from length of this
slices_other_beam_num_particles = np.zeros(n_slices)
# unboosted strong beam moments  
slices_other_beam_x_center    = np.zeros_like(slices_other_beam_num_particles)
slices_other_beam_zeta_center = np.zeros_like(slices_other_beam_num_particles)
slices_other_beam_Sigma_11    = np.zeros_like(slices_other_beam_num_particles)
slices_other_beam_Sigma_22    = np.zeros_like(slices_other_beam_num_particles)
slices_other_beam_Sigma_33    = np.zeros_like(slices_other_beam_num_particles)
slices_other_beam_Sigma_44    = np.zeros_like(slices_other_beam_num_particles)
# only if BS on
slices_other_beam_zeta_bin_width = np.abs(np.diff(slicer.bin_edges))

for i in range(n_slices):
    particles_in_slice = np.argwhere(slicer.get_slice_indices(particles_b2)==i)
    
    slices_other_beam_num_particles[i] = int(len(particles_in_slice))
    
    # nan if 0 particles in slice
    slices_other_beam_x_center     [i] = particles_b2.x   [particles_in_slice].mean()  
    slices_other_beam_zeta_center  [i] = particles_b2.zeta[particles_in_slice].mean()
    
    # nan if 0 particles in slice, 0 if 1 particle in slice
    slices_other_beam_Sigma_11     [i] = particles_b2.x   [particles_in_slice].std()
    slices_other_beam_Sigma_22     [i] = particles_b2.px  [particles_in_slice].std()
    slices_other_beam_Sigma_33     [i] = particles_b2.y   [particles_in_slice].std()
    slices_other_beam_Sigma_44     [i] = particles_b2.py  [particles_in_slice].std()

    # change nans to 0
    slices_other_beam_x_center[np.isnan(slices_other_beam_x_center)] = 0
    slices_other_beam_zeta_center[np.isnan(slices_other_beam_zeta_center)] = 0
    slices_other_beam_Sigma_11[np.isnan(slices_other_beam_Sigma_11)] = 0
    slices_other_beam_Sigma_22[np.isnan(slices_other_beam_Sigma_22)] = 0
    slices_other_beam_Sigma_33[np.isnan(slices_other_beam_Sigma_33)] = 0
    slices_other_beam_Sigma_44[np.isnan(slices_other_beam_Sigma_44)] = 0
    
el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
        _context=context,
        config_for_update = None,
        other_beam_q0=1,
        phi=phi,
        alpha=0,
        # decide between round or elliptical kick formula
        min_sigma_diff     = 1e-28,
        # slice intensity [num. real particles] n_slices inferred from length of this
        slices_other_beam_num_particles      = bunch_intensity/n_macroparticles_b2*slices_other_beam_num_particles,
        slices_other_beam_num_macroparticles = slices_other_beam_num_particles,
        # unboosted strong beam moments  
        slices_other_beam_x_center    = slices_other_beam_x_center,
        slices_other_beam_zeta_center = slices_other_beam_zeta_center,
        slices_other_beam_Sigma_11    = slices_other_beam_Sigma_11**2,
        slices_other_beam_Sigma_22    = slices_other_beam_Sigma_22**2,
        slices_other_beam_Sigma_33    = slices_other_beam_Sigma_33**2,
        slices_other_beam_Sigma_44    = slices_other_beam_Sigma_44**2,
        # only if BS on
        do_beamstrahlung = 1,
        slices_other_beam_zeta_bin_width_star = slices_other_beam_zeta_bin_width*np.cos(phi),  # boosted dz
        # that has to be set
        slices_other_beam_Sigma_12_star    = n_slices*[0],
        slices_other_beam_Sigma_13_star    = n_slices*[0],
        slices_other_beam_Sigma_14_star    = n_slices*[0],
        slices_other_beam_Sigma_23_star    = n_slices*[0],
        slices_other_beam_Sigma_24_star    = n_slices*[0],
        slices_other_beam_Sigma_34_star    = n_slices*[0],
)

#########
# track #
#########

particles_b1.slice_id = slicer.get_slice_indices(particles_b1)
    
line = xt.Line(elements = [el_beambeam_b1])

tracker = xt.Tracker(line=line)
record = tracker.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(1e6)})
tracker.track(particles_b1, num_turns=n_turns)
tracker.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

#########
# plots #
#########

# record.beamstrahlungtable contains the photons
plt.hist(record.beamstrahlungtable.photon_energy/1e9, bins=np.logspace(np.log10(1e-14), np.log10(1e1), 100), histtype="step");
plt.xscale("log")
plt.yscale("log")
plt.xlabel("E [GeV]")
plt.ylabel("Count [1]")
plt.show()
