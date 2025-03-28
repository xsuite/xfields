import pathlib
import pytest

import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts(excluding="ContextPyopencl")
def test_beambeam3d_beamstrahlung_pic(test_context):

    if isinstance(test_context, xo.ContextCupy):
        print(f"[test.py] default_blocksize: {test_context.default_block_size}")
        import cupy as cp

    print(repr(test_context))

    # fcc ttbar 4 IP
    # https://indico.cern.ch/event/1202105/contributions/5408583/attachments/2659051/4608141/FCCWeek_Optics_Oide_230606.pdf
    bunch_intensity     =  1.55e11  # [e]
    energy              =  182.5  # [GeV]
    p0c                 =  182.5e9  # [eV]
    mass0               =  511000  # [eV]
    phi                 =  15e-3  # [rad] half xing
    physemit_x          =  1.59e-09  # [m]
    physemit_y          =  9e-13  # [m]
    beta_x              =  1  # [m]
    beta_y              =  1.6e-3  # [m]
    sigma_x             =  np.sqrt(physemit_x*beta_x)  # [m]
    sigma_px            =  np.sqrt(physemit_x/beta_x)  # [1]
    sigma_y             =  np.sqrt(physemit_y*beta_y)  # [m]
    sigma_py            =  np.sqrt(physemit_y/beta_y)  # [1]
    sigma_z_tot         =  21.7e-4  # [m] sr+bs
    sigma_delta_tot     =  19.2e-4  # [1] sr+bs
    n_ip                =  4  # [1]
    
    n_macroparticles_b1 = int(1e6)
    n_macroparticles_b2 = int(1e6)
    
    ################
    # create beams #
    ################
    
    #e-
    particles_b1_pic = xp.Particles(
                _context = test_context, 
                q0        = -1,
                p0c       = p0c,
                mass0     = mass0,
                x         = sigma_x        *np.random.randn(n_macroparticles_b1),
                y         = sigma_y        *np.random.randn(n_macroparticles_b1),
                zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b1),
                px        = sigma_px       *np.random.randn(n_macroparticles_b1),
                py        = sigma_py       *np.random.randn(n_macroparticles_b1),
                delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b1),
                weight=bunch_intensity/n_macroparticles_b1
                )
    
    # e+
    particles_b2_pic = xp.Particles(
                _context = test_context, 
                q0        = 1,
                p0c       = p0c,
                mass0     = mass0,
                x         = sigma_x        *np.random.randn(n_macroparticles_b2),
                y         = sigma_y        *np.random.randn(n_macroparticles_b2),
                zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b2),
                px        = sigma_px       *np.random.randn(n_macroparticles_b2),
                py        = sigma_py       *np.random.randn(n_macroparticles_b2),
                delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b2),
                weight=bunch_intensity/n_macroparticles_b2
                )
    
    ####################
    # specify PIC grid #
    ####################
    
    n_slices = 100    

    x_lim_grid = 6 * sigma_x
    y_lim_grid = 6 * sigma_y
    z_lim_grid = 6 * sigma_z_tot
    
    bbpic_ip1_b1 = xf.BeamBeamPIC3D(
        _context=test_context,
            phi=phi, alpha=0,
            x_range=(-x_lim_grid, x_lim_grid), dx=2*x_lim_grid/(n_slices),
            y_range=(-y_lim_grid, y_lim_grid), dy=2*y_lim_grid/(n_slices),
            z_range=(-z_lim_grid, z_lim_grid), dz=2*z_lim_grid/(n_slices),
            )
    
    bbpic_ip1_b2 = xf.BeamBeamPIC3D(
        _context=test_context,
            phi=-phi, alpha=0,
            x_range=(-x_lim_grid, x_lim_grid), dx=2*x_lim_grid/(n_slices),
            y_range=(-y_lim_grid, y_lim_grid), dy=2*y_lim_grid/(n_slices),
            z_range=(-z_lim_grid, z_lim_grid), dz=2*z_lim_grid/(n_slices),
            )
    
    #######################
    # set up communicator #
    #######################
    
    pipeline_manager = xt.PipelineManager()
    pipeline_manager.add_particles('p_b1', rank=0)
    pipeline_manager.add_particles('p_b2', rank=0)
    
    pipeline_manager.add_element('IP1')
    bbpic_ip1_b1.name = 'IP1'
    bbpic_ip1_b2.name = 'IP1'
    bbpic_ip1_b1.partner_name = 'p_b2'
    bbpic_ip1_b2.partner_name = 'p_b1'
    particles_b1_pic.init_pipeline('p_b1')
    particles_b2_pic.init_pipeline('p_b2')
    bbpic_ip1_b1.pipeline_manager = pipeline_manager
    bbpic_ip1_b2.pipeline_manager = pipeline_manager
    
    ######################
    # set up xtrack line #
    ######################
    
    line_b1 = xt.Line(elements=[bbpic_ip1_b1])
    line_b2 = xt.Line(elements=[bbpic_ip1_b2])
    
    line_b1.build_tracker(_context=test_context)
    line_b2.build_tracker(_context=test_context)
    
    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_b1, particles=particles_b1_pic),
                  xt.PipelineBranch(line=line_b2, particles=particles_b2_pic)],
        enable_debug_log=True, verbose=True)
    
    ################
    # configure BS #
    ################
    
    assert line_b1._needs_rng == False
    line_b1.configure_radiation(model_beamstrahlung='quantum')
    assert line_b1._needs_rng == True
    
    assert line_b2._needs_rng == False
    line_b2.configure_radiation(model_beamstrahlung='quantum')
    assert line_b2._needs_rng == True
    
    record_b1_pic = line_b1.start_internal_logging_for_elements_of_type(
        xf.BeamBeamPIC3D, capacity={"beamstrahlungtable": int(3e5)})
    record_b2_pic = line_b2.start_internal_logging_for_elements_of_type(
        xf.BeamBeamPIC3D, capacity={"beamstrahlungtable": int(3e5)})

    #####################
    # track 1 collision #
    #####################
    
    multitracker.track(num_turns=1)
    
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamPIC3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamPIC3D)

    record_b1_pic.move(_context=xo.context_default)
    record_b2_pic.move(_context=xo.context_default)

    #####################################
    # test #1: compare against formulas #
    #####################################
    
    # analytical beamstrahlung parameters
    r0 = cst.e**2/(4*np.pi*cst.epsilon_0*cst.m_e*cst.c**2) # - if pp
    upsilon_max = (
        2 * r0**2 * energy/(mass0*1e-9) * bunch_intensity
        / (1/137*sigma_z_tot*(sigma_x + 1.85*sigma_y)))
    upsilon_avg = (5/6 * r0**2 * energy/(mass0*1e-9) * bunch_intensity
                    / (1/137*sigma_z_tot*(sigma_x + sigma_y)))
    
    # get rid of padded zeros in table
    photon_critical_energy_b1 = np.array(sorted(set(record_b1_pic.beamstrahlungtable.photon_critical_energy))[1:])
    photon_critical_energy_b2 = np.array(sorted(set(record_b2_pic.beamstrahlungtable.photon_critical_energy))[1:])
    primary_energy_b1         = np.array(sorted(set(        record_b1_pic.beamstrahlungtable.primary_energy))[1:])
    primary_energy_b2         = np.array(sorted(set(        record_b2_pic.beamstrahlungtable.primary_energy))[1:])
    
    upsilon_avg_sim_b1 = np.mean(0.67 * photon_critical_energy_b1 / primary_energy_b1)
    upsilon_avg_sim_b2 = np.mean(0.67 * photon_critical_energy_b2 / primary_energy_b2)
    upsilon_max_sim_b1 = np.max( 0.67 * photon_critical_energy_b1 / primary_energy_b1)
    upsilon_max_sim_b2 = np.max( 0.67 * photon_critical_energy_b2 / primary_energy_b2)
    
    print("Y max. [1]:", upsilon_max)
    print("Y avg. [1]:", upsilon_avg)
    print("Y max. beam 1 [1]:", upsilon_max_sim_b1)
    print("Y avg. beam 1 [1]:", upsilon_avg_sim_b1)
    print("Y max. beam 2 [1]:", upsilon_max_sim_b2)
    print("Y avg. beam 2 [1]:", upsilon_avg_sim_b2)
    print("Y max. ratio beam 1 [1]:", upsilon_max_sim_b1 / upsilon_max)
    print("Y avg. ratio beam 1 [1]:", upsilon_avg_sim_b1 / upsilon_avg)
    print("Y max. ratio beam 2 [1]:", upsilon_max_sim_b2 / upsilon_max)
    print("Y avg. ratio beam 2 [1]:", upsilon_avg_sim_b2 / upsilon_avg)
    
    rtol = 1e-1
    assert np.allclose(upsilon_max_sim_b1, upsilon_max, rtol=rtol), f"Upsilon max beam 1 does not agree with {rtol} tolerance!"
    assert np.allclose(upsilon_avg_sim_b1, upsilon_avg, rtol=rtol), f"Upsilon avg beam 1 does not agree with {rtol} tolerance!"
    assert np.allclose(upsilon_max_sim_b2, upsilon_max, rtol=rtol), f"Upsilon max beam 2 does not agree with {rtol} tolerance!"
    assert np.allclose(upsilon_avg_sim_b2, upsilon_avg, rtol=rtol), f"Upsilon avg beam 2 does not agree with {rtol} tolerance!"
    
    ###############################################################
    # test #2: compare BS photon E spectrum against soft-gaussian #
    ###############################################################
    
    n_bins = 20
    bins = np.logspace(np.log10(1e-9), np.log10(1e10), 20)
    
    hist_big = np.array([991, 2176, 4559, 9885, 20741, 40261, 56973, 23659, 165])  # this comes from an Xsuite SS simulation with soft-Gaussian bb using 1e6 macroparticles, 100 slices, 1 collision, then the beamstrahlung photon energies from beam 1 are histogrammed using the same binning as above
    photons_pic_b1 = set(record_b1_pic.beamstrahlungtable.photon_energy)
    photons_pic_b2 = set(record_b2_pic.beamstrahlungtable.photon_energy)
    
    # compare the upper part of the energy spectrum bc it has better statistics than low end tail
    hist_pic_b1 = np.histogram(np.array([*photons_pic_b1]), bins=bins)[0][int(n_bins/2):]
    hist_pic_b2 = np.histogram(np.array([*photons_pic_b2]), bins=bins)[0][int(n_bins/2):]
    
    rtol = 20e-1  # should agree within 20%, actually within a few % except for the upper tail bin
    assert np.allclose(hist_pic_b1,    hist_big, rtol=rtol), f"PIC beam 1 and soft-Gaussian beam 1 photon energies do not agree {rtol} tolerance"
    assert np.allclose(hist_pic_b2,    hist_big, rtol=rtol), f"PIC beam 2 and soft-Gaussian beam 1 photon energies do not agree {rtol} tolerance"
    assert np.allclose(hist_pic_b1, hist_pic_b2, rtol=rtol), f"PIC beam 1 and PIC beam 2 photon energies do not agree {rtol} tolerance"
    
