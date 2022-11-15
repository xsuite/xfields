import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

import json

def test_compute_moments():
    for context in xo.context.get_test_contexts():

        #if not isinstance(context, xo.ContextCpu):
        #    print(f'skipping test_beambeam3d_beamstrahlung_single_collision for context {context}')
        #    continue
            
        ###########
        # ttbar 2 #
        ###########
        p0c                 = 182.5e9  # [eV]
        mass0               = .511e6  # [eV]
        
        physemit_x          = 1.46e-09  # [m]
        physemit_y          = 2.9e-12  # [m]
        beta_x              = 1  # [m]
        beta_y              = .0016  # [m]
        sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
        sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
        sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
        sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
        sigma_z_tot         = .00254  # [m] sr+bs
        sigma_delta_tot     = .00192  # [m]
        n_macroparticles_b1 = int(1e2)
        
        n_slices = 2
        threshold_num_macroparticles=20
        
        #############
        # particles #
        #############
        
        #e-
        part_range = np.linspace(-1e-1,1e-1,n_macroparticles_b1)
        particles_b0 = xp.Particles(
                    _context = context,
                    q0        = -1,
                    p0c       = p0c,
                    mass0     = mass0,
                    x         = part_range,
                    zeta      = part_range,
                    )
        
        bin_edges = np.linspace(-1e-1,1e-1,n_slices+1)
        slicer = xf.TempSlicer(bin_edges=bin_edges)
        
        particles_b1 = particles_b0.copy()
        particles_b2 = particles_b0.copy()
        particles_b2.state[:int(n_macroparticles_b1/4)] = 0  # set 1/4 of the particles to lost
        
        # compute slice moments: lost particles are labeled with state=0 and their slice idx will be set to -1
        slice_moments_b1 = slicer.compute_moments(particles_b1, threshold_num_macroparticles=threshold_num_macroparticles)
        slice_moments_b2 = slicer.compute_moments(particles_b2, threshold_num_macroparticles=threshold_num_macroparticles)
        
        other_beam_num_particles_b1 = slice_moments_b1[:n_slices]
        x_center_b1     = slice_moments_b1[   n_slices:2*n_slices]
        Sigma_11_b1 = slice_moments_b1[ 7*n_slices: 8*n_slices]
        
        other_beam_num_particles_b2 = slice_moments_b2[:n_slices]
        x_center_b2     = slice_moments_b2[   n_slices:2*n_slices]
        Sigma_11_b2 = slice_moments_b2[ 7*n_slices: 8*n_slices]
        
        # check if all lost particles have slice idx = -1
        assert np.all(particles_b2.slice[particles_b2.state == 0] == -1)
        
        # check if the mean and std of the alive particles in each slice agrees with compute_moments
        for s in range(n_slices):
            slice_b1 = particles_b1.x[particles_b1.slice==s]
            num_parts_slice_b1 = len(slice_b1)
            mean_b1  = np.mean(slice_b1)
            diff_b1  = slice_b1 - mean_b1
            sigma_b1 = float((diff_b1**2).sum()) / len(slice_b1)
            
            if num_parts_slice_b1 > threshold_num_macroparticles:
                assert num_parts_slice_b1 == other_beam_num_particles_b1[s]
                assert mean_b1 == x_center_b1[s]
                assert sigma_b1 == Sigma_11_b1[s]
            else:
                print(f"Slice {s} has insufficient ({num_parts_slice_b1}) particles! Need at least {threshold_num_macroparticles}.")
            
            slice_b2 = particles_b2.x[particles_b2.slice==s]
            num_parts_slice_b2 = len(slice_b2)
            mean_b2  = np.mean(slice_b2)
            diff_b2  = slice_b2 - mean_b2
            sigma_b2 = float((diff_b2**2).sum()) / len(slice_b2)
            if num_parts_slice_b2 > threshold_num_macroparticles:
                assert num_parts_slice_b2 == other_beam_num_particles_b2[s]
                assert mean_b2 == x_center_b2[s]
                assert sigma_b2 == Sigma_11_b2[s]
            else:
                print(f"Slice {s} has insufficient ({num_parts_slice_b2}) particles! Need at least {threshold_num_macroparticles}.")



def test_beambeam3d_beamstrahlung_single_collision():
    for context in xo.context.get_test_contexts():

        #if not isinstance(context, xo.ContextCpu):
        #    print(f'skipping test_beambeam3d_beamstrahlung_single_collision for context {context}')
        #    continue

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
        n_macroparticles_b1 = int(1e6)
        n_macroparticles_b2 = int(1e6)
        
        n_slices = 100

        # strong bunch intenisities, generated from 1e8 gaussian distributed particles. See test_data/beamstrahlung/gen_nbb.py
        n_bb_json = json.load(open('../test_data/beamstrahlung/gen_nbb.json'))
        n_bb = np.array(n_bb_json["n_bb"])

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
                    zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b1),
                    px        = sigma_px       *np.random.randn(n_macroparticles_b1),
                    py        = sigma_py       *np.random.randn(n_macroparticles_b1),
                    delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b1),
                    )
        
        particles_b1.name = "b1"
        
        particles_b1._init_random_number_generator()
        
        bin_edges = sigma_z_tot*np.linspace(-3.0,3.0,n_slices+1)
        slicer = xf.TempSlicer(bin_edges=bin_edges)

        el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
        _context=context,
        config_for_update = None,
        other_beam_q0=1,
        phi=phi,
        alpha=0,
        # decide between round or elliptical kick formula
        min_sigma_diff     = 1e-28,
        # slice intensity [num. real particles] n_slices inferred from length of this
        slices_other_beam_num_particles = n_bb,
        # unboosted strong beam moments
        slices_other_beam_zeta_center = slicer.bin_centers,
        slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
        slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
        slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
        slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
        # only if BS on
        flag_beamstrahlung = 1,
        slices_other_beam_zeta_bin_width_star_beamstrahlung = np.abs(np.diff(slicer.bin_edges))/np.cos(phi),  # boosted dz
        # has to be set
        slices_other_beam_Sigma_12    = n_slices*[0],
        slices_other_beam_Sigma_34    = n_slices*[0],
        )
        
        #########################
        # track for 1 collision #
        #########################
        
        line = xt.Line(elements = [el_beambeam_b1])
        
        tracker = xt.Tracker(_context=context, line=line)
        record = tracker.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5)})
        tracker.track(particles_b1, num_turns=1)
        tracker.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

        record.move(_context=xo.context_default)

        ###########################################
        # test 1: compare spectrum with guineapig #
        ###########################################

        fname="../test_data/beamstrahlung/guineapig_ttbar2_beamstrahlung_photon_energies_gev.txt"
        guinea_photons = np.loadtxt(fname)  # contains about 250k photons emitted from 1e6 macroparticles in 1 collision
        n_bins = 10
        bins = np.logspace(np.log10(1e-14), np.log10(1e1), n_bins)
        xsuite_hist = np.histogram(record.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]
        guinea_hist = np.histogram(guinea_photons, bins=bins)[0]
        
        bin_rel_errors = np.abs(xsuite_hist - guinea_hist) / guinea_hist
        print(f"bin relative errors [1]: {bin_rel_errors}")
        
        # test if relative error in the last 5 bins is smaller than 1e-1
        assert np.allclose(xsuite_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)

        ############################################
        # test 2: compare beamstrahlung parameters #
        ############################################

        # average and maximum BS parameter
        # https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders
        # page 20
        r0 = cst.e**2/(4*np.pi*cst.epsilon_0*cst.m_e*cst.c**2) # - if pp
        
        upsilon_max =   2 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + 1.85*sigma_y))
        upsilon_avg = 5/6 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + sigma_y))
        
        # get rid of padded zeros in table
        photon_critical_energy = np.array(sorted(set(record.beamstrahlungtable.photon_critical_energy))[1:])
        primary_energy         = np.array(sorted(set(        record.beamstrahlungtable.primary_energy))[1:])
        
        upsilon_avg_sim = np.mean(0.67 * photon_critical_energy / primary_energy)
        upsilon_max_sim = np.max(0.67 * photon_critical_energy / primary_energy)
        
        print("Y max. [1]:", upsilon_max)
        print("Y avg. [1]:", upsilon_avg)
        print("Y max. [1]:", upsilon_max_sim)
        print("Y avg. [1]:", upsilon_avg_sim)
        print("Y max. ratio [1]:", upsilon_max_sim / upsilon_max)
        print("Y avg. ratio [1]:", upsilon_avg_sim / upsilon_avg)
        
        assert np.abs(upsilon_max_sim / upsilon_max - 1) < 2e-2
        assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2



def test_beambeam3d_collective_beamstrahlung_single_collision():
    for context in xo.context.get_test_contexts():

        #if not isinstance(context, xo.ContextCpu):
        #    print(f'skipping test_beambeam3d_collective_beamstrahlung_single_collision for context {context}')
        #    continue
    
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
        n_macroparticles_b1 = int(1e6)
        n_macroparticles_b2 = int(1e6)
        
        n_slices = 100
        
        # strong bunch intenisities, generated from 1e8 gaussian distributed particles. See test_data/beamstrahlung/gen_nbb.py
        n_bb_json = json.load(open('../test_data/beamstrahlung/gen_nbb.json'))
        n_bb = np.array(n_bb_json["n_bb"])
       
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
                    zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b1),
                    px        = sigma_px       *np.random.randn(n_macroparticles_b1),
                    py        = sigma_py       *np.random.randn(n_macroparticles_b1),
                    delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b1),
                    )
        
        particles_b1.name = "b1"
        
        particles_b1._init_random_number_generator()
        
        bin_edges = sigma_z_tot*np.linspace(-3.0,3.0,n_slices+1)
        slicer = xf.TempSlicer(bin_edges=bin_edges)
        
        # this is different w.r.t WS test
        config_for_update=xf.ConfigForUpdateBeamBeamBiGaussian3D(
                        pipeline_manager=None,
                        element_name="beambeam",
                        slicer=slicer,
                        update_every=None, # Never updates (test in weakstrong mode)
                        )
            
        el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
        _context=context,
        config_for_update = config_for_update,
        other_beam_q0=1,
        phi=phi,
        alpha=0,
        # decide between round or elliptical kick formula
        min_sigma_diff     = 1e-28,
        # slice intensity [num. real particles] n_slices inferred from length of this
        slices_other_beam_num_particles = n_bb,
        # unboosted strong beam moments
        slices_other_beam_zeta_center = slicer.bin_centers,
        slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
        slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
        slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
        slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
        # only if BS on
        flag_beamstrahlung = 1,
        slices_other_beam_zeta_bin_width_star_beamstrahlung = np.abs(np.diff(slicer.bin_edges))/np.cos(phi),  # boosted dz
        # has to be set
        slices_other_beam_Sigma_12    = n_slices*[0],
        slices_other_beam_Sigma_34    = n_slices*[0],
        )
        
        el_beambeam_b1.name = "beambeam"
        
        #########################
        # track for 1 collision #
        #########################
        
        line = xt.Line(elements = [el_beambeam_b1])
        
        tracker = xt.Tracker(_context=context, line=line)
        record = tracker.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5)})
        tracker.track(particles_b1, num_turns=1)
        tracker.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

        record.move(_context=xo.context_default)

        ###########################################
        # test 1: compare spectrum with guineapig #
        ###########################################

        fname="../test_data/beamstrahlung/guineapig_ttbar2_beamstrahlung_photon_energies_gev.txt"
        guinea_photons = np.loadtxt(fname)  # contains about 250k photons emitted from 1e6 macroparticles in 1 collision
        n_bins = 10
        bins = np.logspace(np.log10(1e-14), np.log10(1e1), n_bins)
        xsuite_hist = np.histogram(record.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]
        guinea_hist = np.histogram(guinea_photons, bins=bins)[0]
        
        bin_rel_errors = np.abs(xsuite_hist - guinea_hist) / guinea_hist
        print(f"bin relative errors [1]: {bin_rel_errors}")
        
        # test if relative error in the last 5 bins is smaller than 1e-1
        assert np.allclose(xsuite_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)

        ############################################
        # test 2: compare beamstrahlung parameters #
        ############################################

        # average and maximum BS parameter
        # https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders
        # page 20
        r0 = cst.e**2/(4*np.pi*cst.epsilon_0*cst.m_e*cst.c**2) # - if pp
        
        upsilon_max =   2 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + 1.85*sigma_y))
        upsilon_avg = 5/6 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + sigma_y))
        
        # get rid of padded zeros in table
        photon_critical_energy = np.array(sorted(set(record.beamstrahlungtable.photon_critical_energy))[1:])
        primary_energy         = np.array(sorted(set(        record.beamstrahlungtable.primary_energy))[1:])
        
        upsilon_avg_sim = np.mean(0.67 * photon_critical_energy / primary_energy)
        upsilon_max_sim = np.max(0.67 * photon_critical_energy / primary_energy)
        
        print("Y max. [1]:", upsilon_max)
        print("Y avg. [1]:", upsilon_avg)
        print("Y max. [1]:", upsilon_max_sim)
        print("Y avg. [1]:", upsilon_avg_sim)
        print("Y max. ratio [1]:", upsilon_max_sim / upsilon_max)
        print("Y avg. ratio [1]:", upsilon_avg_sim / upsilon_avg)
        
        assert np.abs(upsilon_max_sim / upsilon_max - 1) < 2e-2
        assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2


