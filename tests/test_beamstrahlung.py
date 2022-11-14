import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

def test_beambeam3d_beamstrahlung_single_collision():
    for context in xo.context.get_test_contexts():

        if not isinstance(context, xo.ContextCpu):
            print(f'skipping test_beambeam3d_beamstrahlung_single_collision for context {context}')
            continue

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
        n_bb = np.array([6.6884000e+07, 7.9522500e+07, 9.5608700e+07, 1.1205830e+08,
        1.3225230e+08, 1.5601820e+08, 1.8134120e+08, 2.1465900e+08,
        2.4764560e+08, 2.8871670e+08, 3.3381510e+08, 3.8250380e+08,
        4.3826040e+08, 5.0107570e+08, 5.6947540e+08, 6.4710960e+08,
        7.3120450e+08, 8.2246850e+08, 9.2371450e+08, 1.0346021e+09,
        1.1507452e+09, 1.2783837e+09, 1.4063856e+09, 1.5577785e+09,
        1.7084170e+09, 1.8692997e+09, 2.0361992e+09, 2.2154359e+09,
        2.3930373e+09, 2.5800457e+09, 2.7800560e+09, 2.9737367e+09,
        3.1711020e+09, 3.3699830e+09, 3.5752281e+09, 3.7717654e+09,
        3.9656991e+09, 4.1554813e+09, 4.3364361e+09, 4.5091822e+09,
        4.6808358e+09, 4.8318975e+09, 4.9799485e+09, 5.1031480e+09,
        5.2148291e+09, 5.3077606e+09, 5.3797299e+09, 5.4427200e+09,
        5.4831839e+09, 5.5033043e+09, 5.5040472e+09, 5.4861486e+09,
        5.4415332e+09, 5.3876856e+09, 5.3109116e+09, 5.2073104e+09,
        5.1043394e+09, 4.9745251e+09, 4.8321390e+09, 4.6785772e+09,
        4.5126069e+09, 4.3374481e+09, 4.1570821e+09, 3.9674057e+09,
        3.7675909e+09, 3.5737653e+09, 3.3737757e+09, 3.1710123e+09,
        2.9716644e+09, 2.7770959e+09, 2.5856646e+09, 2.3959054e+09,
        2.2123010e+09, 2.0370410e+09, 1.8734512e+09, 1.7043299e+09,
        1.5525092e+09, 1.4141688e+09, 1.2767990e+09, 1.1494434e+09,
        1.0303885e+09, 9.2130870e+08, 8.2391520e+08, 7.3067090e+08,
        6.4480960e+08, 5.7031260e+08, 4.9965890e+08, 4.3810860e+08,
        3.8243940e+08, 3.3277780e+08, 2.8789100e+08, 2.4941660e+08,
        2.1313180e+08, 1.8287990e+08, 1.5609640e+08, 1.3302970e+08,
        1.1234810e+08, 9.4879600e+07, 8.0106700e+07, 6.6568900e+07])

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
        
        tracker = xt.Tracker(line=line)
        record = tracker.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5)})
        tracker.track(particles_b1, num_turns=1)
        tracker.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

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
        
        assert np.abs(upsilon_max_sim / upsilon_max - 1) < 1e-2
        assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2



def test_beambeam3d_collective_beamstrahlung_single_collision():
    for context in xo.context.get_test_contexts():

        if not isinstance(context, xo.ContextCpu):
            print(f'skipping test_beambeam3d_collective_beamstrahlung_single_collision for context {context}')
            continue
    
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
        n_bb = np.array([6.6884000e+07, 7.9522500e+07, 9.5608700e+07, 1.1205830e+08,
        1.3225230e+08, 1.5601820e+08, 1.8134120e+08, 2.1465900e+08,
        2.4764560e+08, 2.8871670e+08, 3.3381510e+08, 3.8250380e+08,
        4.3826040e+08, 5.0107570e+08, 5.6947540e+08, 6.4710960e+08,
        7.3120450e+08, 8.2246850e+08, 9.2371450e+08, 1.0346021e+09,
        1.1507452e+09, 1.2783837e+09, 1.4063856e+09, 1.5577785e+09,
        1.7084170e+09, 1.8692997e+09, 2.0361992e+09, 2.2154359e+09,
        2.3930373e+09, 2.5800457e+09, 2.7800560e+09, 2.9737367e+09,
        3.1711020e+09, 3.3699830e+09, 3.5752281e+09, 3.7717654e+09,
        3.9656991e+09, 4.1554813e+09, 4.3364361e+09, 4.5091822e+09,
        4.6808358e+09, 4.8318975e+09, 4.9799485e+09, 5.1031480e+09,
        5.2148291e+09, 5.3077606e+09, 5.3797299e+09, 5.4427200e+09,
        5.4831839e+09, 5.5033043e+09, 5.5040472e+09, 5.4861486e+09,
        5.4415332e+09, 5.3876856e+09, 5.3109116e+09, 5.2073104e+09,
        5.1043394e+09, 4.9745251e+09, 4.8321390e+09, 4.6785772e+09,
        4.5126069e+09, 4.3374481e+09, 4.1570821e+09, 3.9674057e+09,
        3.7675909e+09, 3.5737653e+09, 3.3737757e+09, 3.1710123e+09,
        2.9716644e+09, 2.7770959e+09, 2.5856646e+09, 2.3959054e+09,
        2.2123010e+09, 2.0370410e+09, 1.8734512e+09, 1.7043299e+09,
        1.5525092e+09, 1.4141688e+09, 1.2767990e+09, 1.1494434e+09,
        1.0303885e+09, 9.2130870e+08, 8.2391520e+08, 7.3067090e+08,
        6.4480960e+08, 5.7031260e+08, 4.9965890e+08, 4.3810860e+08,
        3.8243940e+08, 3.3277780e+08, 2.8789100e+08, 2.4941660e+08,
        2.1313180e+08, 1.8287990e+08, 1.5609640e+08, 1.3302970e+08,
        1.1234810e+08, 9.4879600e+07, 8.0106700e+07, 6.6568900e+07])
       
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
        
        tracker = xt.Tracker(line=line)
        record = tracker.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5)})
        tracker.track(particles_b1, num_turns=1)
        tracker.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

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
        
        assert np.abs(upsilon_max_sim / upsilon_max - 1) < 1e-2
        assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2


