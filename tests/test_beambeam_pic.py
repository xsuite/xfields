import numpy as np

import xfields as xf
import xtrack as xt
import xobjects as xo

from xobjects.test_helpers import for_all_test_contexts

constant_charge_slicing_gaussian = \
    xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_beambeam_pic(test_context):
    # LHC-like parameter
    mass0 = xt.PROTON_MASS_EV
    p0c = 7e12
    phi = 200e-6
    alpha = np.deg2rad(30)
    betx = 0.15
    bety = 0.2
    sigma_z = 0.1 # used for grids
    sigma_z_b1 = 0.08
    sigma_z_b2 = 0.07
    z_offset_b1 = 0.02
    z_offset_b2 = -0.015
    nemitt_x_b1 = 2e-6
    nemitt_y_b1 = 2.5e-6
    nemitt_x_b2 = 1.5e-6
    nemitt_y_b2 = 1.3e-6
    bunch_intensity_b1 = 2e10
    bunch_intensity_b2 = 3e10
    num_slices = 101
    n_turns = 2
    slice_mode = 'constant_charge'

    lntwiss = xt.Line(elements=[xt.Marker()])
    lntwiss.particle_ref = xt.Particles(p0c=p0c, mass0=mass0, _context=test_context)
    twip = lntwiss.twiss(betx=betx, bety=bety)

    cov_b1 = twip.get_beam_covariance(nemitt_x=nemitt_x_b1, nemitt_y=nemitt_y_b1)
    cov_b2 = twip.get_beam_covariance(nemitt_x=nemitt_x_b2, nemitt_y=nemitt_y_b2)
    sigma_x_b1 = cov_b1.sigma_x[0]
    sigma_y_b1 = cov_b1.sigma_y[0]
    sigma_x_b2 = cov_b2.sigma_x[0]
    sigma_y_b2 = cov_b2.sigma_y[0]

    num_particles = 1_000_000
    bunch_b1 = lntwiss.build_particles(
        num_particles=num_particles,
        nemitt_x=nemitt_x_b1, nemitt_y=nemitt_y_b1,
        zeta=np.random.normal(size=num_particles) * sigma_z_b1 + z_offset_b1,
        x_norm=np.random.normal(size=num_particles),
        px_norm=np.random.normal(size=num_particles),
        y_norm=np.random.normal(size=num_particles),
        py_norm=np.random.normal(size=num_particles),
        W_matrix=twip.W_matrix[0],
        particle_on_co=twip.particle_on_co,
        weight = bunch_intensity_b1 / num_particles,
        _context=test_context,
    )

    bunch_b2 = lntwiss.build_particles(
        num_particles=num_particles,
        nemitt_x=nemitt_x_b2, nemitt_y=nemitt_y_b2,
        zeta=np.random.normal(size=num_particles) * sigma_z_b2 + z_offset_b2,
        x_norm=np.random.normal(size=num_particles),
        px_norm=np.random.normal(size=num_particles),
        y_norm=np.random.normal(size=num_particles),
        py_norm=np.random.normal(size=num_particles),
        W_matrix=twip.W_matrix[0],
        particle_on_co=twip.particle_on_co,
        weight = bunch_intensity_b2 / num_particles,
        _context=test_context,
    )

    n_test = 1000
    p_test_b1 = lntwiss.build_particles(x=1.2 * sigma_x_b1, y=-1.1 * sigma_y_b1,
                                    px=50e-6, py=100e-6,
                    zeta=np.linspace(-2 * sigma_z, 2 * sigma_z, n_test),
                    weight=0)
    p_test_b2 = lntwiss.build_particles(x=-1.4 * sigma_x_b1, y=1.2 * sigma_y_b1,
                                    px=30e-6, py=-80e-6,
                    zeta=np.linspace(-2 * sigma_z, 2 * sigma_z, n_test),
                    weight=0)

    particles_b1 = xt.Particles.merge([p_test_b1, bunch_b1])
    particles_b2 = xt.Particles.merge([p_test_b2, bunch_b2])

    particles_b1.move(_context=test_context)
    particles_b2.move(_context=test_context)

    x_lim_grid = phi * 3 * sigma_z + 5 * sigma_x_b1
    y_lim_grid = phi * 3 * sigma_z + 5 * sigma_y_b1

    pics = []
    for ii in range(4):
        pics.append(xf.BeamBeamPIC3D(
            phi={0: phi, 1: -phi, 2: -1.2*phi, 3: 1.2*phi}[ii],
            alpha={0: alpha, 1: -alpha, 2: 1.3*alpha, 3: -1.3*alpha}[ii],
            x_range=(-x_lim_grid, x_lim_grid), dx=0.2*sigma_x_b1,
            y_range=(-y_lim_grid, y_lim_grid), dy=0.2*sigma_y_b1,
            z_range=(-2.5*sigma_z, 2.5*sigma_z), dz=0.2*sigma_z,
            _context=test_context)
        )

    bbpic_ip1_b1 = pics[0]
    bbpic_ip1_b2 = pics[1]
    bbpic_ip2_b1 = pics[2]
    bbpic_ip2_b2 = pics[3]

    # Pipeline configuration (some rationalization needed here!)
    pipeline_manager = xt.PipelineManager()
    pipeline_manager.add_particles('p_b1', rank=0)
    pipeline_manager.add_particles('p_b2', rank=0)
    # Association of partners
    pipeline_manager.add_element('IP1') # needs to be the same for the two lines (I guess...)
    pipeline_manager.add_element('IP2') # needs to be the same for the two lines (I guess...)
    bbpic_ip1_b1.name = 'IP1'
    bbpic_ip1_b2.name = 'IP1'
    bbpic_ip2_b2.name = 'IP2'
    bbpic_ip2_b1.name = 'IP2'
    bbpic_ip1_b1.partner_name = 'p_b2'
    bbpic_ip1_b2.partner_name = 'p_b1'
    bbpic_ip2_b1.partner_name = 'p_b2'
    bbpic_ip2_b2.partner_name = 'p_b1'
    particles_b1.init_pipeline('p_b1')
    particles_b2.init_pipeline('p_b2')
    bbpic_ip1_b1.pipeline_manager = pipeline_manager
    bbpic_ip2_b2.pipeline_manager = pipeline_manager
    bbpic_ip1_b2.pipeline_manager = pipeline_manager
    bbpic_ip2_b1.pipeline_manager = pipeline_manager

    # Build lines and multitracker
    line_b1 = xt.Line(elements=[bbpic_ip1_b1, bbpic_ip2_b1])
    line_b2 = xt.Line(elements=[bbpic_ip1_b2, bbpic_ip2_b2])

    line_b1.build_tracker(_context=test_context)
    line_b2.build_tracker(_context=test_context)

    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line_b1, particles=particles_b1),
                xt.PipelineBranch(line=line_b2, particles=particles_b2)],
        enable_debug_log=True, verbose=True)

    # Tracker
    multitracker.track(num_turns=n_turns)

    # Compare against Hirata
    z_centroids, z_cuts, num_part_per_slice = constant_charge_slicing_gaussian(
                                    1., 1., num_slices)
    z_centroids_from_tail = z_centroids[::-1]
    common_hirata_kwargs_b1 = dict(
        other_beam_q0=1.,
        slices_other_beam_num_particles=num_part_per_slice * bunch_intensity_b2,
        slices_other_beam_zeta_center=z_centroids_from_tail * sigma_z_b2 + z_offset_b2,
        slices_other_beam_Sigma_11=cov_b2.Sigma11[0],
        slices_other_beam_Sigma_12=cov_b2.Sigma12[0],
        slices_other_beam_Sigma_22=cov_b2.Sigma22[0],
        slices_other_beam_Sigma_33=cov_b2.Sigma33[0],
        slices_other_beam_Sigma_34=cov_b2.Sigma34[0],
        slices_other_beam_Sigma_44=cov_b2.Sigma44[0])
    common_hirata_kwargs_b2 = dict(
        other_beam_q0=1.,
        slices_other_beam_num_particles=num_part_per_slice * bunch_intensity_b1,
        slices_other_beam_zeta_center=z_centroids_from_tail * sigma_z_b1 + z_offset_b1,
        slices_other_beam_Sigma_11=cov_b1.Sigma11[0],
        slices_other_beam_Sigma_12=cov_b1.Sigma12[0],
        slices_other_beam_Sigma_22=cov_b1.Sigma22[0],
        slices_other_beam_Sigma_33=cov_b1.Sigma33[0],
        slices_other_beam_Sigma_34=cov_b1.Sigma34[0],
        slices_other_beam_Sigma_44=cov_b1.Sigma44[0])

    bbg_b1_ip1 = xf.BeamBeamBiGaussian3D(phi=phi, alpha=alpha,
                                        **common_hirata_kwargs_b1)
    bbg_b1_ip2 = xf.BeamBeamBiGaussian3D(phi=-1.2*phi, alpha=1.3*alpha,
                                        **common_hirata_kwargs_b1)
    bbg_b2_ip1 = xf.BeamBeamBiGaussian3D(phi=-phi, alpha=-alpha,
                                        **common_hirata_kwargs_b2)
    bbg_b2_ip2 = xf.BeamBeamBiGaussian3D(phi=1.2*phi, alpha=-1.3*alpha,
                                        **common_hirata_kwargs_b2)

    p_bbg_b1 = p_test_b1.copy()
    p_bbg_b2 = p_test_b2.copy()

    for ii in range(n_turns):
        bbg_b1_ip1.track(p_bbg_b1)
        bbg_b1_ip2.track(p_bbg_b1)
        bbg_b2_ip1.track(p_bbg_b2)
        bbg_b2_ip2.track(p_bbg_b2)

    particles_b1.move(_context=xo.context_default)
    particles_b1.move(_context=xo.context_default)

    xo.assert_allclose(p_bbg_b1.px, particles_b1.px[:n_test], rtol=0, atol=3e-8)
    xo.assert_allclose(p_bbg_b1.py, particles_b1.py[:n_test], rtol=0, atol=5e-8)
    xo.assert_allclose(p_bbg_b1.ptau, particles_b1.ptau[:n_test], rtol=0, atol=3e-12)
    xo.assert_allclose(p_bbg_b2.px, particles_b2.px[:n_test], rtol=0, atol=3e-8)
    xo.assert_allclose(p_bbg_b2.py, particles_b2.py[:n_test], rtol=0, atol=5e-8)
    xo.assert_allclose(p_bbg_b2.ptau, particles_b2.ptau[:n_test], rtol=0, atol=1e-11)
