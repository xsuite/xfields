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

    n_turns = 1
    
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
    
    nx = 64
    ny = 64
    nz = 64
    x_lim = 6
    y_lim = 6
    z_lim = 6
    
    x_lim_grid = x_lim * sigma_x
    y_lim_grid = y_lim * sigma_y
    z_lim_grid = z_lim * sigma_z_tot
    
    dx = 2*x_lim_grid/(nx)
    dy = 2*y_lim_grid/(ny)
    dz = 2*z_lim_grid/(nz)
    
    bbpic_ip1_b1 = xf.BeamBeamPIC3D(
        _context=test_context,
            phi=phi, alpha=0,
            x_range=(-x_lim_grid, x_lim_grid), dx=dx,
            y_range=(-y_lim_grid, y_lim_grid), dy=dy,
            z_range=(-z_lim_grid, z_lim_grid), dz=dz,
            flag_luminosity=1
            )
    
    bbpic_ip1_b2 = xf.BeamBeamPIC3D(
        _context=test_context,
            phi=-phi, alpha=0,
            x_range=(-x_lim_grid, x_lim_grid), dx=dx,
            y_range=(-y_lim_grid, y_lim_grid), dy=dy,
            z_range=(-z_lim_grid, z_lim_grid), dz=dz,
            flag_luminosity=1
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
        )
    
    ##########################
    # configure record table #
    ##########################
    
    record_b1_pic = line_b1.start_internal_logging_for_elements_of_type(
        xf.BeamBeamPIC3D, capacity={"beamstrahlungtable": int(0), "lumitable": int(n_turns)})
    record_b2_pic = line_b2.start_internal_logging_for_elements_of_type(
        xf.BeamBeamPIC3D, capacity={"beamstrahlungtable": int(0), "lumitable": int(n_turns)})

    #####################
    # track 1 collision #
    #####################
    
    multitracker.track(num_turns=1)
    
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamPIC3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamPIC3D)

    record_b1_pic.move(_context=xo.context_default)
    record_b2_pic.move(_context=xo.context_default)

    ###########################################
    # test #1: compare PIC to analytical lumi #
    ###########################################

    # lumi [m^-2]
    piwi    = sigma_z_tot / sigma_x * phi  # [1] 
    lumi_ip = bunch_intensity**2 / (4*np.pi*sigma_x*np.sqrt(1 + piwi**2)*sigma_y)  # [m^-2] lumi per bunch crossing
    numlumi_b1 = record_b1_pic.lumitable.luminosity[0]
    numlumi_b2 = record_b2_pic.lumitable.luminosity[0]
    relabserr_b1 = 100*np.abs(numlumi_b1 - lumi_ip) / lumi_ip
    relabserr_b2 = 100*np.abs(numlumi_b2 - lumi_ip) / lumi_ip
    print(f"lumi formula: {lumi_ip:.4e}, numlumi beam 1: {numlumi_b1:.4e}, numlumi beam 2: {numlumi_b2:.4e}, error beam 1: {relabserr_b1:.4f} [%], error beam 2: {relabserr_b2:.4f} [%]")

    assert relabserr_b1 < 1e1, "beam 1 numerical lumi does not match formula within 10%"
    assert relabserr_b2 < 1e1, "beam 2 numerical lumi does not match formula within 10%"
