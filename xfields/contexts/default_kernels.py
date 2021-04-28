import sysconfig
from pathlib import Path

import numpy as np

import xobjects as xo

thisfolder = Path(__file__).parent.absolute()
pkg_root = thisfolder.parent.absolute()
so_suffix = sysconfig.get_config_var('EXT_SUFFIX')

kernel_descriptions = {
    # 'central_diff':{
    #     'args':(
    #         (('scalar', np.int32),   'nelem'),
    #         (('scalar', np.int32),   'stride_in_dbl'),
    #         (('scalar', np.float64), 'factor'),
    #         (('array',  np.float64), 'matrix'),
    #         (('array',  np.float64), 'res'),),
    #     'num_threads_from_arg': 'nelem'
    #     },
    'central_diff': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='nelem'),
            xo.Arg(xo.Int32,   pointer=False, name='row_size'),
            xo.Arg(xo.Int32,   pointer=False, name='stride_in_dbl'),
            xo.Arg(xo.Float64, pointer=False, name='factor'),
            xo.Arg(xo.Float64, pointer=True,  name='matrix'),
            xo.Arg(xo.Float64, pointer=True,  name='res'),
            ],
        n_threads='nelem'
        ),
    #'p2m_rectmesh3d':{
    #    'args':(
    #        (('scalar', np.int32),   'nparticles'),
    #        (('array',  np.float64), 'x'),
    #        (('array',  np.float64), 'y'),
    #        (('array',  np.float64), 'z'),
    #        (('array',  np.float64), 'part_weights'),
    #        (('scalar', np.float64), 'x0'),
    #        (('scalar', np.float64), 'y0'),
    #        (('scalar', np.float64), 'z0'),
    #        (('scalar', np.float64), 'dx'),
    #        (('scalar', np.float64), 'dy'),
    #        (('scalar', np.float64), 'dz'),
    #        (('scalar', np.int32),   'nx'),
    #        (('scalar', np.int32),   'ny'),
    #        (('scalar', np.int32),   'nz'),
    #        (('array',  np.float64), 'grid1d')),
    #    'num_threads_from_arg': 'nparticles'
    #    },
    'p2m_rectmesh3d': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='nparticles'),
            xo.Arg(xo.Float64, pointer=True, name='x'),
            xo.Arg(xo.Float64, pointer=True, name='y'),
            xo.Arg(xo.Float64, pointer=True, name='z'),
            xo.Arg(xo.Float64, pointer=True, name='part_weights'),
            xo.Arg(xo.Float64, pointer=False, name='x0'),
            xo.Arg(xo.Float64, pointer=False, name='y0'),
            xo.Arg(xo.Float64, pointer=False, name='z0'),
            xo.Arg(xo.Float64, pointer=False, name='dx'),
            xo.Arg(xo.Float64, pointer=False, name='dy'),
            xo.Arg(xo.Float64, pointer=False, name='dz'),
            xo.Arg(xo.Int32,   pointer=False, name='nx'),
            xo.Arg(xo.Int32,   pointer=False, name='ny'),
            xo.Arg(xo.Int32,   pointer=False, name='nz'),
            xo.Arg(xo.Float64, pointer=True, name='grid1d'),
            ],
        n_threads='nparticles'
        ),
    #'m2p_rectmesh3d':{
    #    'args':(
    #        (('scalar', np.int32),   'nparticles'),
    #        (('array',  np.float64), 'x'),
    #        (('array',  np.float64), 'y'),
    #        (('array',  np.float64), 'z'),
    #        (('scalar', np.float64), 'x0'),
    #        (('scalar', np.float64), 'y0'),
    #        (('scalar', np.float64), 'z0'),
    #        (('scalar', np.float64), 'dx'),
    #        (('scalar', np.float64), 'dy'),
    #        (('scalar', np.float64), 'dz'),
    #        (('scalar', np.int32),   'nx'),
    #        (('scalar', np.int32),   'ny'),
    #        (('scalar', np.int32),   'nz'),
    #        (('scalar', np.int32),   'n_quantities'),
    #        (('array',  np.int32), 'offsets_mesh_quantities'),
    #        (('array',  np.float64), 'mesh_quantity'),
    #        (('array',  np.float64), 'particles_quantity')),
    #    'num_threads_from_arg': 'nparticles'
    #    },
    'm2p_rectmesh3d': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='nparticles'),
            xo.Arg(xo.Float64, pointer=True,  name='x'),
            xo.Arg(xo.Float64, pointer=True,  name='y'),
            xo.Arg(xo.Float64, pointer=True,  name='z'),
            xo.Arg(xo.Float64, pointer=False, name='x0'),
            xo.Arg(xo.Float64, pointer=False, name='y0'),
            xo.Arg(xo.Float64, pointer=False, name='z0'),
            xo.Arg(xo.Float64, pointer=False, name='dx'),
            xo.Arg(xo.Float64, pointer=False, name='dy'),
            xo.Arg(xo.Float64, pointer=False, name='dz'),
            xo.Arg(xo.Int32,   pointer=False, name='nx'),
            xo.Arg(xo.Int32,   pointer=False, name='ny'),
            xo.Arg(xo.Int32,   pointer=False, name='nz'),
            xo.Arg(xo.Int32,   pointer=False, name='n_quantities'),
            xo.Arg(xo.Int32,   pointer=True,  name='offsets_mesh_quantities'),
            xo.Arg(xo.Float64, pointer=True,  name='mesh_quantity'),
            xo.Arg(xo.Float64, pointer=True,  name='particles_quantity'),
            ],
        n_threads='nparticles'
        ),
    # 'get_Ex_Ey_Gx_Gy_gauss':{
    #     'args':(
    #         (('scalar', np.int32  ), 'n_points'),
    #         (('array',  np.float64), 'x_ptr'),
    #         (('array',  np.float64), 'y_ptr'),
    #         (('scalar', np.float64), 'sigma_x'),
    #         (('scalar', np.float64), 'sigma_y'),
    #         (('scalar', np.float64), 'min_sigma_diff'),
    #         (('scalar', np.int32  ), 'skip_Gs'),
    #         (('array',  np.float64), 'Ex_ptr'),
    #         (('array',  np.float64), 'Ey_ptr'),
    #         (('array',  np.float64), 'Gx_ptr'),
    #         (('array',  np.float64), 'Gy_ptr')),
    #     'num_threads_from_arg': 'n_points'
    #     },
    'get_Ex_Ey_Gx_Gy_gauss': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='n_points'),
            xo.Arg(xo.Float64, pointer=True,  name='x_ptr'),
            xo.Arg(xo.Float64, pointer=True,  name='y_ptr'),
            xo.Arg(xo.Float64, pointer=False, name='sigma_x'),
            xo.Arg(xo.Float64, pointer=False, name='sigma_y'),
            xo.Arg(xo.Float64, pointer=False, name='min_sigma_diff'),
            xo.Arg(xo.Int32,   pointer=False, name='skip_Gs'),
            xo.Arg(xo.Float64, pointer=True,  name='Ex_ptr'),
            xo.Arg(xo.Float64, pointer=True,  name='Ey_ptr'),
            xo.Arg(xo.Float64, pointer=True,  name='Gx_ptr'),
            xo.Arg(xo.Float64, pointer=True,  name='Gy_ptr'),
            ],
        n_threads='n_points'
        ),
    #'q_gaussian_profile':{
    #    'args':(
    #    (('scalar', np.int32  ), 'n'),
    #    (('array',  np.float64), 'z'),
    #    (('scalar', np.float64), 'z0'),
    #    (('scalar', np.float64), 'z_min'),
    #    (('scalar', np.float64), 'z_max'),
    #    (('scalar', np.float64), 'beta'),
    #    (('scalar', np.float64), 'q'),
    #    (('scalar', np.float64), 'q_tol'),
    #    (('scalar', np.float64), 'factor'),
    #    (('array',  np.float64), 'res'),
    #        ),
    #    'num_threads_from_arg': 'n'
    #    },
    'q_gaussian_profile': xo.Kernel(
        args=[
            xo.Arg(xo.Int32  , pointer=False, name='n'),
            xo.Arg(xo.Float64, pointer=True,  name='z'),
            xo.Arg(xo.Float64, pointer=False, name='z0'),
            xo.Arg(xo.Float64, pointer=False, name='z_min'),
            xo.Arg(xo.Float64, pointer=False, name='z_max'),
            xo.Arg(xo.Float64, pointer=False, name='beta'),
            xo.Arg(xo.Float64, pointer=False, name='q'),
            xo.Arg(xo.Float64, pointer=False, name='q_tol'),
            xo.Arg(xo.Float64, pointer=False, name='factor'),
            xo.Arg(xo.Float64, pointer=True,  name='res'),
            ],
        n_threads='n'
        ),
    }

default_kernels = {
    'kernel_descriptions': kernel_descriptions,
    'src_files': [
        pkg_root.joinpath('src/linear_interpolators.h'),
        pkg_root.joinpath('src/complex_error_function.h'),
        pkg_root.joinpath('src/constants.h'),
        pkg_root.joinpath('src/qgaussian.h'),
        pkg_root.joinpath('src/fields_bigaussian.h'),
        pkg_root.joinpath('src/central_diff.h'),]
    }

