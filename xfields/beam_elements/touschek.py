# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import numpy as np

from ..general import _pkg_root

class TouschekScattering(xt.BeamElement):

    _xofields = {
        '_p0c': xo.Float64,
        '_bunch_population': xo.Float64,
        '_gemitt_x': xo.Float64,
        '_gemitt_y': xo.Float64,
        '_alfx': xo.Float64,
        '_betx': xo.Float64,
        '_alfy': xo.Float64,
        '_bety': xo.Float64,
        '_dx': xo.Float64,
        '_dpx': xo.Float64,
        '_dy': xo.Float64,
        '_dpy': xo.Float64,
        '_deltaN': xo.Float64,
        '_deltaP': xo.Float64,
        '_sigma_z': xo.Float64,
        '_sigma_delta': xo.Float64,
        '_n_simulated': xo.Int64,
        '_nx': xo.Float64,
        '_ny': xo.Float64,
        '_nz': xo.Float64,
        '_theta_min': xo.Float64,
        '_theta_max': xo.Float64,
        '_ignored_portion': xo.Float64,
        '_integrated_piwinski_rate': xo.Float64,
        '_seed': xo.Int64,
        '_inhibit_permute': xo.Int64
    }

    # allow_track = False
    _depends_on = [xt.RandomUniformAccurate]

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/elegant_rng.h'),
        _pkg_root.joinpath('beam_elements/touschek_src/touschek.h'),
    ]

    _per_particle_kernels = {
        '_scatter': xo.Kernel(
            c_name='TouschekScatter',
            args=[
                xo.Arg(xo.Float64, name='x_out', pointer=True),
                xo.Arg(xo.Float64, name='px_out', pointer=True),
                xo.Arg(xo.Float64, name='y_out', pointer=True),
                xo.Arg(xo.Float64, name='py_out', pointer=True),
                xo.Arg(xo.Float64, name='zeta_out', pointer=True),
                xo.Arg(xo.Float64, name='delta_out', pointer=True),
                xo.Arg(xo.Float64, name='theta_out', pointer=True),
                xo.Arg(xo.Float64, name='weight_out', pointer=True),
                xo.Arg(xo.Float64, name='totalMCRate_out', pointer=True),
                xo.Arg(xo.Int64,   name='n_selected_out', pointer=True),
            ],
        ),
    }

    def __init__(self, s=0.0,
                particle_ref=xt.Particles(),
                bunch_population=0.0,
                alfx=0.0, betx=0.0, alfy=0.0, bety=0.0,
                dx=0.0, dpx=0.0, dy=0.0, dpy=0.0,
                deltaN=0.0, deltaP=0.0,
                gemitt_x=0.0, gemitt_y=0.0,
                sigma_z=0.0, sigma_delta=0.0,
                n_simulated=0, nx=0.0, ny=0.0, nz=0.0,
                theta_min=0.0, theta_max=0.0,
                piwinski_rate=0.0,
                ignored_portion=0.0,
                integrated_piwinski_rate=0.0,
                seed=1997,
                inhibit_permute=0,
                **kwargs):
        
        # This gives AttributeError: 'TouschekScattering' object has no attribute '_xobject'
        # if not isinstance(self._context, xo.ContextCpu) or self._context.openmp_enabled:
        #     raise ValueError('TouschekScattering only enabled on CPU.')

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return
        
        super().__init__(**kwargs)

        self._s = s
        self._particle_ref = particle_ref
        self._bunch_population = bunch_population
        self._alfx = alfx
        self._betx = betx
        self._alfy = alfy
        self._bety = bety
        self._dx = dx
        self._dpx = dpx
        self._dy = dy
        self._dpy = dpy
        self._deltaN = deltaN
        self._deltaP = deltaP
        self._gemitt_x = gemitt_x
        self._gemitt_y = gemitt_y
        self._sigma_z = sigma_z
        self._sigma_delta = sigma_delta
        self._n_simulated = n_simulated
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._theta_min = theta_min
        self._theta_max = theta_max
        self._ignored_portion = ignored_portion
        self._integrated_piwinski_rate = integrated_piwinski_rate
        self.piwinski_rate = piwinski_rate
        self._seed = seed
        self._inhibit_permute = inhibit_permute

    def _configure(self, **kwargs):
        config_allowed = {
            "_s", "_particle_ref", "_bunch_population",
            "_gemitt_x", "_gemitt_y",
            "_alfx", "_betx", "_alfy", "_bety",
            "_dx", "_dpx", "_dy", "_dpy",
            "_deltaN", "_deltaP",
            "_sigma_z", "_sigma_delta",
            "_n_simulated", "_nx", "_ny", "_nz",
            "_theta_min", "_theta_max",
            "_ignored_portion", "piwinski_rate",
            "_integrated_piwinski_rate",
            "_seed", "_inhibit_permute"
        }

        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported configure() keys: {bad}")
        
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)
            if kk == "_particle_ref":
                self._p0c = self._particle_ref.p0c[0]

    def scatter(self):
        context = self._context
        particles = xt.Particles(_context=context)

        if not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        x_out      = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        px_out     = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        y_out      = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        py_out     = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        zeta_out   = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        delta_out  = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        theta_out  = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        weight_out = context.zeros(shape=(self._n_simulated,), dtype=np.float64)
        totalMCRate_out = context.zeros(shape=(1,), dtype=np.float64)
        n_selected_out  = context.zeros(shape=(1,), dtype=np.int64)

        self._scatter(particles=particles,
                      x_out=x_out, px_out=px_out,
                      y_out=y_out, py_out=py_out,
                      zeta_out=zeta_out, delta_out=delta_out,
                      theta_out=theta_out,
                      weight_out=weight_out,
                      totalMCRate_out=totalMCRate_out,
                      n_selected_out=n_selected_out)
        
        n = n_selected_out[0]
        # Create particle object for tracking
        # TODO: add at_element, start_tracking_at_element, ...
        part = xt.Particles(_capacity=2*n, 
                            p0c=self._p0c,
                            mass0=self._particle_ref.mass0,
                            q0=self._particle_ref.q0, 
                            pdg_id=self._particle_ref.pdg_id,
                            x=x_out[:n], px=px_out[:n],
                            y=y_out[:n], py=py_out[:n],
                            zeta=zeta_out[:n], delta=delta_out[:n],
                            weight=weight_out[:n],
                            s=getattr(self, '_s', 0.0))
        
        part_ids = part.filter(part.state == 1).particle_id
        self.theta_log = dict(zip(part_ids.astype(int), theta_out[:n].astype(float)))

        self.total_mc_rate = totalMCRate_out[0]
        self.ignored_rate = self._ignored_portion * self.total_mc_rate

        return part

    def track(self, particles):
        super().track(particles)