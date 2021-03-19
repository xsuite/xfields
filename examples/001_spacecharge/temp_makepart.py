import numpy as np

from pysixtrack.particles import Particles

from xfields.contexts import XfCpuContext
from xfields.contexts import XfCupyContext
from xfields.contexts import XfPyopenclContext

class CupyMathlib(object):

    from cupy import sqrt, exp, sin, cos, abs, pi, tan

    @classmethod
    def wfun(cls, z_re, z_im):
        raise NotImplementedError

class PyopenclMathlib(object):

    from pyopencl.clmath import exp, sin, cos, tan
    from pyopencl import clmath
    from numpy import pi

    @classmethod
    def wfun(cls, z_re, z_im):
        raise NotImplementedError

    @classmethod
    def sqrt(cls, a):
        if np.isscalar(a):
            return np.sqrt(a)
        else:
            return cls.clmath.sqrt(a)

def generate_particles_object(context,
                            n_macroparticles,
                            bunch_intensity,
                            sigma_x,
                            sigma_y,
                            sigma_z,
                            p0c,
                            mass,
                            n_probes,
                            r_max_probes,
                            z_probes,
                            theta_probes):

    x_part = sigma_x * np.random.normal(size=(n_macroparticles,))
    y_part = sigma_y * np.random.normal(size=(n_macroparticles,))
    z_part = sigma_z * np.random.normal(size=(n_macroparticles,))
    weights_part = 0*x_part + bunch_intensity/n_macroparticles

    # insert probes
    r_probes= np.linspace(-r_max_probes, r_max_probes, n_probes)
    x_probes = r_probes * np.cos(theta_probes)
    y_probes = r_probes * np.sin(theta_probes)
    z_probes = 0 * x_probes +z_probes

    x_part = np.concatenate([x_probes, x_part])
    y_part = np.concatenate([y_probes, y_part])
    z_part = np.concatenate([z_probes, z_part])
    weights_part = np.concatenate([0*x_probes, weights_part])

    px_part = 0*x_part
    py_part = 0*x_part
    pt_part = 0*x_part

    # Move to context
    np2platf = context.nparray_to_context_array
    x_part_dev = np2platf(x_part)
    y_part_dev = np2platf(y_part)
    z_part_dev = np2platf(z_part)
    px_part_dev = np2platf(px_part)
    py_part_dev = np2platf(py_part)
    ptau_part_dev = np2platf(pt_part)
    weights_part_dev = np2platf(weights_part)

    if isinstance(context, XfCupyContext):
        kwargs = {'mathlib': CupyMathlib()}
    elif isinstance(context, XfPyopenclContext):
        kwargs = {'mathlib': PyopenclMathlib()}
    elif isinstance(context, XfCpuContext):
        kwargs = {}
    else:
        raise ValueError('Unknown context!')

    particles = Particles(
            p0c=p0c,
            mass = mass,
            x=x_part_dev,
            y=y_part_dev,
            zeta=z_part_dev,
            px=px_part_dev,
            py=py_part_dev,
            ptau=ptau_part_dev,
            **kwargs)
    particles.weight = weights_part_dev

    return particles, r_probes, x_probes, y_probes, z_probes

