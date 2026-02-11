import numpy as np
import xwakes as xw
import xtrack as xt


def wake_vs_t(t):
    # act as delta function
    t = np.atleast_1d(t)
    out = np.zeros_like(t)
    out[np.abs(t) < 1e-6] = 1
    return out

custom_component = xw.Component(
    wake=wake_vs_t,
    plane='x',
    source_exponents=(2, 0),
    test_exponents=(0, 0),
)

custom_wake = xw.Wake(components=[custom_component])
custom_wake.configure_for_tracking(zeta_range=(-1, 1), num_slices=201)

p = xt.Particles(zeta=np.linspace(-0.5, 0.5, 101),
                 x=0, weight=1e11)

custom_wake.track(p)