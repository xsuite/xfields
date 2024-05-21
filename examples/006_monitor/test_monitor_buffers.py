import xfields as xf
import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import numpy as np
from scipy.constants import e, c, physical_constants


#@for_all_test_contexts
#def test_bunch_buffer(test_context):
test_context = xo.ContextCpu(omp_num_threads=0)

n_macroparticles = int(1e6)
num_slices = 10
zeta_range = (-1, 1)

E0 = physical_constants['proton mass energy equivalent in MeV'][0]*1e6
E = 7000e9
gamma = E/E0
beta = np.sqrt(1-1/gamma**2)

offs_x = 1e-3
offs_px = 2e-3
offs_y = 3e-3
offs_py = 4e-3
offs_zeta = 5e-3
offs_delta = 6e-3

sigma_x = 1e-3
sigma_px = 2e-3
sigma_y = 3e-3
sigma_py = 4e-3
sigma_zeta = 5e-3
sigma_delta = 6e-3

monitor = xf.CollectiveMonitor(
    file_backend=None,
    monitor_bunches=True,
    monitor_slices=False,
    monitor_particles=False,
    n_steps=10,
    buffer_size=10,
    beta_gamma=beta*gamma,
    slicer_moments='all',
    zeta_range=zeta_range,  # These are [a, b] in the paper
    num_slices=num_slices,  # Per bunch, this is N_1 in the paper
    bunch_spacing_zeta=10,  # This is P in the paper
    num_slots=1
)

x_coord = sigma_x*np.random.random(n_macroparticles) + offs_x
px_coord = sigma_px*np.random.random(n_macroparticles) + offs_px
y_coord = sigma_y*np.random.random(n_macroparticles) + offs_y
py_coord = sigma_py*np.random.random(n_macroparticles) + offs_py
zeta_coord = sigma_zeta*np.random.random(n_macroparticles) + offs_zeta
delta_coord = sigma_delta*np.random.random(n_macroparticles) + offs_delta

particles = xt.Particles(
    _context=test_context, p0c=E,
    x=x_coord,
    px=px_coord,
    y=y_coord,
    py=py_coord,
    zeta=zeta_coord,
    delta=delta_coord
)

monitor.track(particles)

assert monitor.bunch_buffer[0]['mean_x'][0] == np.mean(x_coord)
assert monitor.bunch_buffer[0]['mean_px'][0] == np.mean(px_coord)
assert monitor.bunch_buffer[0]['mean_y'][0] == np.mean(y_coord)
assert monitor.bunch_buffer[0]['mean_py'][0] == np.mean(py_coord)
assert monitor.bunch_buffer[0]['mean_zeta'][0] == np.mean(zeta_coord)
assert monitor.bunch_buffer[0]['mean_delta'][0] == np.mean(delta_coord)

assert monitor.bunch_buffer[0]['sigma_x'][0] == np.std(x_coord)
assert monitor.bunch_buffer[0]['sigma_px'][0] == np.std(px_coord)
assert monitor.bunch_buffer[0]['sigma_y'][0] == np.std(y_coord)
assert monitor.bunch_buffer[0]['sigma_py'][0] == np.std(py_coord)
assert monitor.bunch_buffer[0]['sigma_zeta'][0] == np.std(zeta_coord)
assert monitor.bunch_buffer[0]['sigma_delta'][0] == np.std(delta_coord)

epsn_x = np.sqrt(np.linalg.det(np.cov(x_coord, px_coord))) * beta*gamma
epsn_y = np.sqrt(np.linalg.det(np.cov(y_coord, py_coord))) * beta*gamma
epsn_zeta = np.sqrt(np.linalg.det(np.cov(zeta_coord,
                                         delta_coord))) * beta*gamma

assert np.isclose(monitor.bunch_buffer[0]['epsn_x'][0], epsn_x)
assert np.isclose(monitor.bunch_buffer[0]['epsn_y'][0], epsn_y)
assert np.isclose(monitor.bunch_buffer[0]['epsn_zeta'][0], epsn_zeta)

assert monitor.bunch_buffer[0]['num_particles'][0] == n_macroparticles
'''


test_context = xo.ContextCpu(omp_num_threads=0)

n_macroparticles = int(1e7)
num_slices = 10
zeta_range = (0, 1)

E0 = physical_constants['proton mass energy equivalent in MeV'][0] * 1e6
E = 7000e9
gamma = E / E0
beta = np.sqrt(1 - 1 / gamma ** 2)

offs_x = 1
offs_px = 2
offs_y = 3
offs_py = 4
offs_delta = 5

sigma_x = 6
sigma_px = 7
sigma_y = 8
sigma_py = 9
sigma_delta = 10

monitor = xf.CollectiveMonitor(
    file_backend=None,
    monitor_bunches=False,
    monitor_slices=True,
    monitor_particles=False,
    n_steps=1,
    buffer_size=1,
    beta_gamma=beta * gamma,
    slicer_moments='all',
    zeta_range=zeta_range,  # These are [a, b] in the paper
    num_slices=num_slices,  # Per bunch, this is N_1 in the paper
    bunch_spacing_zeta=10,  # This is P in the paper
    num_slots=1
)

zeta_coord_tot = np.random.random(n_macroparticles)

particles = xt.Particles(
    _context=test_context,
    p0c=E,
    zeta=zeta_coord_tot
)

n_slice = 5

dzeta = monitor.slicer.dzeta

bin_min = monitor.slicer.zeta_centers[n_slice] - dzeta/2
bin_max = monitor.slicer.zeta_centers[n_slice] + dzeta/2

slice_mask = np.logical_and(zeta_coord_tot < bin_max, zeta_coord_tot >= bin_min)

n_macroparticles_slice = np.sum(slice_mask)

x_coord = sigma_x * np.random.random(n_macroparticles_slice) + offs_x
px_coord = sigma_px * np.random.random(n_macroparticles_slice) + offs_px
y_coord = sigma_y * np.random.random(n_macroparticles_slice) + offs_y
py_coord = sigma_py * np.random.random(n_macroparticles_slice) + offs_py
delta_coord = (sigma_delta * np.random.random(n_macroparticles_slice) +
               offs_delta)

particles.x[slice_mask] = x_coord
particles.px[slice_mask] = px_coord
particles.y[slice_mask] = y_coord
particles.py[slice_mask] = py_coord
particles.delta[slice_mask] = delta_coord

monitor.track(particles)


assert np.isclose(monitor.slice_buffer[0]['mean_x'][n_slice, 0],
                  np.mean(x_coord))
assert np.isclose(monitor.slice_buffer[0]['mean_px'][n_slice, 0],
                  np.mean(px_coord))
assert np.isclose(monitor.slice_buffer[0]['mean_y'][n_slice, 0],
                  np.mean(y_coord))
assert np.isclose(monitor.slice_buffer[0]['mean_py'][n_slice, 0],
                  np.mean(py_coord))
zeta_coord = particles.zeta[slice_mask]
assert np.isclose(monitor.slice_buffer[0]['mean_zeta'][n_slice, 0],
                  np.mean(zeta_coord))
assert np.isclose(monitor.slice_buffer[0]['mean_delta'][n_slice, 0],
                  np.mean(delta_coord))

assert np.isclose(monitor.slice_buffer[0]['sigma_x'][n_slice, 0],
                  np.std(x_coord))
assert np.isclose(monitor.slice_buffer[0]['sigma_px'][n_slice, 0],
                  np.std(px_coord))
assert np.isclose(monitor.slice_buffer[0]['sigma_y'][n_slice, 0],
                  np.std(y_coord))
assert np.isclose(monitor.slice_buffer[0]['sigma_py'][n_slice, 0],
                  np.std(py_coord))
assert np.isclose(monitor.slice_buffer[0]['sigma_zeta'][n_slice, 0],
                  np.std(zeta_coord))
assert np.isclose(monitor.slice_buffer[0]['sigma_delta'][n_slice, 0],
                  np.std(delta_coord))


epsn_x = (np.sqrt(np.var(x_coord) * np.var(px_coord) -
                  np.cov(x_coord, px_coord)[0, 1]) * beta * gamma)
epsn_y = (np.sqrt(np.var(y_coord) * np.var(py_coord) -
                  np.cov(y_coord, py_coord)[0, 1])*beta*gamma)
epsn_zeta = (np.sqrt(np.var(zeta_coord) * np.var(delta_coord) -
                     np.cov(zeta_coord, delta_coord)[0, 1]) * beta*gamma)


assert np.isclose(monitor.slice_buffer[0]['epsn_x'][n_slice, 0], epsn_x)
assert np.isclose(monitor.slice_buffer[0]['epsn_y'][n_slice, 0], epsn_y)
assert np.isclose(monitor.slice_buffer[0]['epsn_zeta'][n_slice, 0], epsn_zeta)

assert (monitor.slice_buffer[0]['num_particles'][n_slice, 0] ==
        n_macroparticles_slice)
'''