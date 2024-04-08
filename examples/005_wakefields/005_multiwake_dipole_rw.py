import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer as PyHTUniformBinSlicer
from PyHEADTAIL.impedances.wakes import ResistiveWall as PyHTResistiveWall
from PyHEADTAIL.impedances.wakes import WakeTable as PyHTWakeTable

from PyHEADTAIL.impedances.wakes import WakeField as PyHTWakeField
from PyHEADTAIL.machines.synchrotron import Synchrotron as PyHTSynchrotron

from wakefield import Wakefield, TempTableFunction, MultiWakefield, TempRWFunction

import xtrack as xt

from scipy.stats import linregress
from scipy.signal import hilbert

# Machine settings
n_turns = 500*10**3
n_turns_wake = 1
flatten = False

n_macroparticles = 10**6  # per bunch
intensity = 6e11

alpha = 53.86**-2

p0 = 7000e9 * e / c
accQ_x = 60.31
accQ_y = 60.32
Q_s = 1.919e-3
chroma = -3

h_RF = 35640
bunch_spacing_buckets = 10
n_bunches = 1
n_slices = 100
beta_x = 70
beta_y = 70

h_bunch = h_RF
circumference = 26658.883 / 35640 * h_RF

machine = PyHTSynchrotron(
        optics_mode='smooth', n_segments=1, circumference=circumference,
        accQ_x=accQ_x, accQ_y=accQ_y, beta_x=beta_x, beta_y=beta_y,
        D_x=0, D_y=0, alpha_mom_compaction=alpha, longitudinal_mode='linear',
        h_RF=np.atleast_1d(h_RF), p0=p0,
        charge=e, mass=m_p, wrap_z=False, Q_s=Q_s,
        Qp_x=chroma, Qp_y=chroma
)
transverse_map = machine.transverse_map.segment_maps[0]

# Filling scheme
filling_scheme = [i*bunch_spacing_buckets for i in range(n_bunches)]
bucket_length = machine.circumference / h_RF

epsn_x = 2e-6
epsn_y = 2e-6
sigma_z = 0.07494811

# Initialise beam
bunches = machine.generate_6D_Gaussian_bunch_matched(
                                             n_macroparticles, intensity,
                                             epsn_x, epsn_y, sigma_z=sigma_z,
                                             filling_scheme=filling_scheme)

bucket_id_set = list(set(bunches.bucket_id))
bucket_id_set.sort()

#zeta_range = (-0.48*bucket_length, 0.48*bucket_length)
zeta_range = (-8*sigma_z, 8*sigma_z)

slicer = PyHTUniformBinSlicer(n_slices, z_cuts=zeta_range,
                              circumference=machine.circumference,
                              h_bunch=h_bunch)

z_all = -bunches.bucket_id * bucket_length + bunches.z


# Initialise wakes
conductivity = 1. / 1.7e-8
pipe_radius = 10e-3
resistive_wall_length = circumference
dt_min = 0
Yokoya_X1 = 1
Yokoya_Y1 = 1
Yokoya_X2 = 0
Yokoya_Y2 = 0

gamma = bunches.gamma
momentumCompaction = alpha
eta = momentumCompaction - 1.0 / gamma ** 2
averageRadius = circumference / (2 * np.pi)
sigma_delta = Q_s * sigma_z / (averageRadius * eta)
beta_s = sigma_z / sigma_delta

n_bunches_wake = 120  # Can be longer than filling scheme

beta = np.sqrt(1-1/gamma**2)

rw_func = TempRWFunction(pipe_radius=pipe_radius,
                         resistive_wall_length=resistive_wall_length,
                         conductivity=conductivity, beta=beta,
                         yokoya_factor=Yokoya_X1, dt_min=dt_min)

#table = np.loadtxt('/Users/lorenzogiacomel/example_with_wake_room_T/WxdipWLHC_1layers10.00mm_precise.dat', skiprows=1, dtype=float)
#table[:, 0] /= c
#table[:, 1] *= circumference*Yokoya_X1
#rw_func = TempTableFunction(table=table, beta=beta)


#wakes = PyHTResistiveWall(pipe_radius=pipe_radius,
#                          resistive_wall_length=resistive_wall_length,
#                          conductivity=conductivity,
#                         dt_min=dt_min, Yokoya_X1=Yokoya_X1,
#                          Yokoya_Y1=Yokoya_Y1, Yokoya_X2=Yokoya_X2,
#                          Yokoya_Y2=Yokoya_Y2,
#                          n_turns_wake=n_turns_wake)

#wakes = PyHTWakeTable(wake_file='/Users/lorenzogiacomel/example_with_wake/table_pyht.dat', wake_file_columns=['time', 'dipole_x', 'dipole_y'])

#mpi_settings = False
# mpi_settings = 'memory_optimized'
# mpi_settings = 'linear_mpi_full_ring_fft'
#wake_field = PyHTWakeField(slicer, wakes, mpi=mpi_settings)
#machine.one_turn_map.append(wake_field)

wfx = Wakefield(
    source_moments=['num_particles', 'x'],
    kick='px',
    scale_kick=None,
    function=rw_func
)

wfy = Wakefield(
    source_moments=['num_particles', 'y'],
    kick='py',
    scale_kick=None,
    function=rw_func
)

zeta_range_xf = zeta_range

wf = MultiWakefield(
    wakefields=[wfx, wfy],
    zeta_range=zeta_range_xf,
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
    num_bunches=n_bunches_wake,
    num_turns=n_turns_wake,
    circumference=circumference,
    log_moments=['px'],
    _flatten=flatten
)

betatron_map = xt.LineSegmentMap(
    length=circumference, betx=beta_x, bety=beta_y,
    qx=accQ_x, qy=accQ_y, qs=Q_s, bets=beta_s,
    longitudinal_mode='linear_fixed_qs',
    dqx=chroma, dqy=chroma
)


line = xt.Line(elements=[wf, betatron_map],
               element_names=['wf', 'betatron_map'])

# apply a distortion to the bunch
amplitude = 1e-3
#wavelength = 1
#bunches.x *= 0
bunches.x += amplitude #* np.cos(2 * np.pi * z_all / wavelength)
bunches.y += amplitude #* np.cos(2 * np.pi * z_all / wavelength)

#plt.figure(74)
#plt.plot(z_all, bunches.x, 'x')
#plt.show()

particles = xt.Particles(
    mass0=xt.PROTON_MASS_EV,
    gamma0=bunches.gamma,
    x=bunches.x.copy(),
    px=bunches.xp.copy(),
    y=bunches.y.copy(),
    py=bunches.yp.copy(),
    zeta=z_all,
    delta=bunches.dp.copy(),
    weight=bunches.particlenumber_per_mp,
)

particle_ref = xt.Particles(p0c=particles.p0c, mass0=particles.mass0,
                            q0=particles.q0)
line.build_tracker()

n_skip = 1

flag_plot = True

if flag_plot:
    plt.close('all')

    fig0, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2, figsize=(10, 10))
    ax01.sharex(ax00)
    ax10.sharex(ax00)
    ax11.sharex(ax00)

mean_x_ht = np.zeros(n_turns)
mean_x_xt = np.zeros(n_turns)
mean_y_ht = np.zeros(n_turns)
mean_y_xt = np.zeros(n_turns)

plt.ion()

fig1 = plt.figure()
ax = fig1.add_subplot(111)
line1, = ax.plot(mean_x_xt, 'r-')
line2, = ax.plot(mean_x_xt, 'm-')
line3, = ax.plot(mean_x_ht, 'b-')
line4, = ax.plot(mean_x_ht, 'g-')

turns = np.linspace(0, n_turns - 1, n_turns)

for i_turn in range(n_turns):
    line.track(particles, num_turns=1)

    #machine.track(bunches)

    mean_x_ht[i_turn] = np.mean(bunches.x)
    mean_x_xt[i_turn] = np.mean(particles.x)
    mean_y_ht[i_turn] = np.mean(bunches.y)
    mean_y_xt[i_turn] = np.mean(particles.y)

    if i_turn % 50 == 0:
        print(f'Turn: {i_turn}')

    if i_turn % 50 == 0 and i_turn > 1:
        i_fit_start = int(i_turn * 0.9)
        i_fit_end = i_turn

        # compute x instability growth rate
        ampls_x_xt = np.abs(hilbert(mean_x_xt))
        fit_x_xt = linregress(turns[i_fit_start: i_fit_end],
                              np.log(ampls_x_xt[i_fit_start: i_fit_end]))
        ampls_x_ht = np.abs(hilbert(mean_x_ht))
        fit_x_ht = linregress(turns[i_fit_start: i_fit_end],
                              np.log(ampls_x_ht[i_fit_start: i_fit_end]))

        # compute y instability growth rate

        ampls_y_xt = np.abs(hilbert(mean_y_xt))
        fit_y_xt = linregress(turns[i_fit_start: i_fit_end],
                              np.log(ampls_y_xt[i_fit_start: i_fit_end]))

        #print(np.log10(np.abs(mean_x_xt[:i_turn])))
        line1.set_xdata(turns[:i_turn])
        line1.set_ydata(np.log10(np.abs(mean_x_xt[:i_turn])))
        line2.set_xdata(turns[:i_turn])
        line2.set_ydata(np.log10(np.exp(fit_x_xt.intercept +
                                 fit_x_xt.slope*turns[:i_turn])))
        #line3.set_xdata(turns[:i_turn])
        #line3.set_ydata(np.log10(np.abs(mean_x_ht[:i_turn])))
        #line4.set_xdata(turns[:i_turn])
        #line4.set_ydata(np.log10(np.exp(fit_x_ht.intercept +
        #                         fit_x_ht.slope*turns[:i_turn])))
        print(f'xtrack GR: {fit_x_xt.slope}')
        #print(f'HT GR: {fit_x_ht.slope}')
        ax.set_ylim(np.min(np.log10(np.abs(mean_x_xt[:i_turn]))), np.max(np.log10(np.abs(mean_x_xt[:i_turn])))) #,
        #                                                                                 np.log10(np.abs(mean_x_ht[:i_turn]))))))

        ax.set_xlim(-i_turn*0.1, i_turn*1.1)

        fig1.canvas.draw()
        fig1.canvas.flush_events()

    if flag_plot:
        if i_turn % 50 != 0 or i_turn < 1:
            continue
        #ax00.clear()
        ax01.clear()
        ax10.clear()
        ax11.clear()

        slice_set = bunches.get_slices(slicer)
        wfx.slicer.track(particles)

        ax00.plot(slice_set.z_centers, slicer._mean_x(slice_set, bunches) - np.mean(bunches.x), 'x',
                  color='g', label='ht')
        #ax00.plot(wfx.slicer.zeta_centers[0],  wfx.slicer.mean('x')[0], 'b',
        #          color='b', label='xt')
        ax00.plot(wfx.slicer.zeta_centers[0][wfx.slicer.mean('x')[0] != 0],
                  wfx.slicer.mean('x')[0][wfx.slicer.mean('x')[0] != 0] - np.mean(wfx.slicer.mean('x')[0][wfx.slicer.mean('x')[0] != 0]),'b',
                  color='b', label='xt')
        ax00.set_ylabel('x before wake')
        #ax00.legend()

        ax01.plot(slice_set.z_centers, slicer._mean_xp(slice_set, bunches), 'x',
                  color='g', label='ht')
        ax01.plot(wfx.slicer.zeta_centers[0], wfx.slicer.mean('px')[0], '.',
                  color='b', label='xt')
        ax01.set_ylabel('px before wake')
        ax01.legend()

        '''
        ax10.plot(bunches.z, bunches.x, 'x', color='b', label='ht')
        ax10.plot(particles.zeta, particles.x, '.', color='r', label='xt')
        ax10.set_ylabel('x after map')
        ax10.legend()

        ax11.plot(bunches.z, bunches.xp, 'x', color='b', label='ht')
        ax11.plot(particles.zeta, particles.px, '.', color='r', label='xt')
        ax11.set_ylabel('px after map')
        ax11.legend()
        '''
        plt.suptitle(f'turn {i_turn}')
        plt.pause(0.5)

turns = np.linspace(0, n_turns-1, n_turns)
i_fit_start = int(i_turn*0.9)
i_fit_end = i_turn

# compute x instability growth rate

ampls_x_xt = np.abs(hilbert(mean_x_xt))
fit_x_xt = linregress(turns[i_fit_start: i_fit_end],
                      np.log(ampls_x_xt[i_fit_start: i_fit_end]))
ampls_x_ht = np.abs(hilbert(mean_x_ht))
fit_x_ht = linregress(turns[i_fit_start: i_fit_end],
                      np.log(ampls_x_ht[i_fit_start: i_fit_end]))

# compute y instability growth rate

ampls_y_xt = np.abs(hilbert(mean_y_xt))
fit_y_xt = linregress(turns[i_fit_start: i_fit_end],
                      np.log(ampls_y_xt[i_fit_start: i_fit_end]))
ampls_y_ht = np.abs(hilbert(mean_y_ht))
fit_y_ht = linregress(turns[i_fit_start: i_fit_end],
                      np.log(ampls_y_ht[i_fit_start: i_fit_end]))

plt.figure(123)
plt.semilogy(np.abs(mean_x_ht), 'r', label='PyHT')
plt.semilogy(np.abs(mean_x_xt), 'b', label='xtrack-xfields')
plt.semilogy(turns, np.exp(fit_x_xt.intercept + fit_x_xt.slope*turns), 'm',
             label=f'xtrack GR: {fit_x_xt.slope}')
plt.semilogy(turns, np.exp(fit_x_ht.intercept + fit_x_ht.slope*turns), 'y',
             label=f'PyHT GR: {fit_x_ht.slope}')
plt.xlabel('turn')
plt.ylabel('abs(mean x)')
plt.legend()
plt.title('PyHEADTAIL vs xtrack - resonator wake')
plt.show()

plt.figure(124)
plt.semilogy(np.abs(mean_y_ht), 'r', label='PyHT')
plt.semilogy(np.abs(mean_y_xt), 'b', label='xtrack-xfields')
plt.semilogy(turns, np.exp(fit_y_xt.intercept + fit_y_xt.slope*turns), 'm',
             label=f'xtrack GR: {fit_y_xt.slope}')
plt.semilogy(turns, np.exp(fit_y_ht.intercept + fit_y_ht.slope*turns), 'y',
             label=f'PyHT GR: {fit_y_ht.slope}')
plt.xlabel('turn')
plt.ylabel('abs(mean y)')
plt.legend()
plt.title('PyHEADTAIL vs xtrack - resonator wake')
plt.show()

assert np.isclose(fit_x_xt.slope, fit_x_ht.slope, rtol=1e-2)
assert np.isclose(fit_y_xt.slope, fit_y_ht.slope, rtol=1e-2)
