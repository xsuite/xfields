import json

import numpy as np
import xtrack as xt
import xfields as xf


with open('../../test_data/hllhc14_for_bb_tests/line_b1.json', 'r') as fid:
    dct_b1 = json.load(fid)
with open('../../test_data/hllhc14_for_bb_tests/line_b4.json', 'r') as fid:
    dct_b4 = json.load(fid)
line_b1 = xt.Line.from_dict(dct_b1)
line_b4 = xt.Line.from_dict(dct_b4)

ip_names=['ip1', 'ip2', 'ip5', 'ip8']
num_long_range_elems_per_side=[25, 20, 25, 20]
harmonic_number=35640
bunch_spacing_buckets=10
num_slices_head_on=11
bunch_num_particles=2.2e11
sigmaz_m = 0.076
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
crab_strong_beam = True

circumference = line_b1.get_length()

bb_df_b1_ret, bb_df_b2_ret = xf.install_beambeam_elements_in_lines(
            line_b1, line_b4, ip_names,
            circumference, harmonic_number, bunch_spacing_buckets,
            num_long_range_elems_per_side, num_slices_head_on,
            bunch_num_particles, sigmaz_m)

tracker_b1 = line_b1.build_tracker()
tracker_b4 = line_b4.build_tracker()

keep_columns = ['beam', 'other_beam', 'ip_name', 'elementName', 'other_elementName', 'label',
                'self_num_particles', 'self_particle_charge', 'self_relativistic_beta',
                'identifier', 's_crab']
bb_df_b1 = bb_df_b1_ret[keep_columns].copy()
bb_df_b2 = bb_df_b2_ret[keep_columns].copy()

xf.configure_beam_beam_elements(bb_df_b1, bb_df_b2, tracker_b1, tracker_b4,
                                 nemitt_x, nemitt_y, crab_strong_beam, ip_names)

tracker, tracker_other, beam_name = (tracker_b1, tracker_b4, 'b1')
ip = 8

line = tracker.line
bb_ele_names = []
x_strong =[]
x_weak = []
y_strong =[]
y_weak = []

for nn in line.element_names:
    ee = line[nn]
    if isinstance(ee, (xf.BeamBeamBiGaussian2D, xf.BeamBeamBiGaussian3D)):
        if f'{ip}{beam_name}' in nn:
            bb_ele_names.append(nn)
            x_weak.append(ee.ref_shift_x)
            y_weak.append(ee.ref_shift_y)
            x_strong.append(ee.other_beam_shift_x + ee.ref_shift_x)
            y_strong.append(ee.other_beam_shift_y + ee.ref_shift_y)


import matplotlib.pyplot as plt

tw = tracker.twiss()
tw_other = tracker_other.twiss(reverse=True)
tw_df = tw.to_pandas()
tw_df.set_index('name', inplace=True)

beta0gamma0 = tw.particle_on_co.beta0[0] * tw.particle_on_co.gamma0[0]
s_bb = tw_df.loc[bb_ele_names, 's'].values
betx_bb = tw_df.loc[bb_ele_names, 'betx'].values
bety_bb = tw_df.loc[bb_ele_names, 'bety'].values

plt.close('all')
plt.figure(1, figsize=(6.4*1.4, 4.8*1.2))
spx = plt.subplot(2, 2, 1)
spx.plot(s_bb, x_weak, 'o-')
spx.plot(s_bb, x_strong, 'o-')
spx.plot(tw.s, tw.x, '-')
spx.plot(tw_other.s, tw_other.x, '-')

spy = plt.subplot(2, 2, 2, sharex=spx, sharey=spx)
spy.plot(s_bb, y_weak, 'o-')
spy.plot(s_bb, y_strong, 'o-')
spy.plot(tw.s, tw.y, '-')
spy.plot(tw_other.s, tw_other.y, '-')

spnx = plt.subplot(2, 2, 3, sharex=spx)
spnx.plot(s_bb, x_weak/np.sqrt(betx_bb*nemitt_x/beta0gamma0), 'o-')
spnx.plot(s_bb, x_strong/np.sqrt(betx_bb*nemitt_x/beta0gamma0), 'o-')

spny = plt.subplot(2, 2, 4, sharex=spx, sharey=spnx)
spny.plot(s_bb, y_weak/np.sqrt(bety_bb*nemitt_y/beta0gamma0), 'o-')
spny.plot(s_bb, y_strong/np.sqrt(bety_bb*nemitt_y/beta0gamma0), 'o-')

spx.set_xlim(s_bb[0], s_bb[-1])

plt.show()




