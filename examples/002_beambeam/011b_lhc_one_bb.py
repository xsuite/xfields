import numpy as np
import xtrack as xt

collider = xt.Multiline.from_json(
    '../../../xtrack/test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.lhcb1.twiss_default.clear()
collider.lhcb2.twiss_default.clear()

collider.vars['vrf400'] = 16

collider.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    sigmaz=0.1,
    ip_names=['ip1'],
    delay_at_ips_slots=[0, 0],
    num_long_range_encounters_per_side=[0, 0],
    num_slices_head_on=1)

collider.vars['on_x1hl'] = 200
collider.vars['on_x1vl'] = 100


tw = collider.twiss()

collider.configure_beambeam_interactions(crab_strong_beam=False,
                num_particles=2.3e11, nemitt_x=2e-6, nemitt_y=2e-6)

fields_to_display = [
 'phi',
 'alpha',
 'other_beam_shift_x',
 'other_beam_shift_y',
 'post_subtract_px',
 'post_subtract_py',
 'post_subtract_pzeta',
 'post_subtract_x',
 'post_subtract_y',
 'ref_shift_px',
 'ref_shift_py',
 'ref_shift_pzeta',
 'ref_shift_x',
 'ref_shift_y',
 'ref_shift_zeta',
]

values_b1 = []
values_b2 = []
for field in fields_to_display:
    values_b1.append(getattr(collider.lhcb1['bb_ho.c1b1_00'], field))
    values_b2.append(getattr(collider.lhcb2['bb_ho.c1b2_00'], field))

values_b1 = np.array(values_b1)
values_b2 = np.array(values_b2)

values_b1[np.abs(values_b1) < 6e-6] = 0
values_b2[np.abs(values_b2) < 6e-6] = 0

bb_config = xt.Table(
    data={'name': np.array(fields_to_display),
            'b1': np.array(values_b1),
            'b2': np.array(values_b2)})
bb_config.show()