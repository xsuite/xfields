# ISSUES WITH THE PRESENT TREATMENT OF THE CRAB CAVITIES
#  - The present way of treating the crab cavities has several limitations and
#    is not valid when the crab bump introduces an effect on the angle of the
#    of the strong slice apart from the position (the effect of the crab is
#    applied only on x and y).
#  - The location of the HO bb lenses depends on the bunch length. This means
#    that the lattice needs to be regenerated when changing the bunch length.
#  - It is not clear to me how to handle the case in which the two colliding
#    bunches have different bunch lengths.
#  - In the present implementation of the beambeam3d lenses, the direction of
#    arrival of the strong beam in the reference system of the weak beam is
#    stored exclusively in the variables phi and alpha.
#  - Xavier and Peter for the strong-strong model introduced additional inputs for
#    for slices_other_beam_px_center_star and slices_other_beam_py_center_star, but
#    such variables are not used for now (see https://github.com/xsuite/xsuite/issues/268)

import json
import xtrack as xt
import xpart as xp
import xfields as xf

fname_b1 = 'lineb1.json'
fname_b4 = 'lineb4.json'

with open(fname_b1, 'r') as fid:
    line_b1 = xt.Line.from_dict(json.load(fid))
line_b1.particle_ref = xp.Particles(p0c=7e12, q0=1, mass0=xp.PROTON_MASS_EV)

with open(fname_b4, 'r') as fid:
    line_b4 = xt.Line.from_dict(json.load(fid))
line_b4.particle_ref = xp.Particles(p0c=7e12, q0=1, mass0=xp.PROTON_MASS_EV)

# Install 6d lenses at one IP (old approach)

ip_names=['ip1']
numberOfLRPerIRSide=[0]
harmonic_number=35640
bunch_spacing_buckets=10
numberOfHOSlices=11
bunch_num_particles=2.2e11
sigmaz_m = 0.076
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
crab_strong_beam = True

import pymask as pm
circumference = line_b1.get_length()

# TODO: use keyword arguments
# TODO: what happens if bunch length is different for the two beams
bb_df_b1 = pm.generate_set_of_bb_encounters_1beam(
    circumference, harmonic_number,
    bunch_spacing_buckets,
    numberOfHOSlices,
    bunch_num_particles, line_b1.particle_ref.q0,
    sigmaz_m, line_b1.particle_ref.beta0[0], ip_names, numberOfLRPerIRSide,
    beam_name = 'b1',
    other_beam_name = 'b2')


bb_df_b2 = pm.generate_set_of_bb_encounters_1beam(
    circumference, harmonic_number,
    bunch_spacing_buckets,
    numberOfHOSlices,
    bunch_num_particles, line_b4.particle_ref.q0,
    sigmaz_m,
    line_b4.particle_ref.beta0[0], ip_names, numberOfLRPerIRSide,
    beam_name = 'b2',
    other_beam_name = 'b1')
bb_df_b2['atPosition'] = -bb_df_b2['atPosition'] # I am installing in b4 not in b2

from temp_module import install_dummy_bb_lenses

install_dummy_bb_lenses(bb_df=bb_df_b1, line=line_b1)
install_dummy_bb_lenses(bb_df=bb_df_b2, line=line_b4)

tracker_b1 = line_b1.build_tracker()
tracker_b4 = line_b4.build_tracker()

line_b1.tracker = None # We do it like this for now (to be cleaned up if kept)
line_b4.tracker = None
tracker_b1_4d = xt.Tracker(line=line_b1)
tracker_b4_4d = xt.Tracker(line=line_b4)
tracker_b1_4d.freeze_longitudinal()
tracker_b4_4d.freeze_longitudinal()

twiss_b1 = tracker_b1.twiss()
twiss_b2 = tracker_b4.twiss(reverse=True)

survey_b1 = tracker_b1.survey()
survey_b2 = tracker_b4.survey(reverse=True)

sigmas_b1 = twiss_b1.get_betatron_sigmas(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
sigmas_b2 = twiss_b2.get_betatron_sigmas(nemitt_x=nemitt_x, nemitt_y=nemitt_y)

# Use survey and twiss to get geometry and locations of all encounters
pm.get_geometry_and_optics_b1_b2(
    mad=None,
    bb_df_b1=bb_df_b1,
    bb_df_b2=bb_df_b2,
    xsuite_line_b1=line_b1,
    xsuite_line_b2=line_b4,
    xsuite_twiss_b1=twiss_b1,
    xsuite_twiss_b2=twiss_b2,
    xsuite_survey_b1=survey_b1,
    xsuite_survey_b2=survey_b2,
    xsuite_sigmas_b1=sigmas_b1,
    xsuite_sigmas_b2=sigmas_b2,
)

# Get the position of the IPs in the surveys of the two beams
ip_position_df = pm.get_survey_ip_position_b1_b2(mad=None, ip_names=ip_names,
    xsuite_survey_b1=survey_b1, xsuite_survey_b2=survey_b2)

# Get geometry and optics at the partner encounter
pm.get_partner_corrected_position_and_optics(
        bb_df_b1, bb_df_b2, ip_position_df)

# Compute separation, crossing plane rotation, crossing angle and xma
for bb_df in [bb_df_b1, bb_df_b2]:
    pm.compute_separations(bb_df)
    pm.compute_dpx_dpy(bb_df)
    pm.compute_local_crossing_angle_and_plane(bb_df)
    pm.compute_xma_yma(bb_df)

# Get bb dataframe and mad model (with dummy bb) for beam 3 and 4
bb_df_b3 = pm.get_counter_rotating(bb_df_b1)
bb_df_b4 = pm.get_counter_rotating(bb_df_b2)


bb_dfs = {
    'b1': bb_df_b1,
    'b2': bb_df_b2,
    'b3': bb_df_b3,
    'b4': bb_df_b4}

if crab_strong_beam:
    pm.crabbing_strong_beam_xsuite(bb_dfs,
        tracker_b1, tracker_b4,
        tracker_b1_4d, tracker_b4_4d)
else:
    print('Crabbing of strong beam skipped!')

pm.setup_beam_beam_in_line(line_b1, bb_df_b1, bb_coupling=False)
pm.setup_beam_beam_in_line(line_b4, bb_df_b4, bb_coupling=False)




