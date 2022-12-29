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
line_b4.particle_ref = xp.Particles(p0c=7e12, q0=-1, mass0=xp.PROTON_MASS_EV)


# Install head-on elements

ho_locations = ['ip1', 'ip2', 'ip5', 'ip8']
for ll in ho_locations:
    for lname, line in zip(('b1', 'b4'), (line_b1, line_b4)):
        line.insert_element(ll,
            name = f'bb_ho_at_'+ll,
            element=xf.BeamBeamBiGaussian3D(
                slices_other_beam_num_particles=[0],
                slices_other_beam_zeta_center=[0],
                slices_other_beam_Sigma_11=[1],
                slices_other_beam_Sigma_12=[0],
                slices_other_beam_Sigma_22=[1],
                slices_other_beam_Sigma_33=[1],
                slices_other_beam_Sigma_34=[0],
                slices_other_beam_Sigma_44=[1],
                phi=0, alpha=0, other_beam_q0=1
        ))






tracker_b1 = line_b1.build_tracker()
tracker_b4 = line_b4.build_tracker()



