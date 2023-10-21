import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../../xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
line.particle_ref = xt.Particles(p0c=26e9, mass0=xt.PROTON_MASS_EV)
line.build_tracker()

tw = line.twiss()

num_partilces_per_bunch = 100
num_bunches = 3
total_intensity_particles_bunch = 1e11

beam = xp.generate_matched_gaussian_bunch(
            num_particles=num_partilces_per_bunch * num_bunches,
            total_intensity_particles=total_intensity_particles_bunch * num_bunches,
            sigma_z=0.1, nemitt_x=2.5e-6, nemitt_y=2.5e-6, line=line)

harmonic_number = 4620
dz_bucket = tw.circumference / harmonic_number
bunch_spacing_buckets = 5

for ii in range(num_bunches):
    beam.zeta[ii * num_partilces_per_bunch:(ii+1) * num_partilces_per_bunch] += (
        ii * bunch_spacing_buckets * dz_bucket)