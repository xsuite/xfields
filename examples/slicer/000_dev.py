import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../../xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
line.particle_ref = xt.Particles(p0c=26e9, mass0=xt.PROTON_MASS_EV)
line.build_tracker()

bunch1 = xp.generate_matched_gaussian_bunch(num_particles=100,
            total_intensity_particles=1e11,
            sigma_z=0.1, nemitt_x=2.5e-6, nemitt_y=2.5e-6, line=line)

bunch2 = xp.generate_matched_gaussian_bunch(num_particles=100,
            total_intensity_particles=2e11,
            sigma_z=0.1, nemitt_x=2.5e-6, nemitt_y=2.5e-6, line=line)

tw = line.twiss()

harmonic_number = 4620
dz_bucket = tw.circumference  /harmonic_number
bunch_spacing_buckets = 5