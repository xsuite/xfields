import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from pathlib import Path

# line = xt.Line.from_json(
#     '../../../xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
# line.particle_ref = xt.Particles(p0c=26e9, mass0=xt.PROTON_MASS_EV)
# line.build_tracker()
# tw = line.twiss()

# num_partilces_per_bunch = 100
# num_bunches = 3
# total_intensity_particles_bunch = 1e11

# beam = xp.generate_matched_gaussian_bunch(
#             num_particles=num_partilces_per_bunch * num_bunches,
#             total_intensity_particles=total_intensity_particles_bunch * num_bunches,
#             sigma_z=0.1, nemitt_x=2.5e-6, nemitt_y=2.5e-6, line=line)

# harmonic_number = 4620
# dz_bucket = tw.circumference / harmonic_number
# bunch_spacing_buckets = 5

# for ii in range(num_bunches):
#     beam.zeta[ii * num_partilces_per_bunch:(ii+1) * num_partilces_per_bunch] += (
#         ii * bunch_spacing_buckets * dz_bucket)


_configure_grid = xf.fieldmaps.interpolated._configure_grid



test_source = r"""
/*gpufun*/
void test_function(UniformBinSlicerData el,
                LocalParticle* part0,
                /*gpuglmem*/ double* b){

    double const a = UniformBinSlicerData_get_a(el);

    //start_per_particle_block (part0->part)

        const int64_t ipart = part->ipart;
        double const val = b[ipart];

        LocalParticle_add_to_s(part, val + a);

    //end_per_particle_block
}

/*gpufun*/
void UniformBinSlicer_track_local_particle(UniformBinSlicerData el,
                LocalParticle* part0){

    double const a = UniformBinSlicerData_get_a(el);

    //start_per_particle_block (part0->part)

        LocalParticle_set_s(part, a);

    //end_per_particle_block
}

"""

class UniformBinSlicer(xt.BeamElement):
    _xofields = {
        'z_min': xo.Float64,
        'num_slices': xo.Int64,
        'dzeta': xo.Float64,
        'i_bunch_0': xo.Int64,
        'num_bunches': xo.Int64,
        'bunch_spacing_zeta': xo.Float64,
        'a': xo.Float64,
    }

    _rename = {
        'z_min': '_z_min',
        'num_slices': '_num_slices',
        'dzeta': '_dzeta',
        'i_bunch_0': '_i_bunch_0',
        'num_bunches': '_num_bunches',
        'bunch_spacing_zeta': '_bunch_spacing_zeta',
    }

    _extra_c_sources = [
        Path('uniform_bin_slicer.h')
    ]

    _per_particle_kernels = {
            'test_slice': xo.Kernel(
                c_name='UniformBinSlicer_slice',
                args=[
                    xo.Arg(xo.Float64, pointer=True, name='b')
                ]),
        }

    def __init__(self, zeta_range=None, nbins=None, dzeta=None, zeta_grid=None, **kwargs):

        self._zeta_grid = _configure_grid('zeta', zeta_grid, dzeta, zeta_range, nbins)
        self.xoinitialize(z_min=self.zeta_grid[0], num_slices=self.num_slices,
                          dzeta=self.dzeta,
                          i_bunch_0=0, num_bunches=0, bunch_spacing_zeta=0, # To be implemented
                          **kwargs)
    @property
    def zeta_grid(self):
        """
        Array with the grid points (bin centers).
        """
        return self._zeta_grid


    @property
    def num_slices(self):
        """
        Number of bins
        """
        return len(self.zeta_grid)

    @property
    def dzeta(self):
        """
        Bin size in meters.
        """
        return self.zeta_grid[1] - self.zeta_grid[0]

slicer = UniformBinSlicer(zeta_range=(-1, 1), nbins=3)

p = xt.Particles(zeta=[-1, 0, 1])

ss = 0 * p.x
ctx= xo.ContextCpu()
slicer.test_slice(particles=p, b=p.x*4)