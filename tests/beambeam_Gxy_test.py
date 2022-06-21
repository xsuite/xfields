'''
Test to check if the fix for Gx and Gy are working. This creates a particle orbit around 0, because the HO beam beam element gives no kick. 
The px and py of the track should remain the same through the element. 


'''


import xtrack as xt
import xfields as xf
import xpart as xp
import xobjects as xo

line = xt.Line(
    elements=[xf.BeamBeamBiGaussian3D],
    element_names=['bb'])


bb_coupling = False
params = {}
params['phi'] = 0
params['alpha'] =  0
params['x_bb_co'] =  0
params['y_bb_co'] =  0
params['charge_slices'] = [0.0]
params['zeta_slices'] =  [0.0]
params['sigma_11'] =  1e-10
params['sigma_12'] =  0 # some values taken from teh pymask to reproduce the specific error
params['sigma_13'] =  0
params['sigma_14'] =  0
params['sigma_22'] =  0
params['sigma_23'] =  0
params['sigma_24'] =  0
params['sigma_33'] =  1e-10
params['sigma_34'] =  0
params['sigma_44'] =  0
if not (bb_coupling):
    params['sigma_13'] =  0.0
    params['sigma_14'] =  0.0
    params['sigma_23'] =  0.0
    params['sigma_24'] =  0.0

params["x_co"] = 0
params["px_co"] = 0
params["y_co"] =0
params["py_co"] = 0
params["zeta_co"] = 0
params["delta_co"] = 0

params["d_x"] = 0 # how is this calculated during the pymask?
params["d_px"] = 0
params["d_y"] = 0
params["d_py"] = 0
params["d_zeta"] = 0
params["d_delta"] = 0


newee = xf.BeamBeamBiGaussian3D(old_interface=params)
line.element_dict['bb'] = newee

context = xo.ContextCpu()         # For CPU

## Transfer lattice on context and compile tracking code
tracker = xt.Tracker(_context=context, line=line)

## Build particle object on context
n_part = 1
particles = xp.Particles(_context=context,
                        p0c=6500e9,
                        x=0,      
                        y=0,
                        )
n_turns = 2


tracker.track(particles, num_turns=n_turns,
              turn_by_turn_monitor=True)


# %% Turn-by-turn data is available at:
print('px', tracker.record_last_track.px)
print('py', tracker.record_last_track.py)

print('x', tracker.record_last_track.x)
print('y', tracker.record_last_track.y)


