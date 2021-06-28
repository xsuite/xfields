from .beam_elements.spacecharge import SpaceChargeBiGaussian


def replace_spaceharge_with_quasi_frozen(
                        line, _buffer,
                        update_mean_x_on_track=True,
                        update_mean_y_on_track=True,
                        update_sigma_x_on_track=True,
                        update_sigma_y_on_track=True):

    spch_elements = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__ == 'SCQGaussProfile':
            newee = SpaceChargeBiGaussian.from_xline(ee, _buffer=_buffer)
            newee.update_mean_x_on_track = update_mean_x_on_track
            newee.update_mean_y_on_track = update_mean_y_on_track
            newee.update_sigma_x_on_track = update_sigma_x_on_track
            newee.update_sigma_y_on_track = update_sigma_y_on_track
            newee.iscollective = True
            line.elements[ii] = newee
            spch_elements.append(newee)

    return spch_elements
