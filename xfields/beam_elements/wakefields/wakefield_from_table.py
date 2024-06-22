import pandas
import numpy as np
from scipy.constants import c as clight
from scipy.interpolate import interp1d

from .wakefield import Wakefield, WakeComponent


VALID_WAKE_COMPONENTS = ['constant_x', 'constant_y', 'dipole_x',
                            'dipole_y', 'dipole_xy', 'dipole_yx',
                            'quadrupole_x', 'quadrupole_y',
                            'quadrupole_xy', 'quadrupole_yx',
                            'longitudinal']

class WakefieldFromTable(Wakefield):
    """
    Wakefield from a table.

    Parameters
    ----------
    table : pandas.DataFrame
        DataFrame with the wake functions by point in time.
    use_components : list of str
        List of components to use. If None, all components are used.
    zeta_range : Tuple
        Zeta range for each bunch used in the underlying slicer.
    num_slices : int
        Number of slices per bunch used in the underlying slicer.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    bunch_numbers: np.ndarray
        List of the bunches indicating which slots from the filling scheme are
        used (not all the bunches are used when using multi-processing)
    num_turns : int
        Number of turns which are consiered for the multi-turn wake.
    circumference: float
        Machine length in meters.
    log_moments: list
        List of moments logged in the slicer.
    _flatten: bool
        Use flattened wakes
    """
    def __init__(self, table, use_components, zeta_range, num_slices,
                 num_turns, circumference, beta0=1,
                 bunch_spacing_zeta=None, filling_scheme=None,
                 bunch_numbers=None, log_moments=None, _flatten=False):
        self.table = table
        self.use_components = use_components


        for component in use_components:
            if component not in VALID_WAKE_COMPONENTS:
                raise ValueError(
                    f'Invalid wake component: {component}. '
                    f'Valid wake components are: {VALID_WAKE_COMPONENTS}'
                )

        if 'time' not in list(table.keys()):
                    raise ValueError("No wake_file_column with name 'time' has" +
                                    " been specified. \n")

        if use_components is not None:
            for component in use_components:
                assert component in VALID_WAKE_COMPONENTS

        wake_distance = table['time'] * beta0 * clight
        components = []
        # Loop over the components and create the WakeComponent objects
        # for the components that are used
        for component in list(table.keys()):
            if component != 'time' and (use_components is None or
                                        component in use_components):
                assert component in VALID_WAKE_COMPONENTS
                scale_kick = None
                source_moments = ['num_particles']
                if component == 'longitudinal':
                    kick = 'delta'
                else:
                    tokens = component.split('_')
                    coord_target = tokens[1][0]
                    if len(tokens[1]) == 2:
                        coord_source = tokens[1][1]
                    else:
                        coord_source = coord_target
                    kick = 'p'+coord_target
                    if tokens[0] == 'dipole':
                        source_moments.append(coord_source)
                    elif tokens[0] == 'quadrupole':
                        scale_kick = coord_source
                wake_strength = table[component]
                wakefield = WakeComponent(
                    source_moments=source_moments,
                    kick=kick,
                    scale_kick=scale_kick,
                    function=interp1d(wake_distance, wake_strength,
                                      bounds_error=False, fill_value=0.0)
                )
                components.append(wakefield)

        super().__init__(
            components=components,
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            filling_scheme=filling_scheme,
            bunch_numbers=bunch_numbers,
            num_turns=num_turns,
            circumference=circumference,
            log_moments=log_moments,
            _flatten=_flatten)