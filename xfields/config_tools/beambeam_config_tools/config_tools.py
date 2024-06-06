import copy

import pandas as pd
import numpy as np
from scipy.special import  erfinv

from ._madpoint import MadPoint
import xfields as xf

def install_beambeam_elements_in_lines(line_b1, line_b4, ip_names,
            harmonic_number, bunch_spacing_buckets,
            num_long_range_encounters_per_side, num_slices_head_on,
            sigmaz_m, delay_at_ips_slots=None):

    keep_columns = ['beam', 'other_beam', 'ip_name', 'elementName', 'other_elementName', 'label',
            'self_particle_charge', 'self_relativistic_beta', 'self_frac_of_bunch',
            'identifier', 's_crab']

    # TODO: use keyword arguments
    # TODO: what happens if bunch length is different for the two beams
    if line_b1 is not None:
        circumference = line_b1.get_length()
        bb_df_b1 = generate_set_of_bb_encounters_1beam(
            circumference, harmonic_number,
            bunch_spacing_buckets,
            num_slices_head_on,
            line_b1.particle_ref.q0,
            sigmaz_m, line_b1.particle_ref.beta0[0], ip_names, num_long_range_encounters_per_side,
            beam_name = 'b1',
            other_beam_name = 'b2')
        install_dummy_bb_lenses(bb_df=bb_df_b1, line=line_b1)
        bb_df_b1 = bb_df_b1[keep_columns].copy()
    else:
        bb_df_b1 = None

    if line_b4 is not None:
        circumference = line_b4.get_length()
        bb_df_b2 = generate_set_of_bb_encounters_1beam(
            circumference, harmonic_number,
            bunch_spacing_buckets,
            num_slices_head_on,
            line_b4.particle_ref.q0,
            sigmaz_m,
            line_b4.particle_ref.beta0[0], ip_names, num_long_range_encounters_per_side,
            beam_name = 'b2',
            other_beam_name = 'b1')
        bb_df_b2['atPosition'] = -bb_df_b2['atPosition'] # I am installing in b4 not in b2
        install_dummy_bb_lenses(bb_df=bb_df_b2, line=line_b4)
        bb_df_b2 = bb_df_b2[keep_columns].copy()
    else:
        bb_df_b2 = None

    if delay_at_ips_slots is not None:
        _compute_delays(bb_df_b1, bb_df_b2, delay_at_ips_slots, ip_names,
                        harmonic_number, bunch_spacing_buckets)

    return bb_df_b1, bb_df_b2

def configure_beam_beam_elements(bb_df_cw, bb_df_acw, line_cw, line_acw,
                                 num_particles,
                                 nemitt_x, nemitt_y, crab_strong_beam, ip_names,
                                 use_antisymmetry=False,
                                 separation_bumps=None):

    if line_cw is None or line_acw is None:
        assert use_antisymmetry is True, (
            'If you are not using antisymmetry, you need to provide both beams')
    else:
        assert use_antisymmetry is False, (
            'If you are using antisymmetry, you need to provide only one beam'
            ' (for now...).')

    twisses = {}
    for bb_df, line, orientation in zip(
        [bb_df_cw, bb_df_acw], [line_cw, line_acw], ['cw', 'acw']):

        if bb_df is None:
            continue

        twiss = line.twiss()
        if orientation == 'acw':
            tw_acw = twiss
            twiss = twiss.reverse()
        twisses[orientation] = twiss

        surveys = {}

        for ip_name in ip_names:
            sv_ip = line.survey(element0=ip_name)
            if orientation == 'acw':
                sv_ip = sv_ip.reverse()
            surveys[ip_name] = sv_ip
            assert sv_ip['X', ip_name] == 0
            assert sv_ip['Y', ip_name] == 0
            assert sv_ip['Z', ip_name] == 0

        sigmas = twiss.get_betatron_sigmas(nemitt_x=nemitt_x, nemitt_y=nemitt_y)

        bb_df['self_num_particles'] = num_particles * bb_df['self_frac_of_bunch']

        # Use survey and twiss to get geometry and locations of all encounters
        compute_geometry_and_optics(
            bb_df=bb_df,
            xsuite_twiss=twiss,
            xsuite_survey=surveys,
            xsuite_sigmas=sigmas)

        if crab_strong_beam:
            measure_crabbing(line, bb_df, reverse=orientation=='acw')

    # Get geometry and optics at the partner encounter
    if not use_antisymmetry:
        get_partner_position_and_optics(bb_df_cw, bb_df_acw,
                                        crab_strong_beam=crab_strong_beam)
    else:
        if line_cw is not None:
            get_partner_position_and_optics_antisymmetry(bb_df_cw,
                                    crab_strong_beam=crab_strong_beam,
                                    separation_bumps=separation_bumps)
        elif line_acw is not None:
            get_partner_position_and_optics_antisymmetry(bb_df_acw,
                                    crab_strong_beam=crab_strong_beam,
                                    separation_bumps=separation_bumps)

    # Compute separation, crossing plane rotation, crossing angle and xma
    for bb_df in [bb_df_cw, bb_df_acw]:

        if bb_df is None:
            continue

        bb_df['separation_x'], bb_df['separation_y'] = find_bb_separations(
             points_weak=bb_df['self_lab_position'].values,
             points_strong=bb_df['other_lab_position'].values,
             names=bb_df.index.values)

        compute_dpx_dpy(bb_df)
        compute_local_crossing_angle_and_plane(bb_df)

        if crab_strong_beam:
            bb_df['separation_x_no_crab'] = bb_df['separation_x']
            bb_df['separation_y_no_crab'] = bb_df['separation_y']
            bb_df['separation_x'] += bb_df['other_x_crab']
            bb_df['separation_y'] += bb_df['other_y_crab']

    # Get bb dataframe and mad model (with dummy bb) for beam 4
    bb_df_b4 = get_counter_rotating(bb_df_acw) if bb_df_acw is not None else None

    if line_cw is not None:
        setup_beam_beam_in_line(line_cw, bb_df_cw, bb_coupling=False)
        xf.configure_orbit_dependent_parameters_for_bb(line=line_cw,
                        particle_on_co=twisses['cw'].particle_on_co)
    if line_acw is not None:
        setup_beam_beam_in_line(line_acw, bb_df_b4, bb_coupling=False)
        xf.configure_orbit_dependent_parameters_for_bb(line=line_acw,
                    particle_on_co=twisses['acw'].reverse().particle_on_co)

def install_dummy_bb_lenses(bb_df, line):

    ip_names = bb_df['ip_name'].unique().tolist()

    s_ips = {}
    for iipp in ip_names:
        s_ips[iipp] = line.get_s_position(iipp)

    s_insertions = []
    for nn in bb_df.index:
        s_insertions.append(s_ips[bb_df.loc[nn, 'ip_name']] + bb_df.loc[nn, 'atPosition'])
    line.cut_at_s(s_insertions)

    for nn in bb_df.index:
        print(f'Insert: {nn}     ', end='\r', flush=True)
        ll = bb_df.loc[nn, 'label']
        iipp = bb_df.loc[nn, 'ip_name']

        if ll == 'bb_ho':
            new_bb = xf.BeamBeamBiGaussian3D(phi=0, alpha=0, other_beam_q0=0.,
                slices_other_beam_num_particles=[0],
                slices_other_beam_zeta_center=[0],
                slices_other_beam_Sigma_11=[1],
                slices_other_beam_Sigma_12=[0],
                slices_other_beam_Sigma_22=[0],
                slices_other_beam_Sigma_33=[1],
                slices_other_beam_Sigma_34=[0],
                slices_other_beam_Sigma_44=[0],
                )
        elif ll == 'bb_lr':
            new_bb = xf.BeamBeamBiGaussian2D(
                other_beam_beta0=1.,
                other_beam_q0=0,
                other_beam_num_particles=0.,
                other_beam_Sigma_11=1,
                other_beam_Sigma_33=1,
            )
        else:
            raise ValueError('Unknown label')

        line.insert_element(element=new_bb,
                                    at_s=(s_ips[bb_df.loc[nn, 'ip_name']]
                                        + bb_df.loc[nn, 'atPosition']),
                                    name=nn)

_sigma_names = [11, 12, 13, 14, 22, 23, 24, 33, 34, 44]

def norm(v):
    return np.sqrt(np.sum(v ** 2))

# From https://github.com/giadarol/WeakStrong/blob/master/slicing.py
def constant_charge_slicing_gaussian(N_part_tot, sigmaz, N_slices):
    if N_slices>1:
        # working with intensity 1. and rescling at the end
        Qi = (np.arange(N_slices)/float(N_slices))[1:]

        z_cuts = np.sqrt(2)*sigmaz*erfinv(2*Qi-1.)

        z_centroids = []
        first_centroid = -sigmaz/np.sqrt(2*np.pi)*np.exp(
                -z_cuts[0]**2/(2*sigmaz*sigmaz))*float(N_slices)
        z_centroids.append(first_centroid)
        for ii in range(N_slices-2):
            this_centroid = -sigmaz/np.sqrt(2*np.pi)*(
                    np.exp(-z_cuts[ii+1]**2/(2*sigmaz*sigmaz))-
                    np.exp(-z_cuts[ii]**2/(2*sigmaz*sigmaz)))*float(N_slices)
            # the multiplication times n slices comes from the fact
            # that we have to divide by the slice charge, i.e. 1./N
            z_centroids.append(this_centroid)

        last_centroid = sigmaz/np.sqrt(2*np.pi)*np.exp(
                -z_cuts[-1]**2/(2*sigmaz*sigmaz))*float(N_slices)
        z_centroids.append(last_centroid)

        z_centroids = np.array(z_centroids)

        N_part_per_slice = z_centroids*0.+N_part_tot/float(N_slices)
    elif N_slices==1:
        z_centroids = np.array([0.])
        z_cuts = []
        N_part_per_slice = np.array([N_part_tot])

    else:
        raise ValueError('Invalid number of slices')

    return z_centroids, z_cuts, N_part_per_slice

def elementName(label, IRNumber, beam, identifier):
    if identifier >0:
        sideTag='.r'
    elif identifier < 0:
        sideTag='.l'
    else:
        sideTag='.c'
    return f'{label}{sideTag}{IRNumber}{beam}_{np.abs(identifier):02}'

def generate_set_of_bb_encounters_1beam(
    circumference=None,
    harmonic_number=None,
    bunch_spacing_buckets=None,
    numberOfHOSlices=None,
    bunch_particle_charge=None,
    sigt=None,
    relativistic_beta=None,
    ip_names=None,
    numberOfLRPerIRSide=None,
    beam_name=None,
    other_beam_name=None
    ):


    # Long-Range
    myBBLRlist=[]
    for ii, ip_nn in enumerate(ip_names):
        for identifier in (list(range(-numberOfLRPerIRSide[ii],0))
                           + list(range(1,numberOfLRPerIRSide[ii]+1))):
            myBBLRlist.append({'label': 'bb_lr', 'ip_name': ip_nn,
                               'beam': beam_name, 'other_beam':other_beam_name,
                               'identifier':identifier})

    if len(myBBLRlist)>0:
        myBBLR=pd.DataFrame(myBBLRlist)[
            ['beam','other_beam','ip_name','label','identifier']]

        myBBLR['self_particle_charge'] = bunch_particle_charge
        myBBLR['self_relativistic_beta'] = relativistic_beta
        myBBLR['elementName']=myBBLR.apply(
            lambda x: elementName(
                x.label, x.ip_name.replace('ip', ''), x.beam, x.identifier),
                axis=1)
        myBBLR['other_elementName']=myBBLR.apply(
            lambda x: elementName(
                x.label, x.ip_name.replace('ip', ''), x.other_beam, x.identifier), axis=1)
        # where circ is used
        BBSpacing = circumference / harmonic_number * bunch_spacing_buckets / 2.
        myBBLR['atPosition']=BBSpacing*myBBLR['identifier']
        myBBLR['s_crab'] = 0.
        myBBLR['self_frac_of_bunch'] = 1.
        # assuming a sequence rotated in IR3
    else:
        myBBLR = pd.DataFrame()

    # Head-On
    numberOfSliceOnSide=int((numberOfHOSlices-1)/2)
    # to check: sigz of the luminous region
    # where sigt is used
    sigzLumi=sigt/2
    z_centroids, z_cuts, N_part_per_slice = constant_charge_slicing_gaussian(
                                                    1,sigzLumi,numberOfHOSlices)
    myBBHOlist=[]

    for ip_nn in ip_names:
        for identifier in (list(range(-numberOfSliceOnSide,0))
                           +[0] + list(range(1,numberOfSliceOnSide+1))):
            myBBHOlist.append({'label': 'bb_ho', 'ip_name': ip_nn,
                                'other_beam': other_beam_name, 'beam':beam_name,
                                'identifier':identifier})

    myBBHO=pd.DataFrame(myBBHOlist)[
        ['beam','other_beam', 'ip_name','label','identifier']]


    myBBHO['self_frac_of_bunch'] = 1./numberOfHOSlices
    myBBHO['self_particle_charge'] = bunch_particle_charge
    myBBHO['self_relativistic_beta'] = relativistic_beta
    for ip_nn in ip_names:
        myBBHO.loc[myBBHO['ip_name'] == ip_nn, 'atPosition']=list(z_centroids)
    myBBHO['s_crab'] = myBBHO['atPosition']

    myBBHO['elementName'] = myBBHO.apply(
        lambda x: elementName(
            x.label, x.ip_name.replace('ip', ''), x.beam, x.identifier), axis=1)
    myBBHO['other_elementName']=myBBHO.apply(
        lambda x: elementName(x.label, x.ip_name.replace('ip', ''),
                              x.other_beam, x.identifier), axis=1)
    # assuming a sequence rotated in IR3

    myBB=pd.concat([myBBHO, myBBLR],sort=False)
    myBB = myBB.set_index('elementName', drop=False, verify_integrity=True).sort_index()


    for ww in ['self', 'other']:
        for coord in ['x', 'px', 'y', 'py']:
            myBB[f'{ww}_{coord}_crab'] = 0

    return myBB

def get_counter_rotating(bb_df):

    c_bb_df = pd.DataFrame(index=bb_df.index)

    c_bb_df['beam'] = bb_df['beam']
    c_bb_df['other_beam'] = bb_df['other_beam']
    c_bb_df['ip_name'] = bb_df['ip_name']
    c_bb_df['label'] = bb_df['label']
    c_bb_df['identifier'] = bb_df['identifier']
    if 'elementClass' in bb_df.columns:
        c_bb_df['elementClass'] = bb_df['elementClass']
    c_bb_df['elementName'] = bb_df['elementName']
    c_bb_df['self_num_particles'] = bb_df['self_num_particles']
    c_bb_df['other_num_particles'] = bb_df['other_num_particles']
    c_bb_df['self_particle_charge'] = bb_df['self_particle_charge']
    c_bb_df['other_particle_charge'] = bb_df['other_particle_charge']
    c_bb_df['other_elementName'] = bb_df['other_elementName']

    if 'atPosition' in bb_df.columns:
        c_bb_df['atPosition'] = bb_df['atPosition'] * (-1.)

    c_bb_df['elementDefinition'] = np.nan
    c_bb_df['elementInstallation'] = np.nan

    c_bb_df['self_lab_position'] = np.nan
    c_bb_df['other_lab_position'] = np.nan

    c_bb_df['self_Sigma_11'] = bb_df['self_Sigma_11'] * (-1.) * (-1.)                  # x * x
    c_bb_df['self_Sigma_12'] = bb_df['self_Sigma_12'] * (-1.) * (-1.) * (-1.)          # x * dx / ds
    c_bb_df['self_Sigma_13'] = bb_df['self_Sigma_13'] * (-1.)                          # x * y
    c_bb_df['self_Sigma_14'] = bb_df['self_Sigma_14'] * (-1.) * (-1.)                  # x * dy / ds
    c_bb_df['self_Sigma_22'] = bb_df['self_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)  # dx / ds * dx / ds
    c_bb_df['self_Sigma_23'] = bb_df['self_Sigma_23'] * (-1.) * (-1.)                  # dx / ds * y
    c_bb_df['self_Sigma_24'] = bb_df['self_Sigma_24'] * (-1.) * (-1.) * (-1.)          # dx / ds * dy / ds
    c_bb_df['self_Sigma_33'] = bb_df['self_Sigma_33']                                  # y * y
    c_bb_df['self_Sigma_34'] = bb_df['self_Sigma_34'] * (-1.)                          # y * dy / ds
    c_bb_df['self_Sigma_44'] = bb_df['self_Sigma_44'] * (-1.) * (-1.)                  # dy / ds * dy / ds

    c_bb_df['other_Sigma_11'] = bb_df['other_Sigma_11'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_12'] = bb_df['other_Sigma_12'] * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_13'] = bb_df['other_Sigma_13'] * (-1.)
    c_bb_df['other_Sigma_14'] = bb_df['other_Sigma_14'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_22'] = bb_df['other_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_23'] = bb_df['other_Sigma_23'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_24'] = bb_df['other_Sigma_24'] * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_33'] = bb_df['other_Sigma_33']
    c_bb_df['other_Sigma_34'] = bb_df['other_Sigma_34'] * (-1.)
    c_bb_df['other_Sigma_44'] = bb_df['other_Sigma_44'] * (-1.) * (-1.)

    c_bb_df['other_relativistic_beta']=bb_df['other_relativistic_beta']
    c_bb_df['separation_x'] = bb_df['separation_x'] * (-1.)
    c_bb_df['separation_y'] = bb_df['separation_y']

    c_bb_df['dpx'] = bb_df['dpx'] * (-1.) * (-1.)
    c_bb_df['dpy'] = bb_df['dpy'] * (-1.)

    if 'self_x_crab' in c_bb_df.columns:
        # Crab cavities are accounted for
        c_bb_df['separation_x_no_crab'] = bb_df['separation_x_no_crab'] * (-1.)
        c_bb_df['separation_y_no_crab'] = bb_df['separation_y_no_crab']
        for ww in ['self', 'other']:
            c_bb_df[f'{ww}_x_crab'] = bb_df[f'{ww}_x_crab'] * (-1)
            c_bb_df[f'{ww}_px_crab'] = bb_df[f'{ww}_px_crab'] * (-1) * (-1)
            c_bb_df[f'{ww}_y_crab'] = bb_df[f'{ww}_y_crab']
            c_bb_df[f'{ww}_py_crab'] = bb_df[f'{ww}_py_crab'] * (-1)

    # Compute phi and alpha from dpx and dpy
    compute_local_crossing_angle_and_plane(c_bb_df)

    return c_bb_df

def compute_geometry_and_optics(bb_df=None, xsuite_twiss=None, xsuite_survey=None,
                            xsuite_sigmas=None):


    # Get positions of the bb encounters (absolute from survey), closed orbit
    # and orientation of the local reference system (MadPoint objects)

    # Add empty columns to dataframe
    bb_df['self_lab_position'] = None
    bb_df['s'] = None
    bb_df['s_ip'] = None
    bb_df['self_Sigma_11'] = None
    bb_df['self_Sigma_12'] = None
    bb_df['self_Sigma_13'] = None
    bb_df['self_Sigma_14'] = None
    bb_df['self_Sigma_22'] = None
    bb_df['self_Sigma_23'] = None
    bb_df['self_Sigma_24'] = None
    bb_df['self_Sigma_33'] = None
    bb_df['self_Sigma_34'] = None
    bb_df['self_Sigma_44'] = None

    for ele_name in bb_df.index.values:
        ip_name = bb_df['ip_name'][ele_name]

        bb_df.loc[ele_name, 'self_lab_position'] = MadPoint(ele_name, None,
                        use_twiss=True, use_survey=True,
                        xsuite_survey=xsuite_survey[ip_name],
                        xsuite_twiss=xsuite_twiss)

        bb_df.loc[ele_name, 's'] = xsuite_twiss['s', ele_name]
        bb_df.loc[ele_name, 's_ip'] = xsuite_twiss['s', ip_name]

        # Get the sigmas for the element
        i_sigma = np.where(np.array(xsuite_sigmas.name) == ele_name)[0][0]
        for ss in [
            '11', '12', '13', '14', '22', '23', '24', '33', '34', '44']:
            bb_df.loc[ele_name, f'self_Sigma_{ss}'] = xsuite_sigmas[
                                                    'Sigma'+ss][i_sigma]


def get_partner_position_and_optics(bb_df_b1, bb_df_b2, crab_strong_beam):

    dict_dfs = {'b1': bb_df_b1, 'b2': bb_df_b2}

    for self_beam_nn in ['b1', 'b2']:

        self_df = dict_dfs[self_beam_nn]
        self_df['other_num_particles'] = None
        self_df['other_particle_charge'] = None
        self_df['other_relativistic_beta'] = None
        for ee in self_df.index:
            other_beam_nn = self_df.loc[ee, 'other_beam']
            other_df = dict_dfs[other_beam_nn]
            other_ee = self_df.loc[ee, 'other_elementName']

            # Get position of the other beam in its own survey
            other_lab_position = copy.deepcopy(other_df.loc[other_ee, 'self_lab_position'])

            # Store positions
            self_df.loc[ee, 'other_lab_position'] = other_lab_position

            # Get sigmas of the other beam in its own survey
            for ss in _sigma_names:
                self_df.loc[ee, f'other_Sigma_{ss}'] = other_df.loc[other_ee, f'self_Sigma_{ss}']
            # Get charge of other beam
            self_df.loc[ee, 'other_num_particles'] = other_df.loc[other_ee, 'self_num_particles']
            self_df.loc[ee, 'other_particle_charge'] = other_df.loc[other_ee, 'self_particle_charge']
            self_df.loc[ee, 'other_relativistic_beta'] = other_df.loc[other_ee, 'self_relativistic_beta']

            if crab_strong_beam:
                for coord in ['x', 'y']:
                    self_df.loc[ee, f'other_{coord}_crab'] = other_df.loc[
                        other_ee, f'self_{coord}_crab']

def get_partner_position_and_optics_antisymmetry(bb_df, crab_strong_beam,
                    separation_bumps=None):

    bb_df['other_num_particles'] = None
    bb_df['other_particle_charge'] = None
    bb_df['other_relativistic_beta'] = None
    for ee in bb_df.index:

        ds = bb_df.loc[ee, 's'] - bb_df.loc[ee, 's_ip']
        s_antisim = bb_df.loc[ee, 's_ip'] - ds
        i_antisym = np.argmin(np.abs(bb_df.s - s_antisim))
        other_ee = bb_df.index[i_antisym]

        assert np.isclose(
            bb_df.loc[other_ee, 's'], s_antisim, rtol=0, atol=1e-5)

        position_ee = bb_df.loc[ee, 'self_lab_position']
        position_other_ee = copy.deepcopy(
            bb_df.loc[other_ee, 'self_lab_position'])
        # Assuming survey has been made starting from the IP and neglecting
        # angle between the two surveys
        position_other_ee.sz = position_ee.sz # longitudinal component
        position_other_ee.p[2] = position_ee.p[2] # longitudinal component
        position_other_ee.tpx *= -1 # anti-symmetry
        position_other_ee.tpy *= -1 # anti-symmetry

        if separation_bumps is not None:
            if bb_df.loc[ee, 'ip_name'] in separation_bumps:
                sep_plane = separation_bumps[bb_df.loc[ee, 'ip_name']]
                setattr(position_other_ee, f't{sep_plane}',
                    -getattr(position_other_ee, f't{sep_plane}'))
                setattr(position_other_ee, f'tp{sep_plane}',
                    -getattr(position_other_ee, f'tp{sep_plane}'))
                position_other_ee.p[{'x': 0, 'y': 1}[sep_plane]] += (
                    2 * getattr(position_other_ee, f't{sep_plane}')
                    # 2 is to compensate for the fact that the orbit was already
                    # added with the wrong sign
                )

        # Store positions
        bb_df.loc[ee, 'other_lab_position'] = position_other_ee

        # Get sigmas of the other beam (signs come from anti-symmetry)
        bb_df.loc[ee, 'other_Sigma_11'] = bb_df.loc[other_ee, 'self_Sigma_11']
        bb_df.loc[ee, 'other_Sigma_12'] = -bb_df.loc[other_ee, 'self_Sigma_12']
        bb_df.loc[ee, 'other_Sigma_13'] = bb_df.loc[other_ee, 'self_Sigma_13']
        bb_df.loc[ee, 'other_Sigma_14'] = -bb_df.loc[other_ee, 'self_Sigma_14']
        bb_df.loc[ee, 'other_Sigma_22'] = bb_df.loc[other_ee, 'self_Sigma_22']
        bb_df.loc[ee, 'other_Sigma_23'] = -bb_df.loc[other_ee, 'self_Sigma_23']
        bb_df.loc[ee, 'other_Sigma_24'] = bb_df.loc[other_ee, 'self_Sigma_24']
        bb_df.loc[ee, 'other_Sigma_33'] = bb_df.loc[other_ee, 'self_Sigma_33']
        bb_df.loc[ee, 'other_Sigma_34'] = -bb_df.loc[other_ee, 'self_Sigma_34']
        bb_df.loc[ee, 'other_Sigma_44'] = bb_df.loc[other_ee, 'self_Sigma_44']

        # Get charge of other beam
        bb_df.loc[ee, 'other_num_particles'] = bb_df.loc[other_ee, 'self_num_particles']
        bb_df.loc[ee, 'other_particle_charge'] = bb_df.loc[other_ee, 'self_particle_charge']
        bb_df.loc[ee, 'other_relativistic_beta'] = bb_df.loc[other_ee, 'self_relativistic_beta']

        if crab_strong_beam:
            for coord in ['x', 'y']:
                bb_df.loc[ee, f'other_{coord}_crab'] = bb_df.loc[
                    other_ee, f'self_{coord}_crab']

def compute_dpx_dpy(bb_df):
    # Defined as (weak) - (strong)
    for ee in bb_df.index:
        dpx = (bb_df.loc[ee, 'self_lab_position'].tpx
                - bb_df.loc[ee, 'other_lab_position'].tpx)
        dpy = (bb_df.loc[ee, 'self_lab_position'].tpy
                - bb_df.loc[ee, 'other_lab_position'].tpy)

        bb_df.loc[ee, 'dpx'] = dpx
        bb_df.loc[ee, 'dpy'] = dpy

def compute_local_crossing_angle_and_plane(bb_df):

    for ee in bb_df.index:
        alpha, phi = find_alpha_and_phi(
                bb_df.loc[ee, 'dpx'], bb_df.loc[ee, 'dpy'])

        bb_df.loc[ee, 'alpha'] = alpha
        bb_df.loc[ee, 'phi'] = phi

def find_alpha_and_phi(dpx, dpy):

    absphi = np.sqrt(dpx ** 2 + dpy ** 2) / 2.0

    if absphi < 1e-20:
        phi = absphi
        alpha = 0.0
    else:
        if dpy>=0.:
            if dpx>=0:
                # First quadrant
                if np.abs(dpx) >= np.abs(dpy):
                    # First octant
                    phi = absphi
                    alpha = np.arctan(dpy/dpx)
                else:
                    # Second octant
                    phi = absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
            else: #dpx<0
                # Second quadrant
                if np.abs(dpx) <  np.abs(dpy):
                    # Third octant
                    phi = absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
                else:
                    # Forth  octant
                    phi = -absphi
                    alpha = np.arctan(dpy/dpx)
        else: #dpy<0
            if dpx<=0:
                # Third quadrant
                if np.abs(dpx) >= np.abs(dpy):
                    # Fifth octant
                    phi = -absphi
                    alpha = np.arctan(dpy/dpx)
                else:
                    # Sixth octant
                    phi = -absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
            else: #dpx>0
                # Forth quadrant
                if np.abs(dpx) <= np.abs(dpy):
                    # Seventh octant
                    phi = -absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
                else:
                    # Eighth octant
                    phi = absphi
                    alpha = np.arctan(dpy/dpx)

    return alpha, phi


def find_bb_separations(points_weak, points_strong, names=None):

    if names is None:
        names = ["bb_%d" % ii for ii in range(len(points_weak))]

    sep_x = []
    sep_y = []
    for i_bb, name_bb in enumerate(names):

        pbw = points_weak[i_bb]
        pbs = points_strong[i_bb]

        # Find vws
        vbb_ws = points_strong[i_bb].p - points_weak[i_bb].p

        # Check that the two reference system are parallel
        try:
            assert norm(pbw.ex - pbs.ex) < 1e-10  # 1e-4 is a reasonable limit
            assert norm(pbw.ey - pbs.ey) < 1e-10  # 1e-4 is a reasonable limit
            assert norm(pbw.ez - pbs.ez) < 1e-10  # 1e-4 is a reasonable limit
        except AssertionError:
            print(name_bb, "Reference systems are not parallel")
            if (
                np.sqrt(
                    norm(pbw.ex - pbs.ex) ** 2
                    + norm(pbw.ey - pbs.ey) ** 2
                    + norm(pbw.ez - pbs.ez) ** 2
                )
                < 5e-3
            ):
                print("Smaller that 5e-3, tolerated.")
            else:
                raise ValueError("Too large! Stopping.")

        # Check that there is no longitudinal separation
        try:
            assert np.abs(np.dot(vbb_ws, pbw.ez)) < 1e-4
        except AssertionError:
            print(name_bb, "The beams are longitudinally shifted")

        # Find separations
        sep_x.append(np.dot(vbb_ws, pbw.ex))
        sep_y.append(np.dot(vbb_ws, pbw.ey))

    return sep_x, sep_y

def setup_beam_beam_in_line(
    line,
    bb_df,
    bb_coupling=False,
):
    import xfields as xf
    assert bb_coupling is False  # Not implemented

    for ii, (ee, eename) in enumerate(zip(line.elements, line.element_names)):
        if isinstance(ee, xf.BeamBeamBiGaussian2D):
            ee.other_beam_num_particles=bb_df.loc[eename, 'other_num_particles']
            ee.other_beam_q0 = bb_df.loc[eename, 'other_particle_charge']
            ee.other_beam_Sigma_11 = bb_df.loc[eename, 'other_Sigma_11']
            ee.other_beam_Sigma_33 = bb_df.loc[eename, 'other_Sigma_33']
            ee.other_beam_beta0 = bb_df.loc[eename, 'other_relativistic_beta']
            ee.other_beam_shift_x = bb_df.loc[eename, 'separation_x']
            ee.other_beam_shift_y = bb_df.loc[eename, 'separation_y']
        if isinstance(ee, xf.BeamBeamBiGaussian3D):
            params = {}
            params['phi'] = bb_df.loc[eename, 'phi']
            params['alpha'] =  bb_df.loc[eename, 'alpha']
            params['other_beam_shift_x'] =  bb_df.loc[eename, 'separation_x']
            params['other_beam_shift_y'] =  bb_df.loc[eename, 'separation_y']
            params['slices_other_beam_num_particles'] =  [bb_df.loc[eename, 'other_num_particles']]
            params['other_beam_q0'] =  bb_df.loc[eename, 'other_particle_charge']
            params['slices_other_beam_zeta_center'] =  [0.0]
            params['slices_other_beam_Sigma_11'] = [bb_df.loc[eename, 'other_Sigma_11']]
            params['slices_other_beam_Sigma_12'] = [bb_df.loc[eename, 'other_Sigma_12']]
            params['slices_other_beam_Sigma_13'] = [bb_df.loc[eename, 'other_Sigma_13']]
            params['slices_other_beam_Sigma_14'] = [bb_df.loc[eename, 'other_Sigma_14']]
            params['slices_other_beam_Sigma_22'] = [bb_df.loc[eename, 'other_Sigma_22']]
            params['slices_other_beam_Sigma_23'] = [bb_df.loc[eename, 'other_Sigma_23']]
            params['slices_other_beam_Sigma_24'] = [bb_df.loc[eename, 'other_Sigma_24']]
            params['slices_other_beam_Sigma_33'] = [bb_df.loc[eename, 'other_Sigma_33']]
            params['slices_other_beam_Sigma_34'] = [bb_df.loc[eename, 'other_Sigma_34']]
            params['slices_other_beam_Sigma_44'] = [bb_df.loc[eename, 'other_Sigma_44']]

            if not (bb_coupling):
                params['slices_other_beam_Sigma_13'] = [0.0]
                params['slices_other_beam_Sigma_14'] = [0.0]
                params['slices_other_beam_Sigma_23'] = [0.0]
                params['slices_other_beam_Sigma_24'] = [0.0]

            newee = xf.BeamBeamBiGaussian3D(**params)

            # needs to be generalized for lenses with multiple slices
            assert newee._xobject._size == ee._xobject._size
            # move to the location of the old element (ee becomese newee)
            newee.move(_buffer=ee._buffer, _offset=ee._offset)

def measure_crabbing(line, bb_df, reverse):

    tw = line.twiss()
    if reverse:
        tw = tw.reverse()

    for nn in bb_df.index:
        s_crab = bb_df.loc[nn, 's_crab']
        if s_crab != 0.0:
            print(f'Crabbing at {nn}     ', end='\r', flush=True)
            zeta0 = 2 * s_crab
            if reverse:
                zeta0 = -zeta0 # LHC convention
            tw4d_crab = line.twiss(method='4d',zeta0=zeta0,
                                   freeze_longitudinal=True)
            if reverse:
                tw4d_crab = tw4d_crab.reverse()
            ii = np.where(np.array(tw.name) == nn)[0][0]

            for coord in ['x', 'px', 'y', 'py']:
                bb_df.loc[nn, f'self_{coord}_crab'] = (
                    tw4d_crab[coord][ii] - tw[coord][ii])
        else:
            for coord in ['x', 'px', 'y', 'py']:
                bb_df.loc[nn, f'self_{coord}_crab'] = 0.0

def _compute_delays(bb_df_cw, bb_df_acw, delay_at_ips_slots, ip_names,
                    harmonic_number, bunch_spacing_buckets):

    ring_length_in_slots = harmonic_number / bunch_spacing_buckets

    for orientation, bbdf  in zip(['clockwise', 'anticlockwise'],
                                  [bb_df_cw, bb_df_acw]):

        if orientation == 'clockwise':
            delay_at_ips_dict = {iipp: dd
                                for iipp, dd in zip(ip_names, delay_at_ips_slots)}
        elif orientation == 'anticlockwise':
            delay_at_ips_dict = {iipp: np.mod(ring_length_in_slots - dd, ring_length_in_slots)
                                for iipp, dd in zip(ip_names, delay_at_ips_slots)}
        else:
            raise ValueError('?!')

        delay_in_slots = []

        for nn in bbdf.index.values:
            ip_name = bbdf.loc[nn, 'ip_name']
            this_delay = delay_at_ips_dict[ip_name]

            if nn.startswith('bb_lr.'):
                if orientation == 'clockwise':
                    this_delay += bbdf.loc[nn, 'identifier']
                elif orientation == 'anticlockwise':
                    this_delay -= bbdf.loc[nn, 'identifier']
                else:
                    raise ValueError('?!')

            delay_in_slots.append(int(this_delay))

        bbdf['delay_in_slots'] = delay_in_slots

def apply_filling_pattern(collider, filling_pattern_cw, filling_pattern_acw,
                          i_bunch_cw, i_bunch_acw):

    dframes = collider._bb_config['dataframes']

    ring_length_in_slots = int(collider._bb_config['harmonic_number']
                            / collider._bb_config['bunch_spacing_buckets'])

    for orientation_self in ['clockwise', 'anticlockwise']:

        if orientation_self == 'clockwise':
            filling_pattern_self = np.array(filling_pattern_cw, dtype=int)
            filling_pattern_other = np.array(filling_pattern_acw, dtype=int)
            i_bunch_self = i_bunch_cw
        else:
            filling_pattern_self = np.array(filling_pattern_acw)
            filling_pattern_other = np.array(filling_pattern_cw)
            i_bunch_self = i_bunch_acw

        assert set(list(filling_pattern_self)).issubset({0, 1})
        assert set(list(filling_pattern_other)).issubset({0, 1})

        assert filling_pattern_self[i_bunch_self] == 1, "Selected bunch is not in the filling scheme"

        temp_df = dframes[orientation_self].loc[:, ['delay_in_slots', 'ip_name']].copy()
        temp_df['partner_bunch_index'] = dframes[orientation_self]['delay_in_slots'] + i_bunch_self
        temp_df['partner_bunch_index'] = np.mod(temp_df['partner_bunch_index'], ring_length_in_slots)
        temp_df['is_active'] = filling_pattern_other[temp_df['partner_bunch_index']] == 1

        for nn, state in temp_df['is_active'].items():
            if state:
                collider.vars[nn + '_scale_strength'] = collider.vars['beambeam_scale']
            else:
                collider.vars[nn + '_scale_strength'] = 0
