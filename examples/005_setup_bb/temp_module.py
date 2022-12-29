import xfields as xf

def install_dummy_bb_lenses(bb_df, line):

    ip_names = bb_df['ip_name'].unique().tolist()

    s_ips = {}
    for iipp in ip_names:
        s_ips[iipp] = line.get_s_position(iipp)

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