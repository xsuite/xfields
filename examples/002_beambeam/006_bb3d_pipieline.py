

pipeline_manager = PipelineManager(communicator = MPI.COMM_WORLD)

update_config = UpdateConfigBeamBeamBiGaussian3D(
    pipeline_manager=pipeline_manager,
    element_name='bbho_IP1_b1',
    partner_element_name='bbho_IP1_b2'
    slicer = SlicerConstantCharge(num_slices=10),
    element_name = 'bb_beam1_at_IP1'
    partner_element_name = 'bb_beam2_at_IP1'
    collision_schedule = {
        'beam1_bunch1': 'beam2_bunch1',
        'beam1_bunch2': 'beam2_bunch2',
        'beam1_bunch3': 'beam2_bunch3',
    }
    update_every=10 # Update every 10 turns
)

bb = xf.BeamBeamBiGaussian3D(
        phi=phi, alpha=alpha, q0_other_beam=1,

        ref_shift_x=x_co,
        ref_shift_px=px_co,
        ref_shift_y=y_co,
        ref_shift_py=py_co,
        ref_shift_zeta=zeta_co,
        ref_shift_pzeta=delta_co,

        update_config=update_config,
)