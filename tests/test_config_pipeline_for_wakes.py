import xtrack as xt
import xwakes as xw
from xobjects.test_helpers import for_all_test_contexts

class DummyCommunicator:
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

@for_all_test_contexts
def test_config_pipeline_for_wakes(test_context):
    comm_size = 3
    for my_rank in range(comm_size):
        communicator = DummyCommunicator(rank=my_rank, size=comm_size)

        zeta_range = (-1, 1)
        num_slices = 10
        filling_scheme = [1, 0, 1, 1, 1]
        bunch_spacing_zeta = 5
        num_turns = 1
        circumference = 100

        wake1 = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

        wake1.configure_for_tracking(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_scheme=filling_scheme,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_turns=num_turns,
            circumference=circumference
        )

        wake2 = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

        wake2.configure_for_tracking(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_scheme=filling_scheme,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_turns=num_turns,
            circumference=circumference
        )

        wake3 = wake1 + wake2

        wake3.configure_for_tracking(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_scheme=filling_scheme,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_turns=num_turns,
            circumference=circumference
        )

        wake_no_config = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

        drift = xt.Drift(length=1)

        line = xt.Line(elements=[wake1, wake2, wake3, wake_no_config, drift])

        particles = xt.Particles(context=test_context)

        pipeline_manager = xw.config_pipeline_for_wakes(particles, line, communicator,
                                    elements_to_configure=None)

        assert len(pipeline_manager._IDs) == comm_size
        assert len(pipeline_manager._particles_per_rank) == comm_size

        for ii in range(comm_size):
            assert list(pipeline_manager._IDs.keys())[ii] == f'particles{ii}'
            assert pipeline_manager._particles_per_rank[ii] == [f'particles{ii}']

        assert len(pipeline_manager._elements) == 3
        assert pipeline_manager._elements['e0'] == 0
        assert pipeline_manager._elements['e1'] == 1
        assert pipeline_manager._elements['e2'] == 2

        assert pipeline_manager._communicator == communicator
        assert pipeline_manager.verbose == False
        assert pipeline_manager._pending_requests == {}
        assert pipeline_manager._last_request_turn == {}

        assert wake1._wake_tracker.partner_names == [f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank]
        assert wake2._wake_tracker.partner_names == [f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank]
        assert wake3._wake_tracker.partner_names == [f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank]

        assert particles.name == f'particles{my_rank}'
