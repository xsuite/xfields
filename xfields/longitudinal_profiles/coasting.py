
class LongitudinalProfileCoasting(object):


    def __init__(self, context=None, beam_line_density=None):

        if context is None:
            context = ContextDefault()

        assert beam_line_density is not None

        self.beam_line_density = beam_line_density

    def line_density(self, z):
        return self.beam_line_density
