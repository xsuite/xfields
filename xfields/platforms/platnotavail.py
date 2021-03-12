
class ModuleNotAvailable(object):
    def __init__(self, message='Module not available'):
        self.message=message

    def __getattr__(self, attr):
        raise NameError(self.message)
