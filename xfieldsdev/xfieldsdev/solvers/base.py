# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def __init__(self, context=None, **kwargs):
        pass

    @abstractmethod
    def solve(self, rho):
        return phi


