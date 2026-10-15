# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from ..beam_elements.touschek import TouschekMarker


class TouschekGenerator:
    """
    Touschek scattering simulation of a line, based on the
    :class:`xfields.TouschekMarker` elements installed in it.

    An instance bound to a line is available as ``line.xfields.touschek``.
    The generator holds no state other than the line: the configuration is
    stored in the markers, so that a line saved to JSON after it has been
    configured generates the same particles once loaded again.

    Parameters
    ----------
    line : xtrack.Line
        Line containing, or receiving, the Touschek markers.
    """

    def __init__(self, line):
        self._line = line

    @property
    def line(self):
        """
        Line of the Touschek simulation.

        Returns
        -------
        line : xtrack.Line
            Line bound to this generator.
        """
        return self._line

    @property
    def marker_names(self):
        """
        Names of the Touschek markers of the line.

        Returns
        -------
        names : list of str
            Names of the :class:`xfields.TouschekMarker` elements, in line
            order.
        """
        element_dict = self.line.element_dict
        names = [nn for nn in self.line.element_names
                 if isinstance(element_dict[nn], TouschekMarker)]
        if len(set(names)) != len(names):
            repeated = sorted({nn for nn in names if names.count(nn) > 1})
            raise ValueError(
                f"Touschek markers {repeated} appear more than once in the "
                "line: each marker must be a distinct element.")
        return names

    def _markers(self, at=None):
        """
        Validated names of the selected markers, in line order.

        Parameters
        ----------
        at : str, sequence of str, or None
            Selected markers. ``None`` selects all the markers of the line.

        Returns
        -------
        names : list of str
            Names of the selected markers, in line order.
        """
        names = self.marker_names
        if at is None:
            if len(names) == 0:
                raise ValueError(
                    "The line does not contain any TouschekMarker. Install "
                    "them with `line.xfields.touschek.install_markers(...)`.")
            return names

        selected = [at] if isinstance(at, str) else list(at)
        if len(selected) == 0:
            raise ValueError("No Touschek marker selected.")
        position = {nn: ii for ii, nn in enumerate(names)}
        for nn in selected:
            if not isinstance(nn, str):
                raise TypeError(
                    "Touschek markers are selected by name "
                    f"(got {type(nn).__name__}).")
            if nn in position:
                continue
            if nn in set(self.line.element_names):
                raise TypeError(
                    f"Element '{nn}' is not a TouschekMarker "
                    f"(got {type(self.line.element_dict[nn]).__name__}).")
            raise ValueError(f"Element '{nn}' is not present in the line.")

        return sorted(set(selected), key=position.__getitem__)
