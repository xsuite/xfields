# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from .beam_elements.touschek import TouschekScattering
from .touschek.study import TouschekStudy


class XfieldsLineAPI:
    """
    Xfields-specific helpers associated with an xtrack Line.

    The public methods use flat prefixes (for example ``touschek_*``) so the
    reference guide can document one facade class without introducing many
    nested API containers.
    """

    def __init__(self, line):
        self._line = line

    @property
    def line(self):
        return self._line

    def spacecharge_install_frozen(self, **kwargs):
        """
        Install frozen space-charge elements in the line.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to
            :func:`xfields.install_spacecharge_frozen`. The ``line`` argument
            is supplied by this facade.

        Returns
        -------
        object
            Value returned by :func:`xfields.install_spacecharge_frozen`.
        """
        from .config_tools import install_spacecharge_frozen

        return install_spacecharge_frozen(line=self.line, **kwargs)

    def spacecharge_replace_with_quasi_frozen(self, **kwargs):
        """
        Replace frozen space-charge elements with quasi-frozen elements.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to
            :func:`xfields.replace_spacecharge_with_quasi_frozen`. The
            ``line`` argument is supplied by this facade.

        Returns
        -------
        object
            Value returned by
            :func:`xfields.replace_spacecharge_with_quasi_frozen`.
        """
        from .config_tools import replace_spacecharge_with_quasi_frozen

        return replace_spacecharge_with_quasi_frozen(line=self.line, **kwargs)

    def spacecharge_replace_with_pic(self, **kwargs):
        """
        Replace frozen space-charge elements with PIC space-charge elements.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to
            :func:`xfields.replace_spacecharge_with_PIC`. The ``line``
            argument is supplied by this facade.

        Returns
        -------
        object
            Value returned by :func:`xfields.replace_spacecharge_with_PIC`.
        """
        from .config_tools import replace_spacecharge_with_PIC

        return replace_spacecharge_with_PIC(line=self.line, **kwargs)

    def ibs_configure(self, element=None, update_every=None, **kwargs):
        """
        Configure an intrabeam-scattering kick element in the line.

        Parameters
        ----------
        element : xfields.ibs.IBSKick, optional
            IBS kick element to insert before configuration. If omitted, the
            line must already contain exactly one IBS kick element.
        update_every : int
            Number of turns between kick-coefficient recomputations.
        **kwargs
            Keyword arguments forwarded to
            :func:`xfields.ibs.configure_intrabeam_scattering`. When
            ``element`` is provided, these are passed to ``line.insert`` by the
            underlying implementation.

        Returns
        -------
        object
            Value returned by
            :func:`xfields.ibs.configure_intrabeam_scattering`.
        """
        from .ibs import configure_intrabeam_scattering

        return configure_intrabeam_scattering(
            self.line, element=element, update_every=update_every, **kwargs)

    def _touschek_elements(self, elements=None):
        line = self.line

        if elements is None:
            tab = line.get_table()
            elements = [
                nn for nn in tab.name[:-1]
                if isinstance(line[nn], TouschekScattering)
            ]
        elif isinstance(elements, str):
            elements = [elements]
        else:
            elements = list(elements)

        if len(elements) == 0:
            raise ValueError(
                "No TouschekScattering elements found. Add them to the line "
                "before using the Touschek facade."
            )

        for nn in elements:
            if nn not in line.element_names:
                raise ValueError(f"Element '{nn}' is not present in the line.")
            if not isinstance(line[nn], TouschekScattering):
                raise TypeError(
                    f"Element '{nn}' is not a TouschekScattering "
                    f"(got {type(line[nn]).__name__})."
                )

        return elements

    def touschek_configure(self, *, local_momentum_acceptance, twiss=None,
                           elements=None, **kwargs):
        """
        Configure Touschek scattering elements in the line.

        Returns a configured :class:`xfields.TouschekStudy`, on which users can
        call ``local_rates()``, ``generate_particles()``, or ``run()``.
        """
        elements = self._touschek_elements(elements)

        study = TouschekStudy(
            self.line,
            elements=elements,
            twiss=twiss,
            local_momentum_acceptance=local_momentum_acceptance,
            **kwargs,
        )
        study.initialise_touschek()
        return study
