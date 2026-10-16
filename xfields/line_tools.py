# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #


class XfieldsLineAPI:
    """
    Xfields-specific helpers associated with an xtrack Line.

    The public methods use flat prefixes (for example ``spacecharge_*``) so
    the reference guide can document one facade class without introducing
    many nested API containers. The Touschek tools are the exception: they
    are grouped in :attr:`touschek`, a :class:`xfields.TouschekGenerator`
    bound to the line.
    """

    def __init__(self, line):
        self._line = line

    @property
    def line(self):
        return self._line

    @property
    def touschek(self):
        """
        Touschek scattering simulation of the line.

        Returns
        -------
        touschek : xfields.TouschekGenerator
            Generator bound to the line, to install and configure the
            Touschek markers, generate the scattered particles and analyse
            them.
        """
        from .touschek.generator import TouschekGenerator

        return TouschekGenerator(self.line)

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
