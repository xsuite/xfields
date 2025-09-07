# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xtrack as xt
import xfields as xf

import numpy as np
from scipy.integrate import quad
from scipy.special import i0
from scipy.constants import physical_constants

ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
C_LIGHT_VACUUM = physical_constants['speed of light in vacuum'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]

class TouschekCalculator:
    def __init__(self, manager):
        self.manager = manager
        self.twiss = None

    def _compute_piwinski_integral(self, tm, B1, B2):
        """
        Compute Piwinski integral for Touschek scattering rate calculation.

        Reference:
            A. Piwinski,
            "The Touschek Effect in Strong Focusing Storage Rings",
            arXiv:physics/9903034, 1999.
            URL: https://arxiv.org/abs/physics/9903034
        """
        km = np.arctan(np.sqrt(tm))

        def int_piwinski(k, km, B1, B2):
            t = np.tan(k) ** 2
            tm = np.tan(km) ** 2
            fact = (
                (2*t + 1)**2 * (t/tm / (1+t) - 1) / t + t - np.sqrt(t*tm * (1 + t))
                - (2 + 1 / (2*t)) * np.log(t/tm / (1+t))
            )
            if B2 * t < 500:
                intp = fact * np.exp(-B1*t) * i0(B2*t) * np.sqrt(1+t)
            else:
                intp = (
                    fact
                    * np.exp(B2*t - B1*t)
                    / np.sqrt(2*np.pi * B2*t)
                    * np.sqrt(1+t)
                )
            return intp

        args = (km, B1, B2)
        val, _ =  quad(
            int_piwinski,
            km,
            np.pi / 2,
            args=args,
            epsabs=1e-16,
            epsrel=1e-12
        )

        return val

    def _compute_piwinski_scattering_rate(self, element):
        """
        Compute Piwinski Touschek scattering rate.

        Reference:
            A. Piwinski,
            "The Touschek Effect in Strong Focusing Storage Rings",
            arXiv:physics/9903034, 1999.
            URL: https://arxiv.org/abs/physics/9903034
        """
        p0c = self.manager.particle_ref.p0c[0]
        bunch_population = self.manager.bunch_population
        gemitt_x = self.manager.gemitt_x
        gemitt_y = self.manager.gemitt_y
        alfx = self.twiss['alfx', element]
        betx = self.twiss['betx', element]
        alfy = self.twiss['alfy', element]
        bety = self.twiss['bety', element]
        sigma_z = self.manager.sigma_z
        sigma_delta = self.manager.sigma_delta
        delta = self.twiss['delta', element]
        dx = self.twiss['dx', element]
        dpx = self.twiss['dpx', element]
        dxt = alfx * dx + betx * dpx # dxt: dx tilde
        dy = self.twiss['dy', element]
        dpy = self.twiss['dpy', element]
        dyt = alfy * dy + bety * dpy # dyt: dy tilde

        deltaN = self.manager.momentum_aperture.at[element, "deltaN"]
        deltaP = self.manager.momentum_aperture.at[element, "deltaP"]

        sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
        sigma_x = np.sqrt(gemitt_x * betx + dx**2 * sigma_delta**2) # Horizontal beam size

        sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size
        sigma_y = np.sqrt(gemitt_y * bety + dy**2 * sigma_delta**2) # Vertical beam size

        sigma_h = (sigma_delta**-2 + (dx**2 + dxt**2)/sigmab_x**2 + (dy**2 + dyt**2)/sigmab_y**2)**(-0.5)

        p = p0c * (1 + delta)
        gamma = np.sqrt(1 + p**2 / ELECTRON_MASS_EV**2)
        beta = np.sqrt(1 - gamma**-2)

        B1 = betx**2 / (2 * beta**2 * gamma**2 * sigmab_x**2) * (1 - sigma_h**2 * dxt**2 / sigmab_x**2) \
             + bety**2 / (2 * beta**2 * gamma**2 * sigmab_y**2) * (1 - sigma_h**2 * dyt**2 / sigmab_y**2)

        B2 = np.sqrt(B1**2 - betx**2 * bety**2 * sigma_h**2 / (beta**4 * gamma**4 * sigmab_x**4 * sigmab_y**4 * sigma_delta**2) \
                             * (sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2))

        tmN = beta**2 * (deltaN**2)
        tmP = beta**2 * (deltaP**2)

        piwinski_integralN = self._compute_piwinski_integral(tmN, B1, B2)
        piwinski_integralP = self._compute_piwinski_integral(tmP, B1, B2)

        rateN = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * bunch_population**2 \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)) \
                * 2 * np.sqrt(np.pi * (B1**2 - B2**2)) * piwinski_integralN

        rateP = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * bunch_population**2 \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)) \
                * 2 * np.sqrt(np.pi * (B1**2 - B2**2)) * piwinski_integralP

        rate = (rateN + rateP) / 2

        return rate

    def _compute_integrated_piwinski_rates(self):
        """
        Integrate the Piwinski Touschek scattering rate along the line using
        the trapezoidal rule, between successive TouschekScattering elements.

        For each TouschekScattering element, the method stores the integrated
        rate per bunch over the preceding section of the line. This per-bunch
        rate is later used to assign the correct weights to Touschek-scattered
        particles at the corresponding element.
        """
        line = self.manager.line
        tab = line.get_table()
        T_rev0 = float(self.twiss.T_rev0)

        # Indexes of the TouschekScatterings
        ii_t = [ii for ii, nn in enumerate(tab.name[:-1]) if isinstance(line[nn], xf.TouschekScattering)]

        integrated = 0.0
        s0 = 0.0
        r0 = self._compute_piwinski_scattering_rate(tab.name[0])

        s_before = s0
        rate_before = r0
        ii_current = 0

        for ii, nn in enumerate(tab.name):
            s = tab.rows[nn].s[0]
            ds = s - s_before
            if ds > 0.0:
                rate = self._compute_piwinski_scattering_rate(nn)
                integrated += 0.5 * (rate_before + rate) * ds
                s_before = s
                rate_before = rate

            if ii_current < len(ii_t) and ii == ii_t[ii_current]:
                # divide by c and by T_rev0 --> per-bunch rate
                integrated_piwinski_rate = integrated / C_LIGHT_VACUUM / T_rev0
                elem = line[nn] # xf.TouschekScattering
                elem._configure(_integrated_piwinski_rate=integrated_piwinski_rate)
                integrated = 0.0
                ii_current += 1
                if ii_current >= len(ii_t):
                    break

class TouschekManager:
    def __init__(self, line, momentum_aperture, nemitt_x=None, nemitt_y=None,
                 sigma_z=None, sigma_delta=None, bunch_population=None,
                 n_simulated=None, gemitt_x=None, gemitt_y=None,
                 momentum_aperture_scale=0.85, ignored_portion=0.01,
                 seed=1997, nx=3, ny=3, nz=3, **kwargs):

        # Input validation
        if line is None:
            raise ValueError("`line` is required.")
        if not hasattr(line, "particle_ref"):
            raise ValueError("`line` must have a `particle_ref`.")
        if momentum_aperture is None:
            raise ValueError("`momentum_aperture` is required.")
        if sigma_z is None:
            raise ValueError("`sigma_z` is required.")
        if sigma_delta is None:
            raise ValueError("`sigma_delta` is required.")
        if bunch_population is None:
            raise ValueError("`bunch_population` is required.")
        if n_simulated is None:
            raise ValueError("`n_simulated` is required.")

        # Momentum aperture validation
        required_cols = {"s", "name", "deltaN", "deltaP"}
        if not hasattr(momentum_aperture, "columns") or not hasattr(momentum_aperture, "__getitem__"):
            raise TypeError("`momentum_aperture` must be a DataFrame-like object with columns "
                            "'s', 'name', 'deltaN', 'deltaP'.")
        missing = required_cols - set(momentum_aperture.columns)
        if missing:
            raise ValueError(f"`momentum_aperture` missing columns: {sorted(missing)}")

        for col in ("s", "deltaN", "deltaP"):
            try:
                vals = momentum_aperture[col].astype(float)
            except Exception:
                raise TypeError(f"`{col}` column must be numeric (cannot coerce to float).")
            if not vals.notna().all():
                bad = list(vals.index[~vals.notna()][:5])
                raise ValueError(f"`{col}` contains NaN at rows {bad}.")
            if (abs(vals) == float("inf")).any():
                bad = list(vals.index[(abs(vals) == float("inf"))][:5])
                raise ValueError(f"`{col}` contains inf at rows {bad}.")

        self.line = line
        self.particle_ref = line.particle_ref

        momentum_aperture = momentum_aperture.copy()
        momentum_aperture['deltaN'] *= momentum_aperture_scale
        momentum_aperture['deltaP'] *= momentum_aperture_scale
        self.momentum_aperture= momentum_aperture.set_index("name")

        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.bunch_population = bunch_population
        self.n_simulated = n_simulated
        self.momentum_aperture_scale = momentum_aperture_scale
        self.ignored_portion = ignored_portion
        self.seed = seed
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Emittance validation
        nemitt_given = nemitt_x is not None and nemitt_y is not None
        gemitt_given = gemitt_x is not None and gemitt_y is not None

        if nemitt_given and gemitt_given:
            raise ValueError("Provide either normalized emittances (nemitt_x, nemitt_y) "
                             "OR geometric emittances (gemitt_x, gemitt_y), not both.")
        if not (nemitt_given or gemitt_given):
            raise ValueError("You must provide either both normalized emittances (nemitt_x, nemitt_y) "
                             "OR both geometric emittances (gemitt_x, gemitt_y).")

        if nemitt_given:
            beta0 = line.particle_ref.beta0[0]
            gamma0 = line.particle_ref.gamma0[0]
            self.gemitt_x = nemitt_x / (beta0 * gamma0)
            self.gemitt_y = nemitt_y / (beta0 * gamma0)
        else:
            self.gemitt_x = gemitt_x
            self.gemitt_y = gemitt_y

        self.kwargs = kwargs

        self.touschek = TouschekCalculator(self)

        # Check that the line contains TouschekScatterings
        tab = self.line.get_table()
        try:
            has = "TouschekScattering" in set(np.unique(tab.element_type))
        except Exception:
            has = "TouschekScattering" in set(getattr(tab, "element_type", []))
        if not has:
            raise ValueError("The line does not contain any TouschekScattering. "
                             "Please add them before initializing the TouschekManager.")


    def initialise_touschek(self, element=None):
        line = self.line
        tab = line.get_table()

        twiss_method = self.kwargs.get("method", "6d")
        twiss = self.line.twiss(method=twiss_method)
        # Pass the twiss to the TouschekCalculator
        self.touschek.twiss = twiss

        self.touschek._compute_integrated_piwinski_rates()

        # Helper to config all fields to a single TouschekScattering
        def _config(nn):
            s = tab.rows[nn].s[0]
            alfx = twiss["alfx", nn]; betx = twiss["betx", nn]
            alfy = twiss["alfy", nn]; bety = twiss["bety", nn]
            dx   = twiss["dx",   nn]; dpx = twiss["dpx",  nn]
            dy   = twiss["dy",   nn]; dpy = twiss["dpy",  nn]
            dN = self.momentum_aperture.at[nn, "deltaN"]
            dP = self.momentum_aperture.at[nn, "deltaP"]

            piwinski_rate = self.touschek._compute_piwinski_scattering_rate(nn)

            elem = line[nn] # xf.TouschekScattering

            elem._configure(
                _s=s,
                _particle_ref=self.particle_ref,
                _bunch_population=self.bunch_population,
                _gemitt_x=self.gemitt_x,
                _gemitt_y=self.gemitt_y,
                _alfx=alfx, _betx=betx,
                _alfy=alfy, _bety=bety,
                _dx=dx, _dpx=dpx,
                _dy=dy, _dpy=dpy,
                _deltaN=dN, _deltaP=dP,
                _sigma_z=self.sigma_z,
                _sigma_delta=self.sigma_delta,
                _n_simulated=self.n_simulated,
                _nx=self.nx, _ny=self.ny, _nz=self.nz,
                _ignored_portion=self.ignored_portion,
                piwinski_rate=piwinski_rate,
                _seed=self.seed, _inhibit_permute=0
            )

        if element is None:
            for nn in tab.name[:-1]: # Avoid the last tab.name which is _end_point
                if isinstance(line[nn], xf.TouschekScattering):
                    _config(nn)
        else:
            if not isinstance(element, str):
                raise TypeError(f"`element` must be a string (got {type(element).__name__}).")
            if element not in set(tab.name):
                raise ValueError(
                    f"`element='{element}'` is not present in the line provided to the TouschekManager."
                )
            if not isinstance(line[element], xf.TouschekScattering):
                raise TypeError(
                    f"`line['{element}']` is not a TouschekScattering (got {type(line[element]).__name__})."
                )
            _config(element)