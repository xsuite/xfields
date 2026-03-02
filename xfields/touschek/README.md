# Monte Carlo simulation of Touschek scattering

The Monte Carlo Touschek scattering routine in this package is based on the implementation by A. Xiao and M. Borland developed for [ELEGANT](https://github.com/rtsoliday/elegant).

If you publish results obtained with this routine, please cite:

- M. Borland, “elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,” APS LS-287 (2000).

- A. Xiao and M. Borland, “Monte Carlo simulation of Touschek effect,” *Phys. Rev. ST Accel. Beams* **13**, 074201 (2010).  
  DOI: 10.1103/PhysRevSTAB.13.074201

## RNG compatibility with ELEGANT

For reproducibility with ELEGANT, this routine uses the same RNG conventions; see `xfields/xfields/headers/elegant_rng.h`.

The uniform RNG core is LAPACK’s DLARAN (48-bit LCG, modified BSD license) as used by Elegant.
See `xfields/third_party/LAPACK/LICENSE`.

## Third-party notice and licenses

This routine contains code adapted from **ELEGANT** and **SDDS**.  
© 2002 The University of Chicago; © 2002 The Regents of the University of California.  
Distributed subject to their Software License Agreements. See:
- `third_party/elegant/LICENSE`
- `third_party/SDDS/LICENSE`