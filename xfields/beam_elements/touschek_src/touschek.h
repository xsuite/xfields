/*
 *  touschek.h — Touschek scattering kernel (C99, header-only)
 *
 *  Overview
 *  --------
 *  Header-only C99 implementation of the Monte Carlo Touschek scattering
 *  routine for Xsuite/xfields. The physics and selection logic follow the
 *  Elegant implementation by A. Xiao and M. Borland (PRSTAB 13, 074201, 2010),
 *  with a simplified API suitable for xobjects; SDDS I/O is omitted.
 *
 *  Provenance (portions adapted from)
 *  ----------------------------------
 *  - Elegant: src/touschekScatter.c  (A. Xiao, M. Borland)
 *  - SDDS Toolkit utilities: mdbmth/drand.c, mdbmth/dlaran.c
 *  - LAPACK DLARAN (48-bit LCG RNG core)
 *
 *  RNG compatibility
 *  -----------------
 *  RNG streams are chosen to reproduce Elegant’s sequences and are stored in
 *  an explicit TouschekRNGState xobject.
 *  See: xfields/xfields/headers/elegant_rng.h
 *  (uses TouschekRNGState_seed, touschek_random_1_elegant,
 *  touschek_random_4, touschek_randomize_order, and DLARAN).
 *
 *  Summary of modifications vs Elegant
 *  -----------------------------------
 *  - Converted to a header-only C99 kernel; simplified API; no SDDS I/O.
 *  - Integrated with xobjects; clarified slope (xp, yp) vs momentum usage.
 *  - Minor safety/cleanup (bounds checks, allocations, comments).
 *  - Physics and selection logic preserved.
 *
  *  References
 *  ----------
 *  - M. Borland, “elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,”
 *    APS LS-287 (2000).
 *  - A. Xiao and M. Borland, "Monte Carlo simulation of Touschek effect",
 *    Phys. Rev. ST Accel. Beams **13**, 074201 (2010).
 */
#ifndef XTRACK_TOUSCHEK_H
#define XTRACK_TOUSCHEK_H

#include "xtrack/headers/track.h"
#include "xtrack/random/random_src/uniform_accurate.h"
#include "xfields/headers/constants.h"
#include "xfields/headers/elegant_rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/*gpufun*/
void TouschekScattering_track_local_particle(TouschekScatteringData el, LocalParticle* part0) {
  (void)el; (void)part0;
  return;
}

/* Adapted from Elegant touschekScatter.c: selectPartGauss (logic unchanged). */
void selectPartGauss(double *p1, double *p2,
                     double *dens1, double *dens2,
                     const double *ran1,
                     const double *range,
                     const double *alfa,
                     const double *beta,
                     const double *disp,
                     const double *gemitt) {
  int i;
  double U[3], V1[3], V2[3], densa[3], densb[3];

  /* Select random particle coordinates in normalized phase space */
  for (i = 0; i < 3; i++) {
    U[i] = (ran1[i] - 0.5) * range[i] * sqrt(gemitt[i]);
    V1[i] = (ran1[i + 3] - 0.5) * range[i] * sqrt(gemitt[i]);
    V2[i] = (ran1[i + 6] - 0.5) * range[i] * sqrt(gemitt[i]);
    densa[i] = exp(-0.5 * (U[i] * U[i] + V1[i] * V1[i]) / gemitt[i]);
  }
  densb[2] = exp(-0.5 * (U[2] * U[2] + V2[2] * V2[2]) / gemitt[2]);
  /* Transform particle coordinates from normalized to real phase space */
  for (i = 0; i < 3; i++) {
    p1[i] = p2[i] = sqrt(beta[i]) * U[i];
    p1[i + 3] = (V1[i] - alfa[i] * U[i]) / sqrt(beta[i]);
    p2[i + 3] = (V2[i] - alfa[i] * U[i]) / sqrt(beta[i]);
  }
  /* Dispersion correction */
  p1[0] = p1[0] + p1[5] * disp[0];
  p1[1] = p1[1] + p1[5] * disp[1];
  p1[3] = p1[3] + p1[5] * disp[2];
  p1[4] = p1[4] + p1[5] * disp[3];

  p2[0] = p1[0] - p2[5] * disp[0];
  p2[1] = p1[1] - p2[5] * disp[1];
  U[0] = p2[0] / sqrt(beta[0]);
  U[1] = p2[1] / sqrt(beta[1]);
  p2[3] = (V2[0] - alfa[0] * U[0]) / sqrt(beta[0]);
  p2[4] = (V2[1] - alfa[1] * U[1]) / sqrt(beta[1]);
  densb[0] = exp(-0.5 * (U[0] * U[0] + V2[0] * V2[0]) / gemitt[0]);
  densb[1] = exp(-0.5 * (U[1] * U[1] + V2[1] * V2[1]) / gemitt[1]);

  p2[0] = p1[0];
  p2[1] = p1[1];
  p2[3] = p2[3] + p2[5] * disp[2];
  p2[4] = p2[4] + p2[5] * disp[3];

  *dens1 = densa[0] * densa[1] * densa[2];
  *dens2 = densb[0] * densb[1] * densb[2];

  return;
}

/* From Elegant touschekScatter.c: bunch2cm */
void bunch2cm(double *p1, double *p2, double *q, double *beta, double *gamma) {
  double pp1, pp2, e1, e2, ee;
  int i;
  double bb, betap1, factor;

  pp1 = 0.0;
  pp2 = 0.0;
  for (i = 3; i < 6; i++) {
    pp1 = pp1 + POW2(p1[i]);
    pp2 = pp2 + POW2(p2[i]);
  }
  e1 = sqrt(MELECTRON_EV * MELECTRON_EV + pp1);
  e2 = sqrt(MELECTRON_EV * MELECTRON_EV + pp2);
  ee = e1 + e2;

  betap1 = 0.0;
  bb = 0.0;
  for (i = 0; i < 3; i++) {
    beta[i] = (p1[i + 3] + p2[i + 3]) / ee;
    betap1 = betap1 + beta[i] * p1[i + 3];
    bb = bb + beta[i] * beta[i];
  }

  *gamma = 1. / sqrt(1. - bb);
  factor = ((*gamma) - 1.) * betap1 / bb;

  for (i = 0; i < 3; i++) {
    q[i] = p1[i + 3] + factor * beta[i] - (*gamma) * e1 * beta[i];
  }

  return;
}


/* Rotate scattered p in c.o.m system */
/* From Elegant touschekScatter.c: eulertrans*/
void eulertrans(double *v0, double theta, double phi, double *v1, double *v) {
  double th, ph, s1, s2, c1, c2;
  double x0, y0, z0;

  *v = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  th = acos(v0[2] / (*v));
  ph = atan2(v0[1], v0[0]);

  s1 = sin(th);
  s2 = sin(ph);
  c1 = cos(th);
  c2 = cos(ph);

  x0 = cos(theta);
  y0 = sin(theta) * cos(phi);
  z0 = sin(theta) * sin(phi);

  v1[0] = (*v) * (s1 * c2 * x0 - s2 * y0 - c1 * c2 * z0);
  v1[1] = (*v) * (s1 * s2 * x0 + c2 * y0 - c1 * s2 * z0);
  v1[2] = (*v) * (c1 * x0 + s1 * z0);

  return;
}

/* From Elegant touschekScatter.c: cm2bunch*/
void cm2bunch(double *p1, double *p2, double *q, double *beta, double *gamma) {
  int i;
  double pq, e, betaq, bb, factor;

  pq = 0.0;
  for (i = 0; i < 3; i++) {
    pq = pq + q[i] * q[i];
  }

  e = sqrt(MELECTRON_EV * MELECTRON_EV + pq);

  betaq = 0.0;
  bb = 0.0;
  for (i = 0; i < 3; i++) {
    betaq = betaq + beta[i] * q[i];
    bb = bb + beta[i] * beta[i];
  }

  factor = ((*gamma) - 1) * betaq / bb;
  for (i = 0; i < 3; i++) {
    p1[i + 3] = q[i] + (*gamma) * beta[i] * e + factor * beta[i];
    p2[i + 3] = -q[i] + (*gamma) * beta[i] * e - factor * beta[i];
  }

  return;
}

/* From Elegant touschekScatter.c: moeller */
double moeller(double beta0, double theta) {
  double cross;
  double beta2, st2;

  beta2 = beta0 * beta0;
  st2 = POW2(sin(theta));

  cross = (1. - beta2) * (POW2(1. + 1. / beta2) * (4. / st2 / st2 - 3. / st2) + 1. + 4. / st2);

  return cross;
}

/* From Elegant touschekScatter.c: pickPart */
void pickPart(double *weight, long *index, long start, long end,
              long *iTotal, double *wTotal, double weight_limit, double weight_ave) {
  long i, i1, i2, N;
  double w1, w2;
  long *index1, *index2;
  double *weight1, *weight2;

  i1 = i2 = 0;
  w1 = w2 = 0.;
  N = end - start;
  if (N < 3)
    return; /* scattered particles normally appear in pair */
  index2 = (long *)malloc(sizeof(long) * N);
  weight2 = (double *)malloc(sizeof(double) * N);
  index1 = (long *)malloc(sizeof(long) * N);
  weight1 = (double *)malloc(sizeof(double) * N);

  for (i = start; i < end; i++) {
    if (weight[i] > weight_ave) {
      weight2[i2] = weight[i];
      index2[i2++] = index[i];
      w2 += weight[i];
    } else {
      weight1[i1] = weight[i];
      index1[i1++] = index[i];
      w1 += weight[i];
    }
  }
  if ((w2 + (*wTotal)) > weight_limit) {
    weight_ave = w2 / (double)i2;
    for (i = 0; i < i2; i++) {
      index[start + i] = index2[i];
      weight[start + i] = weight2[i];
    }
    free(weight1);
    free(index1);
    free(weight2);
    free(index2);
    pickPart(weight, index, start, start + i2,
             iTotal, wTotal, weight_limit, weight_ave);
    return;
  }

  *iTotal += i2;
  *wTotal += w2;
  weight_ave = w1 / (double)i1;
  for (i = 0; i < i2; i++) {
    index[start + i] = index2[i];
    weight[start + i] = weight2[i];
  }
  for (i = 0; i < i1; i++) {
    index[start + i2 + i] = index1[i];
    weight[start + i2 + i] = weight1[i];
  }
  free(weight1);
  free(index1);
  free(weight2);
  free(index2);
  pickPart(weight, index, i2 + start, end,
           iTotal, wTotal, weight_limit, weight_ave);
  return;
}

static long touschek_xtrack_randomize_order(
        char *ptr, long size, long length, LocalParticle* part0) {
    if (!ptr || size <= 0) return 0;
    if (length < 2) return 1;

    RANDOMIZATION_HOLDER *rh =
        (RANDOMIZATION_HOLDER*)malloc(sizeof(*rh) * (size_t)length);
    if (!rh) return 0;

    for (long i = 0; i < length; i++) {
        rh[i].buffer = malloc((size_t)size);
        if (!rh[i].buffer) {
            for (long k = 0; k < i; k++) free(rh[k].buffer);
            free(rh);
            return 0;
        }
        memcpy(rh[i].buffer, ptr + i * size, (size_t)size);
        rh[i].random_value = RandomUniformAccurate_generate(part0);
    }

    qsort((void*)rh, (size_t)length, sizeof(*rh), randomizeOrderCmp);

    for (long i = 0; i < length; i++) {
        memcpy(ptr + i * size, rh[i].buffer, (size_t)size);
        free(rh[i].buffer);
    }
    free(rh);
    return 1;
}

/* Adapted from Elegant touschekScatter.c: TouschekDistribution (logic unchanged) */
void TouschekScatter(TouschekScatteringData el,
                     LocalParticle* part0,
                     double* x_out,
                     double* px_out,
                     double* y_out,
                     double* py_out,
                     double* zeta_out,
                     double* delta_out,
                     double* theta_out,
                     double* weight_out,
                     double* totalMCRate_out,
                     int64_t* n_selected_out,
                     TouschekRNGStateData rng_state){

    const double p0c   = TouschekScatteringData_get_p0c(el);
    const double bunch_intensity = TouschekScatteringData_get_bunch_intensity(el);
    const double gemitt_x = TouschekScatteringData_get_gemitt_x(el);
    const double gemitt_y = TouschekScatteringData_get_gemitt_y(el);
    const double alfx  = TouschekScatteringData_get_alfx(el);
    const double betx  = TouschekScatteringData_get_betx(el);
    const double alfy  = TouschekScatteringData_get_alfy(el);
    const double bety  = TouschekScatteringData_get_bety(el);
    const double dx    = TouschekScatteringData_get_dx(el);
    const double dpx   = TouschekScatteringData_get_dpx(el);
    const double dy    = TouschekScatteringData_get_dy(el);
    const double dpy   = TouschekScatteringData_get_dpy(el);
    const double delta_neg = TouschekScatteringData_get_delta_neg(el);
    const double delta_pos = TouschekScatteringData_get_delta_pos(el);
    const double sigma_z   = TouschekScatteringData_get_sigma_z(el);
    const double sigma_delta = TouschekScatteringData_get_sigma_delta(el);
    const double n_simulated = TouschekScatteringData_get_n_simulated(el);
    const double nx = TouschekScatteringData_get_nx(el);
    const double ny = TouschekScatteringData_get_ny(el);
    const double nz = TouschekScatteringData_get_nz(el);
    const double theta_min = TouschekScatteringData_get_theta_min(el);
    const double theta_max = TouschekScatteringData_get_theta_max(el);
    const double ignoredPortion = TouschekScatteringData_get_ignored_portion(el);
    const double integrated_piwinski_rate = TouschekScatteringData_get_integrated_piwinski_rate(el);
    const int64_t rng_source = TouschekScatteringData_get_rng_source(el);

    long i, j, total_event, simuCount, iTotal;
    double ran1[11];

    long   *index  = NULL; 
    double *weight = (double*)malloc(sizeof(double) * n_simulated); 

    const double twissAlpha[3] = { alfx, alfy, 0.0 };

    const double bets = sigma_z/sigma_delta;
    const double twissBeta[3] = { betx, bety, bets };

    const double twissDisp[4] = { dx, dy, dpx, dpy };

    const double gemitt_z = sigma_z*sigma_delta;
    const double gemitt[3] = { gemitt_x, gemitt_y, gemitt_z };
    const double range[3] = { 2.0*nx, 2.0*ny, 2.0*nz };

    double totalWeight = 0.0;
    double totalMCRate = 0.0;

    double pTemp[6], p1[6], p2[6], dens1, dens2;
    double theta, phi, qa[3], qb[3], beta[3], qabs, gamma;
    double beta0, cross, temp;

    double weight_limit, weight_ave, wTotal;

    const double sigxyz = sqrt(twissBeta[0]*gemitt[0]) * sqrt(twissBeta[1]*gemitt[1]) * sigma_z;
    temp = POW2(bunch_intensity) * POW2(PI) * POW2(RADIUS_ELECTRON) * C_LIGHT / 4.;
    double factor = temp * pow(range[0], 3.0) * pow(range[1], 3.0) * pow(range[2], 3.0) / pow(2 * PI, 6.0) / sigxyz;

    double *xtemp      = (double*)malloc(sizeof(double) * n_simulated);
    double *pxtemp     = (double*)malloc(sizeof(double) * n_simulated);
    double *ytemp      = (double*)malloc(sizeof(double) * n_simulated);
    double *pytemp     = (double*)malloc(sizeof(double) * n_simulated);
    double *zetatemp   = (double*)malloc(sizeof(double) * n_simulated);
    double *deltatemp  = (double*)malloc(sizeof(double) * n_simulated);
    double *thetatemp  = (double*)malloc(sizeof(double) * n_simulated);

    i = 0;
    j = 0;
    total_event = 0;
    simuCount = 0;

    while (1) {
        if (i >= n_simulated)
          break;

        /* Select 11 random numbers, then mix them. */

        // These 11 random numbers are assigned to:
        // particle 1 (p1) as: { x, y, px, py, zeta, delta}
        // particle 2 (p2) as: { -, -, px, py, -, delta}
        // scattering angles in the cm frame: theta and phi

        // In ELEGANT the 11 random numbers are assigned to:
        // particle 1 (p1) as: { x, y, xp, yp, zeta, delta}
        // particle 2 (p2) as: { -, -, xp, yp, -, delta}
        // scattering angles in the cm frame: theta and phi

        // NOTE: ELEGANT uses slopes xp=dx/ds, yp=dy/ds instead of the normalized momentum components px=Px/p0c, py=Py/p0c
        for (j = 0; j < 11; j++) {
          if (rng_source == 1) {
            ran1[j] = RandomUniformAccurate_generate(part0);
          } else {
            ran1[j] = touschek_random_1_elegant(rng_state);
          }
        }
        if (rng_source == 1) {
          touschek_xtrack_randomize_order((char*)ran1, sizeof(ran1[0]), 11, part0);
        } else {
          touschek_randomize_order((char*)ran1, sizeof(ran1[0]), 11, rng_state); // like ELEGANT
        }

        total_event++;

        selectPartGauss(p1, p2, &dens1, &dens2, ran1, range, twissAlpha, twissBeta, twissDisp, gemitt);

        if (!dens1 || !dens2) {
          continue;
        }
        /* Here ELEGANT changes from slopes to momentum components */
        // Since we use already the normalized momentum components {px, py} instead of the slopes {xp, yp}
        // here we just unormalize the momentum components: Px=px*p0c, Py=py*p0c
        for (j = 3; j < 5; j++) {
          p1[j] *= p0c;
          p2[j] *= p0c;
        }
        // p1[5] = (p1[5] + 1) * p0c;
        // p2[5] = (p2[5] + 1) * p0c;
        // Use exact formula to compute p1[5]=Pz1 and p2[5]=Pz2
        p1[5] = sqrt(POW2(p0c)*POW2(1. + p1[5]) - POW2(p1[3]) - POW2(p1[4]));
        p2[5] = sqrt(POW2(p0c)*POW2(1. + p2[5]) - POW2(p2[3]) - POW2(p2[4]));

        bunch2cm(p1, p2, qa, beta, &gamma);

        theta = theta_min + (theta_max - theta_min) * ran1[9];
        phi = ran1[10] * PI;

        temp = dens1 * dens2 * sin(theta);
        eulertrans(qa, theta, phi, qb, &qabs);
        cm2bunch(p1, p2, qb, beta, &gamma);
        // p1[5] = (p1[5] - p0c) / p0c;
        // p2[5] = (p2[5] - p0c) / p0c;
        // Use exact formula to compute p1[5]=delta1 and p2[5]=delta2
        p1[5] = (sqrt(POW2(p1[3]) + POW2(p1[4]) + POW2(p1[5])) - p0c) / p0c;
        p2[5] = (sqrt(POW2(p2[3]) + POW2(p2[4]) + POW2(p2[5])) - p0c) / p0c;

        if (p1[5] > p2[5]) {
          for (j = 0; j < 6; j++) {
            pTemp[j] = p2[j];
            p2[j] = p1[j];
            p1[j] = pTemp[j];
          }
        }

        if (p1[5] < delta_neg || p2[5] > delta_pos) {
          beta0 = qabs / sqrt(qabs * qabs + MELECTRON_EV * MELECTRON_EV);
          cross = moeller(beta0, theta);
          temp *= cross * beta0 / gamma / gamma;

          if (p1[5] < delta_neg) {
            totalWeight += temp;
            p1[3] /= p0c;
            p1[4] /= p0c;
            simuCount++;

            xtemp[i] = p1[0];
            pxtemp[i] = p1[3];
            ytemp[i] = p1[1];
            pytemp[i] = p1[4];
            zetatemp[i] = p1[2];
            deltatemp[i] = p1[5];
            thetatemp[i] = theta;
            weight[i] = temp;
            i++;
          }

          if (i >= n_simulated)
            break;

          if (p2[5] > delta_pos) {
            totalWeight += temp;
            p2[3] /= p0c;
            p2[4] /= p0c;
            simuCount++;

            xtemp[i] = p2[0];
            pxtemp[i] = p2[3];
            ytemp[i] = p2[1];
            pytemp[i] = p2[4];
            zetatemp[i] = p2[2];
            deltatemp[i] = p2[5];
            thetatemp[i] = theta;
            weight[i] = temp;
            i++;
          }
        }
      }
      factor = factor / (double)(total_event);
      totalMCRate = totalWeight * factor;

      /* Pick tracking particles from the simulated scattered particles */
      index = (long *)malloc(sizeof(long) * simuCount);
      for (i = 0; i < simuCount; i++) {
          index[i] = i;
      }

      if (ignoredPortion <= 1e-9) {
          iTotal = simuCount;
          wTotal = totalWeight;
          for (long k = 0; k < iTotal; ++k) {
              x_out[k]      = xtemp[k];
              px_out[k]     = pxtemp[k];
              y_out[k]      = ytemp[k];
              py_out[k]     = pytemp[k];
              zeta_out[k]   = zetatemp[k];
              delta_out[k]  = deltatemp[k];
              theta_out[k]  = thetatemp[k];
              weight_out[k] = weight[k];
          }
      } else {
          iTotal = 0;
          wTotal = 0.;
          weight_limit = totalWeight * (1 - ignoredPortion);
          weight_ave   = totalWeight / simuCount;

          pickPart(weight, index, 0, simuCount,
                  &iTotal, &wTotal, weight_limit, weight_ave);

          for (long k = 0; k < iTotal; ++k) {
              long src = index[k];
              x_out[k]      = xtemp[src];
              px_out[k]     = pxtemp[src];
              y_out[k]      = ytemp[src];
              py_out[k]     = pytemp[src];
              zeta_out[k]   = zetatemp[src];
              delta_out[k]  = deltatemp[src];
              theta_out[k]  = thetatemp[src];
              weight_out[k] = weight[src];
          }
      }

      // printf("Total number of random numbers used: %ld\n", total_event * 11);
      // fflush(stdout);
      if (total_event * 11 > (long)2e9) {
        printf("The total number of random numbers used exceeded 2e9. Use smaller n_simulated or smaller delta.");
        fflush(stdout);
      }

      // if (simuCount == 0) {
      //   printf("It appears that the Touschek lifetime is extremely wrong and there is no need to perform Touschek simulation.\n If you think this is a wrong statement, send input to developers for evaluation.\n");
      // }

      if (total_event / simuCount > 20) {
        if (nx < 5 || ny < 5) {
          printf("The Touschek scattering rate is low. Please use >=5 sigma beam for better simulation.");
          fflush(stdout);
        } else {
          printf("The Touschek scattering rate is very low. Please ignore the rate from Monte Carlo simulation. Use Piwinski rate only.");
          fflush(stdout);
        }
      }

      printf("%ld of %ld particles selected for tracking\n", iTotal, simuCount);
      fflush(stdout);

      // Update weight_out to match ELEGANT
      for (long k = 0; k < iTotal; ++k) {
          weight_out[k] *= (factor / totalMCRate) * integrated_piwinski_rate;
      }

      *n_selected_out  = iTotal;
      *totalMCRate_out = totalMCRate;

      free(index);
      free(thetatemp);
      free(weight);
      free(xtemp);
      free(pxtemp);
      free(ytemp);
      free(pytemp);
      free(zetatemp);
      free(deltatemp);
}

#endif // XTRACK_TOUSCHEK_H
