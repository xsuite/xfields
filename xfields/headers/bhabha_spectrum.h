#ifndef XFIELDS_BHABHA_SPECTRUM_H
#define XFIELDS_BHABHA_SPECTRUM_H

/************************************************************************************/
/* Subroutines for the generation of the equivalent virtual photons for one primary */
/* Adapted from GUINEA-PIG                                                          */
/************************************************************************************/

/*gpufun*/
double rndm_sincos(LocalParticle *part, double *theta)
{
    const double twopi=2.0*PI;
    double r1;
    r1 = RandomUniform_generate(part);
    *theta = cos(twopi*r1);
    if (r1 > 0.5)
        return sqrt(1.0- *theta * *theta);
    else
        return -sqrt(1.0- *theta * *theta);
}


/*gpufun*/
float requiv(LocalParticle *part, const double e_primary, const double compt_x_min){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/guinea_pig.c#L1205
    ----
    Returns the number of equivalent virtual (macro)photons for a single primary macroparticle of the beam.
    */
    const double emass2=MELECTRON_GEV*MELECTRON_GEV;
    double lnxmin, s4, xmin, lns4, r_photons, n_photons;

    s4 = e_primary*e_primary;  // [GeV^2] mandelstam s divided by 4: s/4
    xmin = compt_x_min * emass2 / s4;  // [1] virtual photon's energy fraction compared to primary energy

    if (xmin>=1.0) return 0.0;

    lnxmin=-log(xmin);
    lns4 = log(s4 / (MELECTRON_GEV*MELECTRON_GEV));

    r_photons = .00232461*lnxmin*(lnxmin + lns4);  // alpha/pi = 0.00232461

    // account for noninteger photon by randomly emitting n+1 sometimes
    n_photons = (int)floor(r_photons);
    r_photons -= n_photons;
    if(RandomUniform_generate(part) < r_photons) n_photons += 1.0;

    return n_photons;
}


/*gpufun*/
void mequiv (LocalParticle *part,
             const double e_primary,    // [GeV] other beam slice energy
             const double compt_x_min,  // [1] scaling factor in the minimum energy cutoff
             double *xmin,              // [1] cutoff x from where to integrate number density
             double *e_photon,          // [GeV] single equivalent virtual photon energy
             double *q2,                // [GeV^2] single equivalent virtual photon virtuality
             double *one_m_x            // [1] 1 - x
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/guinea_pig.c#L1248
    ----
    Sets the energy and virtuality of a single virtual photon.
    */
    const double emass2=MELECTRON_GEV*MELECTRON_GEV;
    double s4, q2max, q2min, lnx, x, lnxmin, lns4;
  
    s4 = e_primary*e_primary;  // [GeV^2] mandelstam s divided by 4: s/4
    *xmin = compt_x_min * emass2 / s4;  // [1] virtual photon's energy fraction compared to primary energy
  
    lnxmin = -log(*xmin);
    lns4 = log(s4 / (MELECTRON_GEV*MELECTRON_GEV));
  
    // set energy fraction and virtuality boundaires
    if(RandomUniform_generate(part) < lnxmin / (lnxmin + lns4)){
        lnx   = -sqrt(RandomUniform_generate(part)) * lnxmin;
        x     = exp(lnx);
        q2min = x*x*emass2;
        q2max = emass2;
    }
    else{
        lnx   = -RandomUniform_generate(part) * lnxmin;
        x     = exp(lnx);
        q2min = emass2;
        q2max = s4;
    }
  
    // set virtual photon energy and virtuality
    if((1.0 + (1.0 - x) * (1.0 - x)) * 0.5 < RandomUniform_generate(part)){
        *e_photon = 0.0;
        *q2  = 0.0;
    }
    else{
        *e_photon = e_primary * x;
        *q2 = q2min * pow(q2max / q2min, RandomUniform_generate(part));
    }
  
    if (*q2 * (1.0 - x) < x*x*emass2) *e_photon = 0.0;
    *one_m_x = 1.0 - x;
  
    return;
  
}

/*gpufun*/
double compt_tot(double s  // [GeV^2] center of mass energy of the macroparticle - virtual photon Compton scattering
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/background.c#L350
    and
    https://doi.org/10.1016/0167-5087(84)90128-5
    ----
    Total Compton cross section.
    */
    const double sig0 = PI*RE*RE;
    double xi, xp, xip, ln, sigc, x_compt;

    x_compt = s/(MELECTRON_GEV*MELECTRON_GEV);
    xi  = 1.0 / x_compt; 
    xp  = 1.0 + x_compt;
    xip = 1.0 / xp;
    ln  = log(xp);

    sigc = 2.0*sig0*xi*((1.0 - xi*(4.0 + 8.0*xi))*ln + 0.5 + 8.0*xi - 0.5*xip*xip);

    return sigc;  // [m^2]
}


/*gpufun*/
double compt_diff(double y,              // [1] virtual photon energy fraction compared to primary energy
                  const double x_compt   // [1] normalized cm energy of virtual photon - electron compton scattering
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/background.c#L364
    ----
    Differential Compton energy spectrum.
    */
    double r, yp;
    yp = 1.0 - y;
    r = y / (x_compt*yp);
    return 1.0/yp + yp - 4.0*r*(1.0 - r);  // [m^2]
}


/*gpufun*/
double compt_int(double y,             // [1] virtual photon energy fraction compared to primary energy
                 const double x_compt  // [1] normalized cm energy of virtual photon - electron compton scattering
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/background.c#L373
    ----
    Integrated Compton energy spectrum.
    */
    double yp, lny, xi;

    yp = 1.0 - y;
    xi = 1.0 / x_compt;
    lny = -log(yp);

    return lny*(1.0-4.0*xi-8.0*xi*xi) + y - y*y*0.5 + 4.0*y*xi + 4.0*y*xi*xi + 4.0/(x_compt*x_compt*yp);  // [m^2]
}


/***********************************************************************************/
/* Subroutines for Compton scattering of the virtual photon and the opposite slice */
/* Adapted from GUINEA-PIG                                                         */
/***********************************************************************************/

/*gpufun*/
void equal_newton(
                  double xmin,              // left boundary of domain
                  double xmax,              // right boundary of domain
                  double y,                 // random sample on y axis
                  double *x,                // initial guess for root (inverse CDF sampling)
                  const double x_compt      // [1] normalized cm energy of virtual photon - electron compton scattering

){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/background.c#L23
    ----
    Uses the Newton-Raphson root finding method for inverse CDF sampling of the given spectrum.
    */
    double eps=1e-6;
    double ytry,xtry;
    int i=0;
    xtry=*x;
    ytry=compt_int(xtry, x_compt);
    while (fabs(ytry-y)>(fabs(ytry)+fabs(y))*eps
           && (xmax-xmin)>eps) {
        i++;
        xtry-=(ytry-y)/compt_diff(xtry, x_compt);
        if ((xtry>=xmax)||(xtry<=xmin)) {
            xtry=0.5*(xmax+xmin);
        }
        ytry=compt_int(xtry, x_compt);
        if(ytry<y) {
            xmin=xtry;
        }
        else {
            xmax=xtry;
        }
    }
    *x=xtry;
}



/*gpufun*/
double compt_select(LocalParticle *part,
                    double s  // [GeV^2] center of mass energy of the macroparticle - virtual photon Compton scattering
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/background.c#L397
    ----
    Given an input CM energy, returns a generated scattered Compton photon energy (y = hbar*omega/E0)
    s: center of mass energy of electron-virtual photon system, s=(x+1)*(m_e*c**2)**2
    returns a randomly sampled scattered photon energy fraction drawn from the Compton energy ditribution
    */
    double cmin, cmax, c, y, ym;
    const double x_compt = s / (MELECTRON_GEV*MELECTRON_GEV);

    //x_compt = x;
    ym = x_compt / (x_compt + 1.0);  // [1] y_max, maximum energy fraction (right edge of domain on x axis of the spectrum)
    cmin = compt_int(0.0, x_compt);  // [m^2] min of range of CDF 
    cmax = compt_int(ym, x_compt);   // [m^2] max of range of CDF

    y = RandomUniform_generate(part);
    c = cmin + (cmax - cmin)*y;  // [m^2] this is the random sample in the inverse CDF
    y *= ym;                     // [1] this is the initial guess on the domain axis 
    equal_newton(0.0, ym, c, &y, x_compt);  // find root: y=sigma^-1(c) i.e. do the inverse CDF sampling
  
    return y;  // [1]
}


/*gpufun*/
void compt_do(LocalParticle *part, BeamBeamBiGaussian3DRecordData bhabha_record, RecordIndex bhabha_table_index, BhabhaTableData bhabha_table,
              double e_photon,           // [GeV] single equivalent virtual photon energy before Compton scattering
              const double compt_x_min,  // [1] scaling factor in the minimum energy cutoff
              double q2,                 // [GeV^2] single equivalent virtual photon virtuality
              double x_photon, double y_photon, double z_photon,  // [m] (boosted) coords of the virtual photon
              double vx_photon,          // [1] transverse x momentum component of virtual photon (vx = dx/ds/p0)
              double vy_photon,          // [1] transverse y momentum component of virtual photon (vy = dy/ds/p0)
              double vzeta_photon,       // [1] zeta momentum component of virtual photon
              double wgt,                // [m^-2] int. luminosity
              double *vx,                // [1] normalized momenta of the primary macroparticle
              double *vy,
              double *vzeta, 
              double q0             // [e] charge of primary macroparticle
){

    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/background.c#L420
    and
    https://doi.org/10.1016/0167-5087(84)90128-5
    and
    https://doi.org/10.1016/0031-9163(63)90351-2
    ----
    Compton scatter one vitrual macrophoton on one macroparticle. Potentially emit a real Bhabha photon. 
    */

    double e_ref = LocalParticle_get_energy0(part)*1e-9;  // [GeV] reference beam energy
    double e_primary = (LocalParticle_get_energy0(part) + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part))*1e-9;  // [GeV] 
    double beta_ref = LocalParticle_get_beta0(part);         // [1] reference beta
    double part_per_mpart = LocalParticle_get_weight(part);  // [e] num charges represented by 1 macropart

    int n, i;                        // [1]
    double tmp, scal, s, x, y;       // [m^2, m^2, GeV^2, 1, 1]
    double theta_g, theta_e, phi_e;  // [rad] scattering angles
    double e_e_prime, px_e_prime, py_e_prime, pzeta_e_prime, ps_e_prime, pt_e_prime;  // [GeV, 1, 1, 1, 1, 1] scattered primary
    double e_photon_prime, px_photon_prime, py_photon_prime, pzeta_photon_prime;      // [GeV, 1, 1, 1] scattered (real) Compton photon
    double e_loss_primary;            // [GeV] energy lost from one emission
    double e_loss_primary_tot = 0.0;  // [GeV] total energy lost by the macroparticle

    double eps = 0.0;                 // 1e-5 in guinea
    const double compt_scale = 1;       // [1]
    const double compt_emax = 200;      // [GeV] upper cutoff from guineapig
    const double pair_ecut = 0.005;  // [GeV] lower cutoff from guineapig
    double r1, r2;  // [1] uniform random numbers

    if (q2 > MELECTRON_GEV*MELECTRON_GEV) return;  // global upper cut on virtuality; eliminates "constant" part of q2 spectrum i.e. the hadronic virtual photons

    s = 4.0*e_photon*e_primary;  // approximated center of mass energy of primary - photon Compton scattering
    if (q2 > s) return;          // event specific upper cut on virtuality; check against max allowed virtuality to be absorbed in the event

    if (s < compt_x_min * MELECTRON_GEV*MELECTRON_GEV*4.0) return;  // event specific lower cut on x; check against user defined compt_x_min

    tmp = compt_tot(s)*wgt*compt_scale;   // [1] this determines the number of real Compton scattering events
    n = (int)floor(tmp)+1;                // [1] round up e.g. tmp=5.4 will mean n=6 events
    scal = tmp/n;                         // [1] fractional part of event count
    x = s/(MELECTRON_GEV*MELECTRON_GEV);  // [1]

    for (i=0; i<n; i++) {
      y = compt_select(part, s);  // [1] draw compton scattered photon energy

      if (scal > eps){
        double one_m_y = 1 - y;  // + e_photon / e_primary;

        e_e_prime = one_m_y * e_primary;  // + e_photon; neglected. [GeV] scattered electron energy: E_e' = E_e + E_p - E_p' but E_p is negligible compared to other terms
        e_photon_prime = y*e_primary;     // [GeV] scattered photon energy

        // get scattered angle for photon and beam primary
        theta_g = MELECTRON_GEV / e_primary * sqrt((x - (x + 1.0) * y) / y);
        theta_e = theta_g * e_photon_prime / (e_primary - e_photon_prime);  // + e_photon; neglected

        // save computations for tracking: energies below are lost anyways, energies above compt_emax have negligible e loss from bhabha
        if ((e_e_prime < compt_emax) && (e_e_prime > pair_ecut)) {

          // compute scattered primary momenta
          // adjust magnitude
          px_e_prime = *vx * e_e_prime;  // / e_primary;
          py_e_prime = *vy * e_e_prime;  // / e_primary;

          // adjust direction
          pt_e_prime = theta_e * e_e_prime;  // [1] transverse azimuthal momentum
          phi_e = 2.0 * PI * RandomUniform_generate(part);
          px_e_prime  += pt_e_prime * sin(phi_e);
          py_e_prime  += pt_e_prime * cos(phi_e);
          ps_e_prime   = sqrt(e_e_prime*e_e_prime - px_e_prime*px_e_prime - py_e_prime*py_e_prime - MELECTRON_GEV*MELECTRON_GEV);  // [1] longitudinal momentum

          pzeta_e_prime = (e_e_prime - e_ref) / (beta_ref * beta_ref * e_ref);

          // compute scattered photon momenta from momentum conservation
          px_photon_prime    = (*vx  + vx_photon) * e_primary - px_e_prime;
          py_photon_prime    = (*vy  + vy_photon) * e_primary - py_e_prime;
          pzeta_photon_prime = (*vzeta + vzeta_photon) * e_primary - pzeta_e_prime;

          // account for the event weight
          r1 = RandomUniform_generate(part);
          if (r1 < scal) {

            if (bhabha_record){
              // Get a slot in the record (this is thread safe)
              int64_t i_slot = RecordIndex_get_slot(bhabha_table_index);
              // The returned slot id is negative if record is NULL or if record is full
              if (i_slot>=0){
                  BhabhaTableData_set_particle_id(   bhabha_table, i_slot, LocalParticle_get_particle_id(part));
                  BhabhaTableData_set_at_turn(       bhabha_table, i_slot, LocalParticle_get_at_turn(part));
                  BhabhaTableData_set_at_element(    bhabha_table, i_slot, LocalParticle_get_at_element(part));
                  BhabhaTableData_set_primary_energy(bhabha_table, i_slot, e_primary);
                  BhabhaTableData_set_photon_id(     bhabha_table, i_slot, n);
                  BhabhaTableData_set_photon_x(      bhabha_table, i_slot, x_photon);
                  BhabhaTableData_set_photon_y(      bhabha_table, i_slot, y_photon);
                  BhabhaTableData_set_photon_z(      bhabha_table, i_slot, z_photon);
                  BhabhaTableData_set_photon_energy( bhabha_table, i_slot, e_photon_prime*1e9);
                  BhabhaTableData_set_photon_px(     bhabha_table, i_slot, px_photon_prime);
                  BhabhaTableData_set_photon_py(     bhabha_table, i_slot, py_photon_prime);
                  BhabhaTableData_set_photon_pzeta(  bhabha_table, i_slot, pzeta_photon_prime);
                  BhabhaTableData_set_primary_scattering_angle(bhabha_table, i_slot, theta_e);
                  BhabhaTableData_set_photon_scattering_angle( bhabha_table, i_slot, theta_g);
              }
            }

            // scattered photons are real so affect a macropart with a probability
            r2 = RandomUniform_generate(part);
            if (r2 < 1.0 / part_per_mpart){
              e_loss_primary = e_e_prime - e_primary;  // [GeV], <0, loss from a single photon emission

              if (e_loss_primary == 0.0){
                printf("0 energy loss: %g", e_loss_primary);
              }else{
                //printf("[%d] lost %g [GeV]\n", (int)part->ipart, e_loss_primary);
                if (-1.0 * e_loss_primary >= e_primary){  // macropart dies
                  LocalParticle_set_state(part, XT_LOST_ALL_E_IN_SYNRAD); // used to flag this kind of loss
                  return;
                }else{  // macropart doesnt die
                  *vx    = px_e_prime / e_e_prime;
                  *vy    = py_e_prime / e_e_prime;
                  *vzeta = pzeta_e_prime / e_e_prime;
                  e_primary += e_loss_primary;
                  LocalParticle_update_pzeta(part, *vzeta);  // changes energy vars
                  LocalParticle_add_to_energy(part, e_loss_primary*1e9, 0);  // changes pzeta
                }
              }
            }
          }

        }
      }
    }
}
#endif /* XFIELDS_BHABHA_SPECTRUM_H */
