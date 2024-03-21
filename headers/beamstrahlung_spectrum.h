#ifndef XFIELDS_BEAMSTRAHLUNG_SPECTRUM_H
#define XFIELDS_BEAMSTRAHLUNG_SPECTRUM_H

#define SQRT3 1.732050807568877
#define ALPHA_EM 0.0072973525693


/*gpufun*/
int beamstrahlung_0(LocalParticle *part,
             double energy,     // [eV] primary electron energy
             double dz,         // [m] z slice half width
             double rho_inv,    // [1/m] inverse local bending radius, changes after each photon emission
             double* e_photon,  // [GeV] emitted BS photon energy
             double* ecrit      // [GeV] critical energy of emitted BS photon
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/guinea_pig.c#L1894
    and
    K. Yokoya: A COMPUTER SIMULATION CODE FOR THE BEAM-BEAM INTERACTION IN LINEAR COLLIDERS
    https://inis.iaea.org/collection/NCLCollectionStore/_Public/18/033/18033162.pdf?r=1
    ----
    return 0: no photon
    return 1: emit 1 photon, energy stored in e_photon
    */

    // constants for approximating beamstrahlung spectrum
    const double g1_a[5] = {1.0, -0.8432885317, 0.1835132767, -0.0527949659, 0.0156489316};
    const double g2_a[5] = {0.4999456517, -0.5853467515, 0.3657833336, -0.0695055284, 0.019180386};
    const double g1_b[7] = {2.066603927, -0.5718025331, 0.04243170587, -0.9691386396, 5.651947051, -0.6903991322, 1.0};
    const double g2_b[7] = {1.8852203645, -0.5176616313, 0.03812218492, -0.49158806, 6.1800441958, -0.6524469236, 1.0};
    const double g1_c[4] = {1.0174394594, 0.5831679349, 0.9949036186, 1.0};
    const double g2_c[4] = {0.2847316689, 0.58306846, 0.3915531539, 1.0};

    double c1 = 1.5*HBAR_GEVS / pow(MELECTRON_GEV, 3.0) * C_LIGHT;  // [c^4/Gev^2] 2.22e-6 = 1.5*hbar*cst.c/e0**3

    double xcrit = c1 * pow(energy*1e-9, 2.0) * rho_inv; // [1] ecrit/E magnitude of quantum correction, in guineapig: xcrit (C) = xi (doc) = upsbar (C++)
    (*ecrit) = xcrit * energy*1e-9; // [GeV]
    //double omega_crit = (*ecrit)/HBAR_GEVS;  // [1/s] = 1.5 * gamma**3 * cst.c / rho
    //double upsilon = 2.0/3.0 * (*ecrit) / (energy*1e-9);  // [1] beamstrahlung parameter for single macropart
    double p0 = 25.4 * energy*1e-9 * dz * rho_inv;  // [1]  Fr * dz, specific for 1 macropart
 
    // eliminate region A in p0*g-v plane (=normalize with p0 = reject 1-p0 (p0<1) fraction of cases = y axis of p0*g-v plane is now spanning 0--p0=1
    if (RandomUniform_generate(part) > p0){return 0;}

    // 2 random numbers to calculate g(v, xcrit)
    double p = RandomUniform_generate(part);  // if this is 1, then it corresponds to p0 on original p0*g-v plane
    double v;
    while((v=RandomUniform_generate(part))==0); // draw a nonzero random number, variable of the beamstrahlung spectrum
    double v2 = v*v;
    double v3 = v2*v;
    double y = v3 / (1.0 - v3);
    double denom = 1.0 - ( 1.0 - xcrit ) * v3;

    // calculate beamstrahlung spectrum coefficients, depending the value of y
    double g1, g2;
    if (y <= 1.54){
        g1 = pow(y, -2.0/3.0) * (g1_a[0] + g1_a[1]*pow(y, 2.0/3.0) + g1_a[2]*pow(y, 2.0) + g1_a[3]*pow(y, 10.0/3.0) + g1_a[4]*pow(y, 4.0));
        g2 = pow(y, -2.0/3.0) * (g2_a[0] + g2_a[1]*pow(y, 4.0/3.0) + g2_a[2]*pow(y, 2.0) + g2_a[3]*pow(y, 10.0/3.0) + g2_a[4]*pow(y, 4.0));
    }else if (y <= 4.48){
        g1 = ( g1_b[0] + g1_b[1]*y + g1_b[2]*pow(y, 2.0) ) / ( g1_b[3] + g1_b[4]*y + g1_b[5]*pow(y, 2.0) + g1_b[6]*pow(y, 3.0) );
        g2 = ( g2_b[0] + g2_b[1]*y + g2_b[2]*pow(y, 2.0) ) / ( g2_b[3] + g2_b[4]*y + g2_b[5]*pow(y, 2.0) + g2_b[6]*pow(y, 3.0) );
    }else if (y <= 165.0){
        g1 = exp(-y)/sqrt(y) * ( g1_c[0] + g1_c[1]*y ) / ( g1_c[2] + g1_c[3]*y );
        g2 = exp(-y)/sqrt(y) * ( g2_c[0] + g2_c[1]*y ) / ( g2_c[2] + g2_c[3]*y );
    }else{
        // no radiation, y too high
        return 0;
    }
        
    // g normalized (g(v=0, xcrit)=1), g(v, xcrit) gives the no. of emitted photons in a given delta v interval
    double g = v2 / pow(denom, 2.0) * ( g1 + ( pow(xcrit, 2.0) * pow(y, 2.0) ) / ( 1.0 + xcrit * y ) * g2 );  // g (w.o. normalization above) splits the unit rectangle p0*g-v to A,B,C regions
   
    // region C (emit photon) if p<p0*g, region B (no photon) if p>=p0*g, p0=1 bc. of normalization above
    if (p<g){
        (*e_photon) = (*ecrit) * v3 / denom;
        return 1;
    }else{
        (*e_photon) = 0.0;
        return 0;
    }
}


/*gpufun*/
double beamstrahlung_avg(LocalParticle *part, BeamBeamBiGaussian3DRecordData beamstrahlung_record, RecordIndex beamstrahlung_table_index, BeamstrahlungTableData beamstrahlung_table,
        const double n_bb, // [1] strong slice bunch intensity
        const double sigma_x, const double sigma_y, const double sigma_z  // [m] unboosted strong slice RMS
){
    /*
    Based on:
    K. Yokoya: Beam-Beam Phenomena In Linear Colliders
    https://doi.org/10.1007/3-540-55250-2_37
    ----
    n_bb and sigma_z are scaled with the (same) slice weights
    n_avg ~ n_bb -> 1/num_slices less photons per mp in 1 slice
    delta_avg ~ n_bb^2/sigma_z -> 1/num_slices less rel. E loss per mp in 1 slice
    e_photon_avg = delta_avg / n_avg -> avg. photon energy is the same in 1 slice
    */

    // beam properties
    const double m0             = LocalParticle_get_mass0(part); // [eV/c] beam particle mass
    const double initial_energy = LocalParticle_get_energy0(part) + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part); // [eV]
    const double gamma          = initial_energy / m0; // [1] 

    // constants
    const double r  = pow(QELEM, 2.0)/(4.0* PI * EPSILON_0 * MELECTRON_KG * pow(C_LIGHT, 2.0));      // [m] electron radius
    const double c1 = 2.59 * ( 5.0/ 6.0) * (    r*r) / REDUCED_COMPTON_WAVELENGTH_ELECTRON;          // [m]
    const double c2 =  1.2 * (25.0/36.0) * (r*r*r*r) / REDUCED_COMPTON_WAVELENGTH_ELECTRON * 137.0;  // [m^3]

    // compute averaged quantities
    double n_avg        = c1 * n_bb/(sigma_x + sigma_y);  // [1] avg. number of emitted photons from 1 macroparticle in one slice collision
    double delta_avg    = c2 * gamma/sigma_z * (n_bb/(sigma_x + sigma_y))*(n_bb/(sigma_x + sigma_y));  // [1] avg. rel. E loss for 1 macroparticle in one slice collision
    double U_BS         = delta_avg*initial_energy;  // [eV] avg. energy loss per macropart in one slice collision
    double u_avg        = delta_avg/n_avg;           // [1] avg. rel. photon energy normalized to initial electron energy
    double e_photon_avg = u_avg*initial_energy;      // [eV] avg. photon energy
    LocalParticle_add_to_energy(part, -U_BS, 0);
    double energy_loss = -U_BS;  // <0

    if (beamstrahlung_record){
        // Get a slot in the record (this is thread safe)
        int64_t i_slot = RecordIndex_get_slot(beamstrahlung_table_index);
        // The returned slot id is negative if record is NULL or if record is full
        if (i_slot>=0){
            BeamstrahlungTableData_set_particle_id(   beamstrahlung_table, i_slot, LocalParticle_get_particle_id(part));
            BeamstrahlungTableData_set_at_turn(       beamstrahlung_table, i_slot, LocalParticle_get_at_turn(part));
            BeamstrahlungTableData_set_at_element(    beamstrahlung_table, i_slot, LocalParticle_get_at_element(part));
            BeamstrahlungTableData_set_photon_energy( beamstrahlung_table, i_slot, e_photon_avg);
            BeamstrahlungTableData_set_delta_avg(     beamstrahlung_table, i_slot, delta_avg);
            BeamstrahlungTableData_set_n_avg(         beamstrahlung_table, i_slot, n_avg);
            BeamstrahlungTableData_set_primary_energy(beamstrahlung_table, i_slot, initial_energy);
        }
    }

    return energy_loss;
}


/*gpufun*/
double beamstrahlung(LocalParticle *part, BeamBeamBiGaussian3DRecordData beamstrahlung_record, RecordIndex beamstrahlung_table_index, BeamstrahlungTableData beamstrahlung_table,
     	double Fr,  // [1] radial force sqrt[(px' - px)^2 + (py' - py)^2]/Dt, Dt=1
	double dz   // [m] z slice half width: step between 2 slices ((z_max - z_min) / 2)
){
    /*
    Based on:
    GUINEA-PIG
    https://gitlab.cern.ch/clic-software/guinea-pig-legacy/-/blob/master/guinea_pig.c#L1962
    */

    // beam properties
    const double m0       = LocalParticle_get_mass0(part); // particle mass [eV/c]
    double initial_energy = LocalParticle_get_energy0(part) + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part); // [eV]
    double energy         = initial_energy;  // [eV]
    double gamma          = energy / m0;     // [1]

    // single macroparticle trajectory 
    double rho_inv  = Fr / dz;  // [1/m] macropart inverse bending radius
    double tmp      = 25.4 * energy*1e-9 * dz * rho_inv;  // [1] p0, specific for 1 macropart, 1e-9 to convert [eV] to [GeV]
    int max_photons = (int)(tmp*10.0)+1;  // [1]
    dz /= (double)max_photons;  // photons are emitted uniformly in space along dz (between 2 slice interactions)

    // BS photon counter and BS photon energy buffer
    int j = 0;
    double e_photon_array[1000];
    for (int i=0; i<max_photons; i++){
   
        double e_photon, ecrit;  // [GeV] BS photon energy and critical energy
        if (beamstrahlung_0(part, energy, dz, rho_inv, &e_photon, &ecrit)){  // see if quantum photon can be emitted
            e_photon_array[j] = e_photon;  // [GeV]
           
            if (beamstrahlung_record){
                // Get a slot in the record (this is thread safe)
                int64_t i_slot = RecordIndex_get_slot(beamstrahlung_table_index);
                // The returned slot id is negative if record is NULL or if record is full
                if (i_slot>=0){
                    BeamstrahlungTableData_set_particle_id(           beamstrahlung_table, i_slot, LocalParticle_get_particle_id(part));
                    BeamstrahlungTableData_set_at_turn(               beamstrahlung_table, i_slot, LocalParticle_get_at_turn(part));
                    BeamstrahlungTableData_set_at_element(            beamstrahlung_table, i_slot, LocalParticle_get_at_element(part));
                    BeamstrahlungTableData_set_photon_id(             beamstrahlung_table, i_slot, j);
                    BeamstrahlungTableData_set_photon_energy(         beamstrahlung_table, i_slot, e_photon*1e9);
                    BeamstrahlungTableData_set_photon_critical_energy(beamstrahlung_table, i_slot, ecrit*1e9);
                    BeamstrahlungTableData_set_primary_energy(        beamstrahlung_table, i_slot, energy);
                    BeamstrahlungTableData_set_rho_inv(               beamstrahlung_table, i_slot, rho_inv);
                }
            }

            // update bending radius, primary macropart energy and gamma
            rho_inv *= energy/(energy - e_photon*1e9);
            energy  -= e_photon*1e9;
            gamma   *= (energy - e_photon*1e9)/energy;

            // some error handling
            if (e_photon_array[j]<=0.0){
                printf("photon emitted with negative energy: E_photon=%g [eV], E_macropart=%g [eV], photon ID: %d, max_photons: %d\n", e_photon*1e9, energy, j, max_photons);
            }

            // increment photon counter
            j++;

            // break loop and flag macroparticle as dead
            if (j>=1000){
                printf("[%d] too many photons produced by one particle (photon ID: %d), Fr: %.12e, dz: %.12e\n", (int)part->ipart, j, Fr, dz);
                //exit(-1);  // doesnt work on GPU
                LocalParticle_set_state(part, XF_TOO_MANY_PHOTONS); // used to flag this kind of loss
                break;
            }

        }
    }

    // update primary macroparticle energy
    if (energy == 0.0){
        LocalParticle_set_state(part, XT_LOST_ALL_E_IN_SYNRAD); // used to flag this kind of loss
    }else{
       LocalParticle_add_to_energy(part, energy-initial_energy, 0);
    }
    double energy_loss = energy - initial_energy;  // <0

    return energy_loss;
}

#endif /* XFIELDS_BEAMSTRAHLUNG_SPECTRUM_H */
