#ifndef XFIELDS_BEAMSTRAHLUNG_SPECTRUM_H
#define XFIELDS_BEAMSTRAHLUNG_SPECTRUM_H

#define SQRT3 1.732050807568877
#define ALPHA_EM 0.0072973525693

#if !defined(dump_path)
    #define dump_path "/Users/pkicsiny/phd/cern/xsuite/outputs/n34_xsuite" 
#endif

/*gpufun*/
int synrad_0(LocalParticle *part,
             double energy,  // [eV]
             double dz,  // [m]
             double rho_inv,  // [1/m] changes after each photon emission
             double* e_photon,  // [GeV] 
             double* ecrit){  // [GeV] 

    /*
    Based on:
    https://inis.iaea.org/collection/NCLCollectionStore/_Public/18/033/18033162.pdf?r=1
    return 0: no photon
    return 1: emit 1 photon, energy stored in photon_energy
    */
    // constants for approximating synrad spectrum
    const double g1_a[5] = {1.0, -0.8432885317, 0.1835132767, -0.0527949659, 0.0156489316};
    const double g2_a[5] = {0.4999456517, -0.5853467515, 0.3657833336, -0.0695055284, 0.019180386};
    const double g1_b[7] = {2.066603927, -0.5718025331, 0.04243170587, -0.9691386396, 5.651947051, -0.6903991322, 1.0};
    const double g2_b[7] = {1.8852203645, -0.5176616313, 0.03812218492, -0.49158806, 6.1800441958, -0.6524469236, 1.0};
    const double g1_c[4] = {1.0174394594, 0.5831679349, 0.9949036186, 1.0};
    const double g2_c[4] = {0.2847316689, 0.58306846, 0.3915531539, 1.0};

    double c1 = 1.5*HBAR_GEVS / pow(MELECTRON_GEVPERC, 3.0) * C_LIGHT;  // [c^4/Gev^2] 2.22e-6 = 1.5*hbar*cst.c/e0**3
    double xcrit = c1 * pow(energy*1e-9, 2.0) * rho_inv;  // [1] ecrit/E magnitude of quantum correction, in guineapig: xcrit (C) = ξ (doc) = upsbar (C++)
    (*ecrit) = xcrit * energy*1e-9;  // [GeV]
    double omega_crit = (*ecrit)/HBAR_GEVS;  // [1/s] = 1.5 * gamma**3 * cst.c / rho
    double upsilon = 2.0/3.0 * (*ecrit) / (energy*1e-9);  // [1] beamstrahlung parameter for single macropart
    double p0 = 25.4 * energy*1e-9 * dz * rho_inv;  // [1]  Fr * dz, specific for 1 macropart
 
    // eliminate region A in p0*g-v plane (=normalize with p0 = reject 1-p0 (p0<1) fraction of cases = y axis of p0*g-v plane is now spanning 0--p0=1
    if (LocalParticle_generate_random_double(part) > p0){return 0;}

    // 2 random numbers to calculate g(v, xcrit)
    double p = LocalParticle_generate_random_double(part);  // if this is 1, then it corresponds to p0 on original p0*g-v plane
    double v;
    while((v=LocalParticle_generate_random_double(part))==0); // draw a nonzero random number, variable of the beamstrahlung spectrum
    double v2 = v*v;
    double v3 = v2*v;
    double y = v3 / (1.0 - v3);
    double denom = 1.0 - ( 1.0 - xcrit ) * v3;

    // calculate synrad spectrum coefficients, depending the value of y
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
        
    // g normalized (g(v=0, xcrit)=1=p0), g(v, xcrit) gives the no. of emitted photons in a fiven delta v interval
    double g = v2 / pow(denom, 2.0) * ( g1 + ( pow(xcrit, 2.0) * pow(y, 2.0) ) / ( 1.0 + xcrit * y ) * g2 );  // g (w.o. normalization above) splits the unit rectangle p0*g-v to A,B,C regions
   
    char dump_file[1024];
    sprintf(dump_file, "%s/%s", dump_path, "xsuite_synrad0.txt");
    //FILE *f1 = fopen(dump_file, "a");
    //fprintf(f1, "%.10e %.10e %.10e %.10e %.10e\n", c1, xcrit, (*ecrit), p0, g);
    //fclose(f1);

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
double synrad_avg(LocalParticle *part, const double n_bb, double sigma_x, double sigma_y, double sigma_z){

    double r              = pow(QELEM, 2.0)/(4.0* PI * EPSILON_0 * MELECTRON_KG * pow(C_LIGHT, 2.0));  // [m] electron radius
    const double c1       = 2.59*(5.0/6.0)*(r*r)/(REDUCED_COMPTON);
    const double c2       =  1.2*(25.0/36.0)*(r*r*r*r)/(REDUCED_COMPTON)*137.0;
    const double m0       = LocalParticle_get_mass0(part); // particle mass [eV/c]
    double initial_energy = LocalParticle_get_energy0(part) + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part); // [eV]
    double gamma          = initial_energy / m0; // [1] 

    double n_avg        = c1*n_bb/(sigma_x + sigma_y);  // Avg. number of emitted photons from 1 macroparticle in one collision [1]
    double delta_avg    = c2*gamma/sigma_z * (n_bb/(sigma_x + sigma_y))*(n_bb/(sigma_x + sigma_y));  // Avg. rel. E loss for 1 macroparticle in one collision [1]
    double U_BS         = delta_avg*initial_energy;  // Average energy loss per macropart per IP. [eV]
    double u_avg        = delta_avg/n_avg;  // Average photon energy normalized to electron energy before emission [1]
    double e_photon_avg = u_avg*initial_energy;  // Average photon energy [eV]

/*
    printf("[synrad_avg] energy0: %.20f, ptau: %.20f, p0c: %.20f, beta0: %.20f\n", LocalParticle_get_energy0(part), LocalParticle_get_ptau(part), LocalParticle_get_p0c(part), LocalParticle_get_beta0(part));
    printf("[synrad_avg] c2: %.20f, sigma_x: %.20f, sigma_y: %.20f, sigma_z: %.20f, gamma: %.20f, delta_avg: %.20f, initial_energy: %.20f\n", c2, sigma_x, sigma_y, sigma_z, gamma, delta_avg, initial_energy);
*/
    char dump_file[1024];
    sprintf(dump_file, "%s/%s", dump_path, "xsuite_photons_avg.txt");
    //FILE *f1 = fopen(dump_file, "a");
    //fprintf(f1, "%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", part->ipart, initial_energy, sigma_x, sigma_y, sigma_z, n_avg, delta_avg, U_BS);  // save photon ID and energy, all in [ev]
    //fprintf(f1, "%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", part->ipart, initial_energy, n_bb, sigma_x, sigma_y, sigma_z, r, c1, c2, n_avg, delta_avg, U_BS);  // save photon ID and energy, all in [ev]
    //fclose(f1);
    LocalParticle_add_to_energy(part, -U_BS, 0);
    double energy_loss = -U_BS;
    return energy_loss;
}


/*gpufun*/
double synrad(LocalParticle *part,
       	    double Fr,  // [1] sqrt[(px' - px)^2 + (py' - py)^2]/Dt, Dt=1
	    double dz  // [m] z step between 2 slices ((z_max - z_min) / 2)
){

    const double m0 = LocalParticle_get_mass0(part); // particle mass [eV/c]
    double initial_energy = LocalParticle_get_energy0(part) + LocalParticle_get_ptau(part)*LocalParticle_get_p0c(part); // [eV]
    double energy = initial_energy;  // [eV]
    double gamma = energy / m0; // [1] 
    double r = pow(QELEM, 2.0)/(4.0* PI * EPSILON_0 * MELECTRON_KG * pow(C_LIGHT, 2.0));  // [m] electron radius
    double rho_inv = Fr / dz;  // [1/m] macropart bending radius from bb kick: dz/rho = dz/E*sqrt((px' - px)**2 + (py' - py)**2) = Fr * dz / E
    double tmp = 25.4 * energy*1e-9 * dz * rho_inv;  // [1]  Fr * dz, specific for 1 macropart, 1e-9 to convert [eV] to [GeV]
    int max_photons = (int)(tmp*10.0)+1;

    char dump_file[1024];
    sprintf(dump_file, "%s/%s", dump_path, "xsuite_photons.txt");
    //FILE *f1 = fopen(dump_file, "a");
    //FILE *f2 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_rho_inv.txt", "a");
    //FILE *f3 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_tmp.txt", "a");
    //FILE *f4 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_fr.txt", "a");
    //FILE *f5 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_max_photons.txt", "a");
    //FILE *f6 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_dz.txt", "a");
    //FILE *f7 = fopen("/Users/pkicsiny/phd/cern/xsuite/outputs/xsuite_eloss.txt", "a");

    //fprintf(f2, "%.10e\n", rho_inv);
    //fprintf(f3, "%.10e\n", tmp);
    //fprintf(f4, "%.10e\n", Fr);
    //fprintf(f5, "%d\n", max_photons);
    //fprintf(f6, "%.10e\n", dz);

    // photons are emitted uniformly in space along dz (between 2 slice interactions)
    dz /= (double)max_photons;

    // BS photon counter and energy storage
    int j = 0;
    double e_photon_array[1000];
    for (int i=0; i<max_photons; i++){
    
        double e_photon, ecrit;  // [GeV]
        if (synrad_0(part, energy, dz, rho_inv, &e_photon, &ecrit)){  // see if photon can be emitted
            e_photon_array[j] = e_photon;  // [GeV]

            //fprintf(f1, "%d %d %.10e %.10e %.10e %.10e %.10e %.10e %d\n", part->ipart, j, e_photon*1e9, ecrit*1e9, energy, rho_inv, dz, initial_energy, max_photons);  // save photon ID and energy, all in [ev]

            // update bending radius, macropart energy and gamma
            rho_inv *= energy/(energy - e_photon*1e9);
            energy  -= e_photon*1e9;
            gamma   *= (energy - e_photon*1e9)/energy;

            // some error handling
            if (e_photon_array[j]<=0.0){
		printf("photon emitted with negative energy: E_photon=%g [eV], E_macropart=%g [eV], photon ID: %d, max_photons: %d\n", e_photon*1e9, energy, j, max_photons);
       	    }

            // increment photon counter
            j++;
       	    if (j>=1000){
		printf("too many photons produced by one particle (photon ID: %d)\n", j);
		exit(-1);
	    }

        }
    }


    // update electron energy
    if (energy == 0.0){
        LocalParticle_set_state(part, -10); // used to flag this kind of loss
    }else{
        LocalParticle_add_to_energy(part, energy-initial_energy, 0);
    }
    double energy_loss = energy-initial_energy;
    //fprintf(f7, "%.10e\n", energy_loss);

    //fclose(f1);
    //fclose(f2);
    //fclose(f3);
    //fclose(f4);
    //fclose(f5);
    //fclose(f6);
    //fclose(f7);
    return energy_loss;
}



#endif /* XFIELDS_BEAMSTRAHLUNG_SPECTRUM_H */