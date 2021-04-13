#include <math.h>

typedef struct {
    int ix, iy;
    double wij, wi1j, wij1, wi1j1;
    }Ind_And_Weight;

typedef struct{
    int nx; 
    int ny;
    double x0;
    double y0;
    double dx;
    double dy;
    double* dphi_dx;
    double* dphi_dy;
}Fieldmap;

inline Ind_And_Weight get_ind_and_weight(
        double x, double dx, double x0,
        double y, double dy, double y0){
    
    Ind_And_Weight iaw;
    
    iaw.ix = floor((x - x0) / dx);
    iaw.iy = floor((y - y0) / dy);

    double dxi = x - (x0 + iaw.ix * dx);
    double dyi = y - (y0 + iaw.iy * dy);

    iaw.wij =    (1.-dxi/dx)*(1.-dyi/dy);
    iaw.wi1j =   (1.-dxi/dx)*(dyi/dy)   ;
    iaw.wij1 =   (dxi/dx)   *(1.-dyi/dy);
    iaw.wi1j1 =  (dxi/dx)   *(dyi/dy)   ;

    return iaw;
}

inline void fieldmap_add_value_at_point(
        const Ind_And_Weight iaw,
        int nx,
        const double* map,
        const double factor,
              double* res
        ){

    res[0] += factor*(
               iaw.wijk   * map[iaw.ix   + iaw.ix*nx    ]
             + iaw.wij1k  * map[iaw.ix+1 + iaw.ix*nx    ]
             + iaw.wi1jk  * map[iaw.ix+  + (iaw.ix+1)*nx ]
             + iaw.wi1j1k * map[iaw.ix+1 + (iaw.ix+1)*nx ]);

}

// CPU *************

typedef struct{
    double* x;
    double* y;
    int npart;
    int ipart;
}Particle;


// CPU *************

typedef struct{
    double* x;
    double* y;
    int npart;
    int ipart;
}Particle;

double particle_get_x(Particle part){
    return part.x[part.ipart];
}

double* particle_getp_x(Particle part){
    return part.x + part.ipart;
}

double particle_get_y(Particle part){
    return part.y[part.ipart];
}

double* particle_getp_y(Particle part){
    return part.y + part.ipart;
}


void spacecharge_track_CPU(const Spacecharge* el, Particle part){

    double x0 = spacecharge_get_fieldmap_x0(el);
    double y0 = spacecharge_get_fieldmap_y0(el);
    int nx = spacecharge_get_fieldmap_nx(el);
    int ny = spacecharge_get_fieldmap_ny(el);
    double dx = spacecharge_get_fieldmap_dx(el);
    double dy = spacecharge_get_fieldmap_dy(el);

    double* dphi_dx = spacecharge_getp_fieldmap_dphi_dx(el);
    double* dphi_dy = spacecharge_getp_fieldmap_dphi_dx(el);

    double factor = 10.;

    for(part.ipart; part.ipart<part.n_part; part.ipart++){
        double x = particle_get_x(part);
        double y = particle_get_y(part);

        Ind_And_Weight iaw = get_ind_and_weight(x, dx, x0, nx,
                                 y, dy, y0, ny);

        fieldmap_add_value_at_point(iaw, dphi_dx, factor, 
                                        particle_getp_x(part));
        fieldmap_add_value_at_point(iaw, dphi_dy, factor, 
                                    particle_getp_y(part));
    }

}

// GPU *************

typedef struct{
    double x;
    double y;
}Particle;

double particle_get_x(Particle part){
    return part.x;
}

double* particle_getp_x(Particle part){
    return &(part.x);
}

double particle_get_y(Particle part){
    return part.y;
}

double* particle_getp_y(Particle part){
    return &(part.y);
}

void spacecharge_track_CPU(const Spacecharge* el, Particle part){

    double x0 = spacecharge_get_fieldmap_x0(el);
    double y0 = spacecharge_get_fieldmap_y0(el);
    int nx = spacecharge_get_fieldmap_nx(el);
    int ny = spacecharge_get_fieldmap_ny(el);
    double dx = spacecharge_get_fieldmap_dx(el);
    double dy = spacecharge_get_fieldmap_dy(el);

    double* dphi_dx = spacecharge_getp_fieldmap_dphi_dx(el);
    double* dphi_dy = spacecharge_getp_fieldmap_dphi_dx(el);

    double factor = 10.;

    //for(part.ipart; part.ipart<part.n_part; part.ipart++){
        double x = particle_get_x(part);
        double y = particle_get_y(part);

        Ind_And_Weight iaw = get_ind_and_weight(x, dx, x0, nx,
                                 y, dy, y0, ny);


        fieldmap_add_value_at_point(iaw, dphi_dx, factor, 
                                        particle_getp_x(part));
        fieldmap_add_value_at_point(iaw, dphi_dy, factor, 
                                    particle_getp_y(part));
    //}

}



