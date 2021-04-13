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
               iaw.wij   * map[iaw.ix   + iaw.ix*nx    ]
             + iaw.wij1  * map[iaw.ix+1 + iaw.ix*nx    ]
             + iaw.wi1j  * map[iaw.ix+  + (iaw.ix+1)*nx ]
             + iaw.wi1j1 * map[iaw.ix+1 + (iaw.ix+1)*nx ]);

}

typedef struct{
    double* x;
    double* y;
    int npart;
    int ipart;
}Particle;

inline double particle_get_x(Particle part){
    return part.x[part.ipart];
}

inline double* particle_getp_x(Particle part){
    return part.x + part.ipart;
}

inline double particle_get_y(Particle part){
    return part.y[part.ipart];
}

inline double* particle_getp_y(Particle part){
    return part.y + part.ipart;
}

void spacecharge_track_CPU(
    Particle part,
    double x0,
    double y0,
    int nx,
    int ny,
    double dx,
    double dy,
    double* dphi_dx,
    double* dphi_dy,
    double factor){

    for(part.ipart; part.ipart<part.npart; part.ipart++){
        double x = particle_get_x(part);
        double y = particle_get_y(part);

        Ind_And_Weight iaw = get_ind_and_weight(x, dx, x0,
                                                y, dy, y0);

        fieldmap_add_value_at_point(iaw, nx, dphi_dx, factor, 
                                        particle_getp_x(part));
        fieldmap_add_value_at_point(iaw, nx,dphi_dy, factor, 
                                    particle_getp_y(part));
    }

}

