#ifndef XFIELDS_IPTOCP3D_EMPTY_H
#define XFIELDS_IPTOCP3D_EMPTY_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

/*gpufun*/
void IPToCP3D_empty_track_local_particle(IPToCP3D_emptyData el, 
		 	   LocalParticle* part){
   clock_t tt;
   tt = clock(); 
   tt = clock() - tt;
   double ttime_taken = ((double)tt)/CLOCKS_PER_SEC;
   printf("[iptocp.h] IPtoCP full took %.8f seconds to execute\n", ttime_taken);
}


#endif
