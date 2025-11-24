#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "constants.h"
#include "structs.h"
#include "particles.h"

int main() {
    srand(time(NULL));
    
    ParticleSystem *sys = initialize_particle_system();
    
    // Your simulation code here
    
    destroy_particle_system(sys);
    return 0;
}