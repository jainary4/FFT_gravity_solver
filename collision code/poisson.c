#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "constants.h"
#include "mesh.h"
#include "poisson.h"

double*** create_laplacian_equation(int Np, double ***rho_pad ){
    // first we allocate an array that will hold the zero padded laplacian phi values 
    // now laplacian_phi = 4*PI*G*rho_pad
    // this laplacian_phi_pad will be used in the poisson solver to get phi in fourier space

  double ***laplacian_phi_pad= allocate_3d_array(Np);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Np; i++) {
        for (int j = 0; j < Np; j++) {
            for (int k = 0; k < Np; k++) {
                laplacian_phi_pad[i][j][k] = 4.0f * PI * G * rho_pad[i][j][k];
            }
        }
    }
    return laplacian_phi_pad;
}

