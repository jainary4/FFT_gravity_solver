#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "force.h"
#include "mesh.h"

/*
 * Compute gravitational force components (F = -∇(phi)) from potential using finite differences.
 * Central differences in the interior; forward/backward at boundaries.
 */
void compute_forces_from_potential(
    double ***phi_pad,
    int N,
    double h,
    double ***force_x,
    double ***force_y,
    double ***force_z
) {
    if (!phi_pad || !force_x || !force_y || !force_z || N <= 0 || h <= 0.0) {
        fprintf(stderr, "compute_forces_from_potential: invalid arguments\n");
        return;
    }

    double inv_2h = 1.0 / (2.0 * h);
    double inv_h  = 1.0 / h;

    /* Compute F_x = -d(phi)/dx */
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double grad_x;
                if (i == 0) {
                    // Forward difference at i=0
                    grad_x = (phi_pad[1][j][k] - phi_pad[0][j][k]) * inv_h;
                } else if (i == N - 1) {
                    // Backward difference at i=N-1
                    grad_x = (phi_pad[N-1][j][k] - phi_pad[N-2][j][k]) * inv_h;
                } else {
                    // Central difference in interior
                    grad_x = (phi_pad[i+1][j][k] - phi_pad[i-1][j][k]) * inv_2h;
                }
                force_x[i][j][k] = -grad_x;
            }
        }
    }

    /* Compute F_y = -d(phi)/dy */
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double grad_y;
                if (j == 0) {
                    // Forward difference at j=0
                    grad_y = (phi_pad[i][1][k] - phi_pad[i][0][k]) * inv_h;
                } else if (j == N - 1) {
                    // Backward difference at j=N-1
                    grad_y = (phi_pad[i][N-1][k] - phi_pad[i][N-2][k]) * inv_h;
                } else {
                    // Central difference in interior
                    grad_y = (phi_pad[i][j+1][k] - phi_pad[i][j-1][k]) * inv_2h;
                }
                force_y[i][j][k] = -grad_y;
            }
        }
    }

    /* Compute F_z = -d(phi)/dz */
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double grad_z;
                if (k == 0) {
                    // Forward difference at k=0
                    grad_z = (phi_pad[i][j][1] - phi_pad[i][j][0]) * inv_h;
                } else if (k == N - 1) {
                    // Backward difference at k=N-1
                    grad_z = (phi_pad[i][j][N-1] - phi_pad[i][j][N-2]) * inv_h;
                } else {
                    // Central difference in interior
                    grad_z = (phi_pad[i][j][k+1] - phi_pad[i][j][k-1]) * inv_2h;
                }
                force_z[i][j][k] = -grad_z;
            }
        }
    }
}
