#include <math.h>
#include <omp.h>
#include <stdio.h>
#include "constants.h"
#include "force.h"

/* 
 * Compute gravitational force on the padded grid from the potential phi.
 * - phi is periodic on the padded Np^3 domain (FFT assumption).
 * - We use central differences with periodic wrapping:
 *      Fx = -(phi[i+1,j,k] - phi[i-1,j,k]) / (2h), etc.
 * - Output arrays fx, fy, fz must already be allocated Np^3.
 */
void compute_forces_from_potential(double ***phi,
                                   int Np,
                                   double h,
                                   double ***fx,
                                   double ***fy,
                                   double ***fz)
{
    if (!phi || !fx || !fy || !fz || Np <= 0 || h <= 0.0) {
        fprintf(stderr, "compute_forces_from_potential: invalid arguments\n");
        return;
    }

    const double inv_2h = 1.0 / (2.0 * h);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < Np; ++i) {
        for (int j = 0; j < Np; ++j) {
            for (int k = 0; k < Np; ++k) {
                int ip = (i + 1) % Np;
                int im = (i - 1 + Np) % Np;
                int jp = (j + 1) % Np;
                int jm = (j - 1 + Np) % Np;
                int kp = (k + 1) % Np;
                int km = (k - 1 + Np) % Np;

                double dphidx = (phi[ip][j ][k ] - phi[im][j ][k ]) * inv_2h;
                double dphidy = (phi[i ][jp][k ] - phi[i ][jm][k ]) * inv_2h;
                double dphidz = (phi[i ][j ][kp] - phi[i ][j ][km]) * inv_2h;

                fx[i][j][k] = -dphidx;
                fy[i][j][k] = -dphidy;
                fz[i][j][k] = -dphidz;
            }
        }
    }
}
