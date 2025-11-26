#ifndef FORCE_H
#define FORCE_H

/*
 * Compute gravitational force field from potential via finite differences.
 * F = -∇phi (negative gradient)
 *
 * Inputs:
 *   phi_pad    : potential array [N][N][N]
 *   N          : grid dimension (e.g., NMESH_PADDED = 16)
 *   h          : cell size (kpc)
 *
 * Outputs (allocated by caller):
 *   force_x    : F_x = -d(phi)/dx [N][N][N]
 *   force_y    : F_y = -d(phi)/dy [N][N][N]
 *   force_z    : F_z = -d(phi)/dz [N][N][N]
 *
 * Method: central finite differences
 *   d(phi)/dx[i,j,k] ≈ (phi[i+1,j,k] - phi[i-1,j,k]) / (2h)
 * Boundary handling: forward/backward differences at edges (no periodic wrapping).
 */
void compute_forces_from_potential(
    double ***phi_pad,
    int N,
    double h,
    double ***force_x,
    double ***force_y,
    double ***force_z
);

#endif 
