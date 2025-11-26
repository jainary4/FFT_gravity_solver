#ifndef CONSTANTS_H
#define CONSTANTS_H

#define N_STARS      200      // 20 million stars
#define N_DM         800      // 80 million dark matter
#define N_BH         1            // 10 black holes
#define N_TOTAL      (N_STARS + N_DM + N_BH)

// Box and grid
#define L            10.0         // Box size (kpc)
#define NMESH        8         // Physical grid (before padding)
#define NMESH_PADDED 16           // After zero-padding

// Masses (total normalized to 1.0)
#define M_TOTAL      1.0
#define F_STARS      0.20          // 15% in stars
#define F_DM         0.79          // 84% in dark matter  
#define F_BH         0.01          // 1% in black holes

// Plummer scale radii (kpc)
#define A_STARS      1.50          // Stars: concentrated
#define A_DM         2.50          // Dark matter: extended halo
#define A_BH         0.80           // Black holes: very central

// Velocity dispersions (cold collapse)
#define SIGMA_V_STARS 0.5          // Small random velocities
#define SIGMA_V_DM    0.3          // Very cold
#define SIGMA_V_BH    0.2          // Nearly at rest

#define PI 3.1415926
#define G 4.302e-6  // Gravitational constant in (kpc/M_sun)*(km/s)^2

#define IDX(i,j,k,N) ((i)*(N)*(N) + (j)*(N) + (k))

#endif