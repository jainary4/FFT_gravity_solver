#ifndef CONSTANTS_H
#define CONSTANTS_H

#define N_STARS      20000      // 20 million stars
#define N_DM         80000      // 80 million dark matter
#define N_BH         10            // 10 black holes
#define N_TOTAL      (N_STARS + N_DM + N_BH)

// Box and grid
#define L            100.0         // Box size (kpc)
#define NMESH        32          // Physical grid (before padding)
#define NMESH_PADDED 64           // After zero-padding

// Masses (total normalized to 1.0)
#define M_TOTAL      1.0
#define F_STARS      0.15          // 15% in stars
#define F_DM         0.84          // 84% in dark matter  
#define F_BH         0.01          // 1% in black holes

// Plummer scale radii (kpc)
#define A_STARS      15.0          // Stars: concentrated
#define A_DM         25.0          // Dark matter: extended halo
#define A_BH         8.0           // Black holes: very central

// Velocity dispersions (cold collapse)
#define SIGMA_V_STARS 0.5          // Small random velocities
#define SIGMA_V_DM    0.3          // Very cold
#define SIGMA_V_BH    0.2          // Nearly at rest

#define PI 3.1415926

#endif