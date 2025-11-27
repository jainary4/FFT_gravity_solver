// main.c – N-body galaxy collapse with FFT gravity and leapfrog integrator

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "constants.h"
#include "structs.h"
#include "particles.h"
#include "mesh.h"
#include "poisson.h"
#include "fft_solver.h"
#include "force.h"
#include "integrator.h"
#include "snapshot.h"      // for visualization
// #include "bh_collision.h"   // optional, if/when you enable BH collisions

/* ------------------------------------------------------------
 * Optional helper: write a 3D field to text (for debugging)
 * ------------------------------------------------------------ */
int write_to_txt(double ***phi, int N, const char *filepath)
{
    if (!phi || N <= 0 || !filepath) return -1;

    FILE *f = fopen(filepath, "w");
    if (!f) {
        perror("write_to_txt: fopen");
        return -1;
    }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                if (fprintf(f, "%.17g\n", phi[i][j][k]) < 0) {
                    perror("write_to_txt: fprintf");
                    fclose(f);
                    return -1;
                }

    if (fclose(f) != 0) {
        perror("write_to_txt: fclose");
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------
 * Gravity context for FFT-based force computation
 * ------------------------------------------------------------ */
typedef struct {
    ParticleMesh *pm;      /* physical mesh (NMESH^3)      */
    int           Np;      /* padded grid size             */
    int           offset;  /* physical mesh offset in pad  */
    double        h;       /* cell size                    */

    double      ***rho_pad;
    double      ***force_x;
    double      ***force_y;
    double      ***force_z;

    double        last_potential_energy;  /* for diagnostics */
} GravityFFTContext;

/* ------------------------------------------------------------
 * Diagnostics helpers
 * ------------------------------------------------------------ */

static double total_mass(const ParticleSystem *sys)
{
    double M = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:M) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i)
        M += sys->masses[i];
    return M;
}

static double total_kinetic(const ParticleSystem *sys)
{
    double K = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:K) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        double vx = sys->velocities[i].x;
        double vy = sys->velocities[i].y;
        double vz = sys->velocities[i].z;
        double v2 = vx*vx + vy*vy + vz*vz;
        K += 0.5 * sys->masses[i] * v2;
    }
    return K;
}

static void total_momentum(const ParticleSystem *sys, double P[3])
{
    double Px = 0.0, Py = 0.0, Pz = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Px,Py,Pz) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        double m = sys->masses[i];
        Px += m * sys->velocities[i].x;
        Py += m * sys->velocities[i].y;
        Pz += m * sys->velocities[i].z;
    }
    P[0] = Px; P[1] = Py; P[2] = Pz;
}

static void center_of_mass(const ParticleSystem *sys,
                           double r_cm[3], double v_cm[3])
{
    double M  = 0.0;
    double Rx = 0.0, Ry = 0.0, Rz = 0.0;
    double Vx = 0.0, Vy = 0.0, Vz = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:M,Rx,Ry,Rz,Vx,Vy,Vz) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        double m  = sys->masses[i];
        double x  = sys->positions[i].x;
        double y  = sys->positions[i].y;
        double z  = sys->positions[i].z;
        double vx = sys->velocities[i].x;
        double vy = sys->velocities[i].y;
        double vz = sys->velocities[i].z;

        M  += m;
        Rx += m * x;
        Ry += m * y;
        Rz += m * z;
        Vx += m * vx;
        Vy += m * vy;
        Vz += m * vz;
    }

    if (M > 0.0) {
        r_cm[0] = Rx / M;
        r_cm[1] = Ry / M;
        r_cm[2] = Rz / M;
        v_cm[0] = Vx / M;
        v_cm[1] = Vy / M;
        v_cm[2] = Vz / M;
    } else {
        r_cm[0] = r_cm[1] = r_cm[2] = 0.0;
        v_cm[0] = v_cm[1] = v_cm[2] = 0.0;
    }
}

/* Potential energy from mesh */
static double potential_energy_from_mesh(double ***rho_pad,
                                         double ***phi_pad,
                                         int Np,
                                         double h)
{
    double W  = 0.0;
    double dV = h * h * h;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) reduction(+:W) schedule(static)
#endif
    for (int i = 0; i < Np; ++i)
        for (int j = 0; j < Np; ++j)
            for (int k = 0; k < Np; ++k)
                W += 0.5 * rho_pad[i][j][k] * phi_pad[i][j][k] * dV;

    return W;
}

/* Check for NaN/Inf in particle positions & velocities */
static int check_finite_state(const ParticleSystem *sys)
{
    for (int i = 0; i < sys->N; ++i) {
        double x  = sys->positions[i].x;
        double y  = sys->positions[i].y;
        double z  = sys->positions[i].z;
        double vx = sys->velocities[i].x;
        double vy = sys->velocities[i].y;
        double vz = sys->velocities[i].z;

        if (!isfinite(x) || !isfinite(y) || !isfinite(z) ||
            !isfinite(vx) || !isfinite(vy) || !isfinite(vz)) {
            fprintf(stderr,
                    "Non-finite state: particle %d "
                    "pos=(%g,%g,%g) vel=(%g,%g,%g)\n",
                    i, x,y,z, vx,vy,vz);
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------
 * FFT-based gravity force function (called by integrator)
 * ------------------------------------------------------------ */
static void gravity_fft_force(ParticleSystem *sys, void *vctx)
{
    GravityFFTContext *ctx = (GravityFFTContext*)vctx;
    if (!sys || !ctx || !ctx->pm ||
        !ctx->rho_pad || !ctx->force_x ||
        !ctx->force_y || !ctx->force_z) {
        fprintf(stderr, "gravity_fft_force: invalid context\n");
        return;
    }

    const int Np = ctx->Np;
    ParticleMesh *pm = ctx->pm;
    const double h = pm->cell_size;
    const int offset = ctx->offset;  /* currently 0 */

    /* 1) Zero padded density grid */
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (int i = 0; i < Np; ++i)
        for (int j = 0; j < Np; ++j)
            for (int k = 0; k < Np; ++k)
                ctx->rho_pad[i][j][k] = 0.0;

    /* 2) Assign particle mass to mesh via padded CIC */
    assign_mass_cic_padded(ctx->rho_pad, Np, sys, pm->N, h, offset);

    /* 3) Build Laplacian RHS (4πGρ) and solve Poisson for phi */
    double ***laplacian_phi_pad = create_laplacian_equation(Np, ctx->rho_pad);
    if (!laplacian_phi_pad) {
        fprintf(stderr,
                "gravity_fft_force: create_laplacian_equation returned NULL\n");
        return;
    }

    double ***phi_pad = solve_poisson_fftw(laplacian_phi_pad, Np, h);
    if (!phi_pad) {
        fprintf(stderr,
                "gravity_fft_force: solve_poisson_fftw returned NULL\n");
        free_3d_array(laplacian_phi_pad, Np);
        return;
    }

    /* 4) Compute forces on padded grid (periodic stencil) */
    compute_forces_from_potential(phi_pad, Np, h,
                                  ctx->force_x,
                                  ctx->force_y,
                                  ctx->force_z);

    /* 5) Compute potential energy (for diagnostics) */
    ctx->last_potential_energy =
        potential_energy_from_mesh(ctx->rho_pad, phi_pad, Np, h);

    /* 6) Gather forces to particles => sys->accelerations */
    gather_forces_to_particles(ctx->force_x, ctx->force_y, ctx->force_z,
                               Np, sys, pm->N, h, offset);

    /* 7) Free temporaries */
    free_3d_array(laplacian_phi_pad, Np);
    free_3d_array(phi_pad, Np);
}

/* ------------------------------------------------------------
 * Virialization by *velocity* rescaling
 * ------------------------------------------------------------ */
static void rescale_velocities_to_target_virial(ParticleSystem *sys,
                                                GravityFFTContext *gctx,
                                                double Q_target,
                                                double *M0, double *K0,
                                                double *W0, double *E0)
{
    /* Compute current K, W, M */
    gravity_fft_force(sys, gctx);
    *K0 = total_kinetic(sys);
    *W0 = gctx->last_potential_energy;
    *M0 = total_mass(sys);
    *E0 = *K0 + *W0;

    if (fabs(*W0) <= 0.0 || Q_target <= 0.0) {
        fprintf(stderr,
                "rescale_velocities_to_target_virial: invalid W0 or Q_target\n");
        return;
    }

    double Q0   = 2.0 * (*K0) / fabs(*W0);
    double beta = sqrt(Q_target / Q0);

    printf("\n[Virial init] K0 = %.6e, W0 = %.6e, M0 = %.6e\n", *K0, *W0, *M0);
    printf("[Virial init] Q0 = 2K/|W| = %.6e\n", Q0);
    printf("[Virial] Target Q = %.3f, beta (velocity scale) = %.6e\n",
           Q_target, beta);

    /* Scale all velocities */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        sys->velocities[i].x *= beta;
        sys->velocities[i].y *= beta;
        sys->velocities[i].z *= beta;
    }

    /* Recompute diagnostics after rescaling */
    gravity_fft_force(sys, gctx);
    *K0 = total_kinetic(sys);
    *W0 = gctx->last_potential_energy;
    *M0 = total_mass(sys);
    *E0 = *K0 + *W0;
    double Q_new = 2.0 * (*K0) / fabs(*W0);

    printf("[Virial] After velocity rescale:\n");
    printf("         M0 = %.6e, K0 = %.6e, W0 = %.6e, E0 = %.6e\n",
           *M0, *K0, *W0, *E0);
    printf("         Q_new = 2K/|W| = %.6e\n\n", Q_new);
}

/* ------------------------------------------------------------
 * main
 * ------------------------------------------------------------ */
int main(void)
{
#ifdef _OPENMP
    printf("Running main with OpenMP, max threads = %d\n", omp_get_max_threads());
#endif

    /* 1) Initialize particle system */
    printf("Initializing particle system...\n");
    ParticleSystem *sys = initialize_particle_system();
    if (!sys) {
        fprintf(stderr, "ERROR: initialize_particle_system() returned NULL\n");
        return 1;
    }
    printf("  Loaded %d particles\n", sys->N);

    /* 2) Create mesh */
    printf("\nCreating particle mesh...\n");
    ParticleMesh *pm = create_particle_mesh();
    if (!pm) {
        fprintf(stderr, "ERROR: create_particle_mesh() returned NULL\n");
        destroy_particle_system(sys);
        return 1;
    }
    printf("  Mesh: %d^3, cell size h = %.6f\n", pm->N, pm->cell_size);

    /* 3) Allocate padded grids for FFT */
    int Np = NMESH_PADDED;
    if (Np <= 0) {
        fprintf(stderr, "ERROR: NMESH_PADDED must be > 0\n");
        destroy_particle_mesh(pm);
        destroy_particle_system(sys);
        return 1;
    }

    int offset = 0;  /* physical mesh at corner of padded grid */

    double ***rho_pad = allocate_3d_array(Np);
    double ***fx      = allocate_3d_array(Np);
    double ***fy      = allocate_3d_array(Np);
    double ***fz      = allocate_3d_array(Np);

    if (!rho_pad || !fx || !fy || !fz) {
        fprintf(stderr, "ERROR: allocate_3d_array failed for padded grids\n");
        if (rho_pad) free_3d_array(rho_pad, Np);
        if (fx)      free_3d_array(fx, Np);
        if (fy)      free_3d_array(fy, Np);
        if (fz)      free_3d_array(fz, Np);
        destroy_particle_mesh(pm);
        destroy_particle_system(sys);
        return 1;
    }

    GravityFFTContext gctx = {
        .pm   = pm,
        .Np   = Np,
        .offset = offset,
        .h    = pm->cell_size,
        .rho_pad = rho_pad,
        .force_x = fx,
        .force_y = fy,
        .force_z = fz,
        .last_potential_energy = 0.0
    };

    /* 4) Virialize by velocity scaling */
    double M0, K0, W0, E0;
    const double Q_TARGET = 0.5;

    rescale_velocities_to_target_virial(sys, &gctx, Q_TARGET,
                                        &M0, &K0, &W0, &E0);

    /* 5) Initial momentum & COM diagnostics */
    double P0[3]; total_momentum(sys, P0);
    double P0_mag = sqrt(P0[0]*P0[0] + P0[1]*P0[1] + P0[2]*P0[2]);

    double rcm0[3], vcm0[3];
    center_of_mass(sys, rcm0, vcm0);
    double rcm0_mag = sqrt(rcm0[0]*rcm0[0] + rcm0[1]*rcm0[1] + rcm0[2]*rcm0[2]);
    double vcm0_mag = sqrt(vcm0[0]*vcm0[0] + vcm0[1]*vcm0[1] + vcm0[2]*vcm0[2]);

    printf("Initial diagnostics after velocity virial rescale:\n");
    printf("  M0   = %.6e\n", M0);
    printf("  K0   = %.6e\n  W0   = %.6e\n  E0   = %.6e\n", K0, W0, E0);
    printf("  |P0| = %.6e\n", P0_mag);
    printf("  |r_cm0| = %.6e, |v_cm0| = %.6e\n\n", rcm0_mag, vcm0_mag);

    /* 6) Time integration parameters */
    double dt           = 0.01;
    int    n_steps      = 1000;
    int    output_every   = 10;

    /* snapshots: every 5 steps → 200 frames for 1000 steps */
    int    snapshot_every = 5;
    int    imgN           = 512;
    int    frame_id       = 0;

    printf("# step   t        K           W           E         dE/E0      dM/M0     |P|        |r_cm|    |v_cm|\n");

    for (int step = 0; step < n_steps; ++step) {

        leapfrog_step(sys, dt, gravity_fft_force, &gctx);

        if ((step + 1) % output_every == 0 || step == 0) {
            double K  = total_kinetic(sys);
            double W  = gctx.last_potential_energy;
            double E  = K + W;
            double dE = (fabs(E0) > 0.0) ? (E - E0) / fabs(E0) : 0.0;

            double M  = total_mass(sys);
            double dM = (M0 != 0.0) ? (M - M0) / M0 : 0.0;

            double P[3]; total_momentum(sys, P);
            double P_mag = sqrt(P[0]*P[0] + P[1]*P[1] + P[2]*P[2]);

            double rcm[3], vcm[3];
            center_of_mass(sys, rcm, vcm);
            double rcm_mag = sqrt(rcm[0]*rcm[0] + rcm[1]*rcm[1] + rcm[2]*rcm[2]);
            double vcm_mag = sqrt(vcm[0]*vcm[0] + vcm[1]*vcm[1] + vcm[2]*vcm[2]);

            printf("%6d  %8.3e  %11.4e  %11.4e  %11.4e  %9.2e  %9.2e  %11.4e  %8.3e  %8.3e\n",
                   step + 1,
                   (step + 1) * dt,
                   K, W, E,
                   dE, dM,
                   P_mag, rcm_mag, vcm_mag);
        }

        /* snapshots with consecutive numbering */
        if (snapshot_every > 0 && (step + 1) % snapshot_every == 0) {
            frame_id++;
            char fname[256];
            snprintf(fname, sizeof(fname), "frames/frame_%04d.pgm", frame_id);
            int rc = write_xy_density_pgm(sys, imgN, fname);
            if (rc != 0) {
                fprintf(stderr,
                        "WARNING: failed to write snapshot %s at step %d\n",
                        fname, step + 1);
            }
        }

        if (!check_finite_state(sys)) {
            fprintf(stderr,
                    "ERROR: detected NaN/Inf in particle state at step %d\n",
                    step + 1);
            break;
        }
    }

    /* 7) Cleanup */
    free_3d_array(rho_pad, Np);
    free_3d_array(fx,      Np);
    free_3d_array(fy,      Np);
    free_3d_array(fz,      Np);
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);

    return 0;
}
