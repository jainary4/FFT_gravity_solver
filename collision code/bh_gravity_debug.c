// bh_debug.c
//
// Debug / visualization run for a single BH in the PM + FFT gravity solver.
//
// - Uses the same initial conditions as main.c (initialize_particle_system).
// - Identifies the BH as the heaviest particle.
// - CASE A: BH at its natural, COM-centred position -> prints acceleration.
// - CASE B: BH displaced by +1 kpc in x -> prints acceleration again
//           (should be noticeably larger, pointing back toward the mass).
// - Then integrates forward in time with leapfrog, writing 200 color PPM
//   frames "bh_frames/frame_XXXX.ppm" where the BH pixel is colored red.
//
// To use:
//   mkdir -p bh_frames
//   make bh_debug
//   ./bh_debug | tee bh_debug_log.txt
//
// Then use ffmpeg or other tools on bh_frames/frame_*.ppm.

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

// ------------------------------------------------------------
// Gravity context (same idea as in main.c)
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// Potential energy from mesh:
//   W = 0.5 * sum_{i,j,k} rho(x_ijk) * phi(x_ijk) * dV
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// FFT-based gravity force function (same as in main.c)
//
// Fills sys->accelerations and updates ctx->last_potential_energy.
// ------------------------------------------------------------
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

    /* 3) Build Laplacian RHS (4¦ÐG¦Ñ) and solve Poisson for phi */
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

    /* 4) Compute forces on padded grid */
    compute_forces_from_potential(phi_pad, Np, h,
                                  ctx->force_x,
                                  ctx->force_y,
                                  ctx->force_z);

    /* 5) Potential energy (for diagnostics) */
    ctx->last_potential_energy =
        potential_energy_from_mesh(ctx->rho_pad, phi_pad, Np, h);

    /* 6) Gather forces to particles => sys->accelerations */
    gather_forces_to_particles(ctx->force_x, ctx->force_y, ctx->force_z,
                               Np, sys, pm->N, h, offset);

    /* 7) Free temporaries */
    free_3d_array(laplacian_phi_pad, Np);
    free_3d_array(phi_pad, Np);
}

// ------------------------------------------------------------
// Compute net internal force ¦² m_i a_i (should be ~0 for self-gravity)
// ------------------------------------------------------------
static void total_force(const ParticleSystem *sys, double F[3])
{
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:Fx,Fy,Fz) schedule(static)
#endif
    for (int i = 0; i < sys->N; ++i) {
        double m  = sys->masses[i];
        double ax = sys->accelerations[i].x;
        double ay = sys->accelerations[i].y;
        double az = sys->accelerations[i].z;
        Fx += m * ax;
        Fy += m * ay;
        Fz += m * az;
    }

    F[0] = Fx; F[1] = Fy; F[2] = Fz;
}

// ------------------------------------------------------------
// Find the BH index by picking the heaviest particle.
// This avoids depending on TYPE_BH enum names.
// ------------------------------------------------------------
static int find_bh_index(const ParticleSystem *sys)
{
    if (!sys || sys->N <= 0) return -1;
    int idx = 0;
    double mmax = sys->masses[0];

    for (int i = 1; i < sys->N; ++i) {
        if (sys->masses[i] > mmax) {
            mmax = sys->masses[i];
            idx  = i;
        }
    }
    return idx;
}

// ------------------------------------------------------------
// Wrap coordinate into [0, L)
// ------------------------------------------------------------
static inline double wrap_box(double x)
{
    x = fmod(x, L);
    if (x < 0.0) x += L;
    if (x >= L)  x -= L;
    return x;
}

// ------------------------------------------------------------
// Print particle position & acceleration
// ------------------------------------------------------------
static void print_accel(const ParticleSystem *sys,
                        const char *label,
                        int idx)
{
    if (!sys || idx < 0 || idx >= sys->N) {
        printf("  %s: invalid index %d\n", label, idx);
        return;
    }
    double x  = sys->positions[idx].x;
    double y  = sys->positions[idx].y;
    double z  = sys->positions[idx].z;
    double ax = sys->accelerations[idx].x;
    double ay = sys->accelerations[idx].y;
    double az = sys->accelerations[idx].z;
    double r_c = sqrt( (x - 0.5*L)*(x - 0.5*L) +
                       (y - 0.5*L)*(y - 0.5*L) +
                       (z - 0.5*L)*(z - 0.5*L) );
    double amag = sqrt(ax*ax + ay*ay + az*az);
    printf("  %s: pos=(%.4f, %.4f, %.4f), r_from_box_center=%.4f\n",
           label, x, y, z, r_c);
    printf("       a=(%.3e, %.3e, %.3e), |a|=%.3e\n",
           ax, ay, az, amag);
}

// ------------------------------------------------------------
// Write x¨Cy projected density as a color PPM (BH pixel in red).
//
// - imgN x imgN image
// - Gray background from mass density (log-scaled).
// - The BH's pixel is set to (255, 0, 0).
// ------------------------------------------------------------
static int write_xy_density_ppm_with_bh(const ParticleSystem *sys,
                                        int imgN,
                                        int bh_index,
                                        const char *filename)
{
    if (!sys || imgN <= 0 || !filename) {
        fprintf(stderr, "write_xy_density_ppm_with_bh: invalid args\n");
        return -1;
    }

    const int Np = sys->N;
    if (Np <= 0) {
        fprintf(stderr, "write_xy_density_ppm_with_bh: no particles\n");
        return -1;
    }

    double *img = (double*)calloc((size_t)imgN * imgN, sizeof(double));
    unsigned char *buf = (unsigned char*)malloc((size_t)3 * imgN * imgN);
    if (!img || !buf) {
        fprintf(stderr, "write_xy_density_ppm_with_bh: allocation failed\n");
        free(img);
        free(buf);
        return -1;
    }

    /* Deposit mass onto 2D pixels (x-y projection) */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < Np; ++p) {
        double x = wrap_box(sys->positions[p].x);
        double y = wrap_box(sys->positions[p].y);
        double m = sys->masses[p];

        int ix = (int)(x / L * imgN);
        int iy = (int)(y / L * imgN);

        if (ix < 0)        ix = 0;
        if (ix >= imgN)    ix = imgN - 1;
        if (iy < 0)        iy = 0;
        if (iy >= imgN)    iy = imgN - 1;

        size_t idx = (size_t)iy * imgN + ix;

#ifdef _OPENMP
#pragma omp atomic
#endif
        img[idx] += m;
    }

    /* Find max density for scaling */
    double max_val = 0.0;
    for (int i = 0; i < imgN * imgN; ++i) {
        if (img[i] > max_val) max_val = img[i];
    }
    if (max_val <= 0.0) max_val = 1.0;

    /* Compute BH pixel coordinates */
    int ix_bh = -1, iy_bh = -1;
    if (bh_index >= 0 && bh_index < sys->N) {
        double xbh = wrap_box(sys->positions[bh_index].x);
        double ybh = wrap_box(sys->positions[bh_index].y);
        ix_bh = (int)(xbh / L * imgN);
        iy_bh = (int)(ybh / L * imgN);
        if (ix_bh < 0)        ix_bh = 0;
        if (ix_bh >= imgN)    ix_bh = imgN - 1;
        if (iy_bh < 0)        iy_bh = 0;
        if (iy_bh >= imgN)    iy_bh = imgN - 1;
    }

    /* Fill RGB buffer: gray from density, BH pixel in red */
    const double SCALE = 9.0;  /* log contrast */

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < imgN; ++iy) {
        for (int ix = 0; ix < imgN; ++ix) {
            size_t idx = (size_t)iy * imgN + ix;
            size_t o   = 3 * idx;

            if (ix == ix_bh && iy == iy_bh) {
                /* Color BH red */
                buf[o + 0] = 255;
                buf[o + 1] = 0;
                buf[o + 2] = 0;
            } else {
                double n = img[idx] / max_val;  /* 0..1 */
                if (n < 0.0) n = 0.0;
                if (n > 1.0) n = 1.0;
                double v = log(1.0 + SCALE * n) / log(1.0 + SCALE);
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                unsigned char gray = (unsigned char)(255.0 * v + 0.5);
                buf[o + 0] = gray;
                buf[o + 1] = gray;
                buf[o + 2] = gray;
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("write_xy_density_ppm_with_bh: fopen");
        free(img);
        free(buf);
        return -1;
    }

    /* PPM binary (P6) */
    fprintf(f, "P6\n%d %d\n255\n", imgN, imgN);
    size_t want = (size_t)3 * imgN * imgN;
    size_t got  = fwrite(buf, 1, want, f);
    if (got != want) {
        fprintf(stderr,
                "write_xy_density_ppm_with_bh: fwrite wrote %zu of %zu bytes\n",
                got, want);
    }

    fclose(f);
    free(img);
    free(buf);
    return 0;
}

// ------------------------------------------------------------
// main: check BH acceleration, then integrate and dump 200 frames
// ------------------------------------------------------------
int main(void)
{
#ifdef _OPENMP
    printf("Running bh_debug with OpenMP, max threads = %d\n",
           omp_get_max_threads());
#else
    printf("Running bh_debug (no OpenMP)\n");
#endif

    /* 1) Initialize particle system with your usual ICs */
    printf("\n[1] Initializing particle system...\n");
    ParticleSystem *sys = initialize_particle_system();
    if (!sys) {
        fprintf(stderr,
                "ERROR: initialize_particle_system() returned NULL\n");
        return 1;
    }
    printf("  N = %d particles\n", sys->N);

    int bh_index = find_bh_index(sys);
    if (bh_index < 0) {
        fprintf(stderr, "ERROR: could not identify BH (heaviest particle)\n");
        destroy_particle_system(sys);
        return 1;
    }
    printf("  BH guess: index = %d, mass = %.6e\n",
           bh_index, sys->masses[bh_index]);
    printf("  BH initial pos = (%.4f, %.4f, %.4f)\n",
           sys->positions[bh_index].x,
           sys->positions[bh_index].y,
           sys->positions[bh_index].z);
    printf("  BH initial vel = (%.4f, %.4f, %.4f)\n",
           sys->velocities[bh_index].x,
           sys->velocities[bh_index].y,
           sys->velocities[bh_index].z);

    /* 2) Create mesh + padded arrays (same as main) */
    printf("\n[2] Creating particle mesh and padded FFT grids...\n");
    ParticleMesh *pm = create_particle_mesh();
    if (!pm) {
        fprintf(stderr,
                "ERROR: create_particle_mesh() returned NULL\n");
        destroy_particle_system(sys);
        return 1;
    }
    printf("  Mesh: %d^3, cell size h = %.6f\n", pm->N, pm->cell_size);

    int Np = NMESH_PADDED;
    if (Np <= 0) {
        fprintf(stderr, "ERROR: NMESH_PADDED must be > 0\n");
        destroy_particle_mesh(pm);
        destroy_particle_system(sys);
        return 1;
    }

    int offset = 0;

    double ***rho_pad = allocate_3d_array(Np);
    double ***fx      = allocate_3d_array(Np);
    double ***fy      = allocate_3d_array(Np);
    double ***fz      = allocate_3d_array(Np);

    if (!rho_pad || !fx || !fy || !fz) {
        fprintf(stderr,
                "ERROR: allocate_3d_array failed for padded grids\n");
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

    /* 3) CASE A: BH at natural initial position */
    printf("\n[3] CASE A: BH at natural initial position (COM-centred ICs)\n");
    gravity_fft_force(sys, &gctx);
    print_accel(sys, "BH (case A)", bh_index);

    double Ftot[3];
    total_force(sys, Ftot);
    double Fmag = sqrt(Ftot[0]*Ftot[0] +
                       Ftot[1]*Ftot[1] +
                       Ftot[2]*Ftot[2]);
    printf("  Total internal force ¦² m_i a_i (case A): "
           "(%.3e, %.3e, %.3e), |F|=%.3e\n",
           Ftot[0], Ftot[1], Ftot[2], Fmag);
    printf("  (Should be ~0 for self-gravity; small non-zero = discretization)\n");

    /* 4) CASE B: Displace BH by +1.0 kpc in x to show it feels gravity */
    printf("\n[4] CASE B: displace BH by +1.0 kpc along x\n");
    double old_x = sys->positions[bh_index].x;
    double old_y = sys->positions[bh_index].y;
    double old_z = sys->positions[bh_index].z;

    sys->positions[bh_index].x = wrap_box(old_x + 1.0);

    printf("  BH moved from (%.4f, %.4f, %.4f)\n"
           "             to (%.4f, %.4f, %.4f)\n",
           old_x, old_y, old_z,
           sys->positions[bh_index].x,
           sys->positions[bh_index].y,
           sys->positions[bh_index].z);

    gravity_fft_force(sys, &gctx);
    print_accel(sys, "BH (case B, displaced)", bh_index);

    total_force(sys, Ftot);
    Fmag = sqrt(Ftot[0]*Ftot[0] +
                Ftot[1]*Ftot[1] +
                Ftot[2]*Ftot[2]);
    printf("  Total internal force ¦² m_i a_i (case B): "
           "(%.3e, %.3e, %.3e), |F|=%.3e\n",
           Ftot[0], Ftot[1], Ftot[2], Fmag);

    printf("\n[5] Time integration + frames (starting from CASE B state)\n");
    printf("    Writing 200 color PPM frames to 'bh_frames/frame_XXXX.ppm'\n");
    printf("    (Make sure directory 'bh_frames' exists.)\n\n");

    /* 5) Integrate with leapfrog and write 200 frames */
    double dt = 0.01;     /* small-ish timestep for this debug */
    int    n_frames = 200;
    int    steps_per_frame = 1;  /* 1 leapfrog step per frame => t ~ 2.0 */

    int imgN = 512;       /* image resolution */

    for (int frame = 0; frame < n_frames; ++frame) {
        int step = frame * steps_per_frame;

        /* Write frame */
        char fname[256];
        snprintf(fname, sizeof(fname),
                 "bh_frames/frame_%04d.ppm", frame + 1);
        int rc = write_xy_density_ppm_with_bh(sys, imgN, bh_index, fname);
        if (rc != 0) {
            fprintf(stderr,
                    "WARNING: failed to write frame %s at step %d\n",
                    fname, step);
        }

        /* Print BH radius & speed every 20 frames */
        if ((frame % 20) == 0) {
            double bx = sys->positions[bh_index].x;
            double by = sys->positions[bh_index].y;
            double bz = sys->positions[bh_index].z;
            double bvx = sys->velocities[bh_index].x;
            double bvy = sys->velocities[bh_index].y;
            double bvz = sys->velocities[bh_index].z;
            double r_c = sqrt( (bx - 0.5*L)*(bx - 0.5*L) +
                               (by - 0.5*L)*(by - 0.5*L) +
                               (bz - 0.5*L)*(bz - 0.5*L) );
            double v_bh = sqrt(bvx*bvx + bvy*bvy + bvz*bvz);
            printf("  Frame %3d: t = %.3f, BH r_from_center = %.4e, |v_BH| = %.4e\n",
                   frame + 1, (frame + 1) * dt * steps_per_frame,
                   r_c, v_bh);
        }

        /* Advance steps_per_frame leapfrog steps */
        for (int s = 0; s < steps_per_frame; ++s) {
            leapfrog_step(sys, dt, gravity_fft_force, &gctx);
        }
    }

    printf("\n[Done] bh_debug finished. Frames are in 'bh_frames/'.\n");

    /* 6) Cleanup */
    free_3d_array(rho_pad, Np);
    free_3d_array(fx, Np);
    free_3d_array(fy, Np);
    free_3d_array(fz, Np);
    destroy_particle_mesh(pm);
    destroy_particle_system(sys);

    return 0;
}
