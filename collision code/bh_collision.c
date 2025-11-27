#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "constants.h"
#include "structs.h"
#include "bh_collision.h"

/* ------------------------------------------------------------
 * Periodic helpers
 * ------------------------------------------------------------ */

/* Wrap coordinate into [0, L) */
static inline double wrap_box(double x)
{
    x = fmod(x, L);
    if (x < 0.0) x += L;
    if (x >= L)  x -= L;
    return x;
}

/* Minimum-image separation in one dimension */
static inline double periodic_delta(double dx)
{
    if (dx >  0.5 * L) dx -= L;
    if (dx < -0.5 * L) dx += L;
    return dx;
}

/* Swap particle j into slot i (and keep all arrays consistent) */
static void swap_particle(ParticleSystem *sys, int i, int j)
{
    if (i == j) return;

    Vector3 tmp_pos = sys->positions[i];
    Vector3 tmp_vel = sys->velocities[i];
    Vector3 tmp_acc = sys->accelerations[i];
    double tmp_m = sys->masses[i];
    int tmp_t    = sys->types[i];

    sys->positions[i]     = sys->positions[j];
    sys->velocities[i]    = sys->velocities[j];
    sys->accelerations[i] = sys->accelerations[j];
    sys->masses[i]        = sys->masses[j];
    sys->types[i]         = sys->types[j];

    sys->positions[j]     = tmp_pos;
    sys->velocities[j]    = tmp_vel;
    sys->accelerations[j] = tmp_acc;
    sys->masses[j]        = tmp_m;
    sys->types[j]         = tmp_t;
}

/* After we swap in the last particle, if that last one was a BH,
 * fix its index in the bh_indices array.
 */
static void update_bh_indices_after_swap(int *bh_indices,
                                         int n_bh,
                                         int old_index,
                                         int new_index)
{
    for (int b = 0; b < n_bh; ++b) {
        if (bh_indices[b] == old_index) {
            bh_indices[b] = new_index;
            break;
        }
    }
}

/* ------------------------------------------------------------
 * Main BH collision step
 * ------------------------------------------------------------ */
void bh_collision_step(ParticleSystem *sys,
                       int *bh_indices,
                       int *n_bh)
{
    if (!sys || !bh_indices || !n_bh || *n_bh <= 0) {
        return;
    }

    int  N   = sys->N;
    int  nb  = *n_bh;
    int  n_coll = 0;

    const double R_star2 = BH_STAR_COLLISION_RADIUS * BH_STAR_COLLISION_RADIUS;
    const double R_bh2   = BH_BH_COLLISION_RADIUS   * BH_BH_COLLISION_RADIUS;

    if (N <= 0 || nb <= 0) return;

    /* --------------------------------------------------------
     * 1) BH¨Cstar accretion: swallow TYPE_STAR within R_star
     * -------------------------------------------------------- */
    for (int b = 0; b < nb; ++b) {
        int ibh = bh_indices[b];
        if (ibh < 0 || ibh >= sys->N) continue;

        double xbh = sys->positions[ibh].x;
        double ybh = sys->positions[ibh].y;
        double zbh = sys->positions[ibh].z;

        int i = 0;
        while (i < sys->N) {
            /* Skip BHs themselves */
            if (sys->types[i] == 2) {
                ++i;
                continue;
            }

            /* Only swallow stars (you can change this to include DM if desired) */
            if (sys->types[i] != 0) {
                ++i;
                continue;
            }

            double xi = sys->positions[i].x;
            double yi = sys->positions[i].y;
            double zi = sys->positions[i].z;

            double dx = periodic_delta(xi - xbh);
            double dy = periodic_delta(yi - ybh);
            double dz = periodic_delta(zi - zbh);
            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < R_star2 && n_coll < BH_MAX_COLLISIONS_PER_STEP) {
                /* Swallow star i into BH ibh */
                double m_bh = sys->masses[ibh];
                double m_i  = sys->masses[i];

                if (m_i > 0.0) {
                    /* Momentum-conserving velocity update */
                    double vx_bh = sys->velocities[ibh].x;
                    double vy_bh = sys->velocities[ibh].y;
                    double vz_bh = sys->velocities[ibh].z;

                    double vx_i  = sys->velocities[i].x;
                    double vy_i  = sys->velocities[i].y;
                    double vz_i  = sys->velocities[i].z;

                    double Mnew = m_bh + m_i;
                    if (Mnew > 0.0) {
                        sys->velocities[ibh].x =
                            (m_bh * vx_bh + m_i * vx_i) / Mnew;
                        sys->velocities[ibh].y =
                            (m_bh * vy_bh + m_i * vy_i) / Mnew;
                        sys->velocities[ibh].z =
                            (m_bh * vz_bh + m_i * vz_i) / Mnew;
                    }

                    /* Move BH slightly toward star (mass-weighted average) */
                    double x_new = wrap_box((m_bh * xbh + m_i * xi) / Mnew);
                    double y_new = wrap_box((m_bh * ybh + m_i * yi) / Mnew);
                    double z_new = wrap_box((m_bh * zbh + m_i * zi) / Mnew);

                    sys->positions[ibh].x = x_new;
                    sys->positions[ibh].y = y_new;
                    sys->positions[ibh].z = z_new;

                    sys->masses[ibh] = Mnew;

                    /* Remove star i from the system by swapping with last */
                    int last = sys->N - 1;
                    if (i != last) {
                        swap_particle(sys, i, last);
                        /* If we swapped in a BH from the end, fix its index */
                        update_bh_indices_after_swap(bh_indices, nb, last, i);
                    }
                    sys->N--;
                    N = sys->N;  /* keep local N consistent */

                    n_coll++;

                    /* After removing particle i, we don't increment i,
                       because a new particle has been swapped into slot i. */
                    continue;
                }
            }

            ++i;
        } /* end while(i) */

        /* Refresh BH position for next BH (it might have moved) */
        if (ibh >= 0 && ibh < sys->N) {
            xbh = sys->positions[ibh].x;
            ybh = sys->positions[ibh].y;
            zbh = sys->positions[ibh].z;
        }
    }

    /* --------------------------------------------------------
     * 2) BH¨CBH merging
     * -------------------------------------------------------- */
    int b = 0;
    while (b < nb) {
        int i1 = bh_indices[b];
        if (i1 < 0 || i1 >= sys->N) {
            /* This BH index is invalid; drop it from the list */
            bh_indices[b] = bh_indices[nb - 1];
            nb--;
            continue;
        }

        double x1 = sys->positions[i1].x;
        double y1 = sys->positions[i1].y;
        double z1 = sys->positions[i1].z;

        int b2 = b + 1;
        while (b2 < nb) {
            int i2 = bh_indices[b2];
            if (i2 < 0 || i2 >= sys->N) {
                bh_indices[b2] = bh_indices[nb - 1];
                nb--;
                continue;
            }

            double x2 = sys->positions[i2].x;
            double y2 = sys->positions[i2].y;
            double z2 = sys->positions[i2].z;

            double dx = periodic_delta(x2 - x1);
            double dy = periodic_delta(y2 - y1);
            double dz = periodic_delta(z2 - z1);
            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < R_bh2 && n_coll < BH_MAX_COLLISIONS_PER_STEP) {
                /* Merge BH at i2 into BH at i1 */
                double m1 = sys->masses[i1];
                double m2 = sys->masses[i2];
                double Mnew = m1 + m2;

                double vx1 = sys->velocities[i1].x;
                double vy1 = sys->velocities[i1].y;
                double vz1 = sys->velocities[i1].z;

                double vx2 = sys->velocities[i2].x;
                double vy2 = sys->velocities[i2].y;
                double vz2 = sys->velocities[i2].z;

                if (Mnew > 0.0) {
                    sys->velocities[i1].x =
                        (m1 * vx1 + m2 * vx2) / Mnew;
                    sys->velocities[i1].y =
                        (m1 * vy1 + m2 * vy2) / Mnew;
                    sys->velocities[i1].z =
                        (m1 * vz1 + m2 * vz2) / Mnew;

                    double x_new =
                        wrap_box((m1 * x1 + m2 * x2) / Mnew);
                    double y_new =
                        wrap_box((m1 * y1 + m2 * y2) / Mnew);
                    double z_new =
                        wrap_box((m1 * z1 + m2 * z2) / Mnew);

                    sys->positions[i1].x = x_new;
                    sys->positions[i1].y = y_new;
                    sys->positions[i1].z = z_new;

                    sys->masses[i1] = Mnew;
                }

                /* Remove BH at i2 from system */
                int last = sys->N - 1;
                if (i2 != last) {
                    swap_particle(sys, i2, last);
                    update_bh_indices_after_swap(bh_indices, nb, last, i2);
                }
                sys->N--;
                N = sys->N;

                /* Remove BH index entry for this merged BH */
                bh_indices[b2] = bh_indices[nb - 1];
                nb--;

                n_coll++;

                /* Do not increment b2; we need to examine the new entry
                   that was just moved into position b2. */
                continue;
            }

            ++b2;
        } /* end while(b2) */

        ++b;
    }

    *n_bh = nb;

    /* Optional debug: total collisions this call */
    /* printf("bh_collision_step: total collisions this step = %d\n", n_coll); */
}
