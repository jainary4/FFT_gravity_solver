#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "constants.h"
#include "structs.h"
#include "bh_collision.h"

/* ----------------- small helpers ----------------- */

/* Periodic minimum-image separation in box [0, L). */
static inline double periodic_delta(double dx)
{
    if (dx >  0.5 * L) dx -= L;
    if (dx < -0.5 * L) dx += L;
    return dx;
}

/* Map (x,y,z) to cell index on NMESH^3 grid with box size L. */
static int cell_index(double x, double y, double z,
                      int *ix, int *iy, int *iz)
{
    const int    N = NMESH;
    const double h = L / (double)NMESH;

    double xg = x / h;
    double yg = y / h;
    double zg = z / h;

    int i = (int)floor(xg);
    int j = (int)floor(yg);
    int k = (int)floor(zg);

    /* periodic wrap to [0, N-1] */
    i = (i % N + N) % N;
    j = (j % N + N) % N;
    k = (k % N + N) % N;

    if (ix) *ix = i;
    if (iy) *iy = j;
    if (iz) *iz = k;

    return i + N * (j + N * k);
}

/* Squared distance with periodic BCs. */
static double distance2_periodic(const Vector3 *a, const Vector3 *b)
{
    double dx = periodic_delta(b->x - a->x);
    double dy = periodic_delta(b->y - a->y);
    double dz = periodic_delta(b->z - a->z);
    return dx*dx + dy*dy + dz*dz;
}

/* --------------------------------------------------
 * Compact the particle system by removing all particles
 * with mass <= 0.0. Removal is done by swapping with the
 * last active particle and decrementing sys->N.
 *
 * The BH index list bh_indices[0..*n_bh-1] is kept consistent:
 *   - if a removed particle is a BH, it is removed from the list;
 *   - if the last particle is a BH and gets swapped into a new slot,
 *     its index in the list is updated.
 *
 * Complexity:
 *   - O(sys->N) per call, plus O(N_BH) per removed BH.
 *   - N_BH is tiny (~10), so effectively O(N).
 * -------------------------------------------------- */
static void compact_particle_system(ParticleSystem *sys,
                                    int           *bh_indices,
                                    int           *n_bh)
{
    int i = 0;
    while (i < sys->N) {
        if (sys->masses[i] <= 0.0) {
            int last = sys->N - 1;
            int type_i    = sys->types[i];
            int type_last = sys->types[last];

            /* If we are removing a BH, drop it from bh_indices. */
            if (type_i == TYPE_BH && bh_indices && n_bh) {
                for (int k = 0; k < *n_bh; ++k) {
                    if (bh_indices[k] == i) {
                        bh_indices[k] = bh_indices[*n_bh - 1];
                        (*n_bh)--;
                        break;
                    }
                }
            }

            if (i != last) {
                /* If the last particle is a BH, update its index in bh_indices. */
                if (type_last == TYPE_BH && bh_indices && n_bh) {
                    for (int k = 0; k < *n_bh; ++k) {
                        if (bh_indices[k] == last) {
                            bh_indices[k] = i;
                            break;
                        }
                    }
                }

                /* Swap particle data: positions, velocities, mass, type. */
                Vector3 tmp_pos = sys->positions[i];
                sys->positions[i] = sys->positions[last];
                sys->positions[last] = tmp_pos;

                Vector3 tmp_vel = sys->velocities[i];
                sys->velocities[i] = sys->velocities[last];
                sys->velocities[last] = tmp_vel;

                double tmp_m = sys->masses[i];
                sys->masses[i] = sys->masses[last];
                sys->masses[last] = tmp_m;

                sys->types[i]      = sys->types[last];
            }

            /* One fewer active particle. */
            sys->N--;
            /* Do NOT increment i: process the swapped-in particle next. */
        } else {
            ++i;
        }
    }
}

/* --------------------------------------------------
 * Main BH collision / merge step.
 *
 * Parallelisation strategy:
 *   - Build the cell-linked list (head[cell], next[p]) in parallel
 *     over all particles using OpenMP, with an atomic capture on
 *     head[cell] to avoid races.
 *   - The BH neighbour search and compaction are kept serial, because
 *     the number of BHs is tiny (~10) and these operations involve
 *     complex shared updates.
 * -------------------------------------------------- */
void bh_collision_step(ParticleSystem *sys,
                       int           *bh_indices,
                       int           *n_bh)
{
    if (!sys || !bh_indices || !n_bh) return;
    if (*n_bh <= 0 || sys->N <= 0)     return;

    const int    N_cells       = NMESH * NMESH * NMESH;
    const double h             = L / (double)NMESH;
    const double search_radius = (BH_STAR_COLLISION_RADIUS > BH_BH_COLLISION_RADIUS)
                                ? BH_STAR_COLLISION_RADIUS
                                : BH_BH_COLLISION_RADIUS;
    const double r2_star = BH_STAR_COLLISION_RADIUS * BH_STAR_COLLISION_RADIUS;
    const double r2_bh   = BH_BH_COLLISION_RADIUS   * BH_BH_COLLISION_RADIUS;

    /* --------- Build cell-linked list: head[cell], next[particle] --------- */

    int *head = (int*)malloc((size_t)N_cells * sizeof(int));
    int *next = (int*)malloc((size_t)sys->N   * sizeof(int));

    if (!head || !next) {
        free(head);
        free(next);
        return;
    }

    for (int c = 0; c < N_cells; ++c) {
        head[c] = -1;
    }

    #pragma omp parallel for
    for (int p = 0; p < sys->N; ++p) {
        next[p] = -1;
    }

    /* We only put stars and BH into the cell list (DM is ignored). */
    #pragma omp parallel for
    for (int p = 0; p < sys->N; ++p) {
        if (sys->masses[p] <= 0.0) {
            next[p] = -1;
            continue;
        }

        int t = sys->types[p];
        if (t == TYPE_DM) {
            next[p] = -1;
            continue;
        }

        int ix, iy, iz;
        int cell = cell_index(sys->positions[p].x,
                              sys->positions[p].y,
                              sys->positions[p].z,
                              &ix, &iy, &iz);

        /* Atomic push-front into linked list for this cell. */
        #pragma omp atomic capture
        {
            next[p]  = head[cell];
            head[cell] = p;
        }
    }

    /* --------- For each BH, search neighbours within search_radius --------- */

    for (int ibh_list = 0; ibh_list < *n_bh; ++ibh_list) {
        int i_bh = bh_indices[ibh_list];

        if (i_bh < 0 || i_bh >= sys->N) continue;
        if (sys->masses[i_bh] <= 0.0)  continue;
        if (sys->types[i_bh] != TYPE_BH) continue;

        Vector3 pos_bh = sys->positions[i_bh];

        int ixBH, iyBH, izBH;
        (void)cell_index(pos_bh.x, pos_bh.y, pos_bh.z, &ixBH, &iyBH, &izBH);

        int nCellR = (int)ceil(search_radius / h);

        for (int ix = ixBH - nCellR; ix <= ixBH + nCellR; ++ix) {
            if (ix < 0 || ix >= NMESH) continue;
            for (int iy = iyBH - nCellR; iy <= iyBH + nCellR; ++iy) {
                if (iy < 0 || iy >= NMESH) continue;
                for (int iz = izBH - nCellR; iz <= izBH + nCellR; ++iz) {
                    if (iz < 0 || iz >= NMESH) continue;

                    int cell = ix + NMESH * (iy + NMESH * iz);

                    /* We use a (prev, curr) iteration pattern so we can
                     * unlink eaten particles from the cell list on the fly,
                     * without breaking the traversal.
                     */
                    int prev = -1;
                    int p    = head[cell];

                    while (p != -1) {
                        int curr   = p;
                        int next_p = next[curr];

                        /* Skip already-removed particles quickly. */
                        if (sys->masses[curr] <= 0.0) {
                            /* Unlink curr from list. */
                            if (prev == -1) head[cell] = next_p;
                            else            next[prev] = next_p;
                            p = next_p;
                            continue;
                        }

                        if (curr == i_bh) {
                            prev = curr;
                            p    = next_p;
                            continue;
                        }

                        int t_p = sys->types[curr];

                        if (t_p == TYPE_DM) {
                            prev = curr;
                            p    = next_p;
                            continue;
                        }

                        double r2 = distance2_periodic(&pos_bh, &sys->positions[curr]);

                        if (t_p == TYPE_STAR) {
                            /* BH-star collision */
                            if (r2 <= r2_star) {
                                double Mbh_old = sys->masses[i_bh];
                                double Ms      = sys->masses[curr];
                                double Mtot    = Mbh_old + Ms;

                                Vector3 v_bh = sys->velocities[i_bh];
                                Vector3 v_s  = sys->velocities[curr];

                                /* Momentum conservation: v_new = (M1 v1 + M2 v2)/Mtot */
                                sys->velocities[i_bh].x =
                                    (Mbh_old * v_bh.x + Ms * v_s.x) / Mtot;
                                sys->velocities[i_bh].y =
                                    (Mbh_old * v_bh.y + Ms * v_s.y) / Mtot;
                                sys->velocities[i_bh].z =
                                    (Mbh_old * v_bh.z + Ms * v_s.z) / Mtot;

                                /* Optional: shift BH position to center of mass. */
                                sys->positions[i_bh].x =
                                    (Mbh_old * sys->positions[i_bh].x +
                                     Ms      * sys->positions[curr].x) / Mtot;
                                sys->positions[i_bh].y =
                                    (Mbh_old * sys->positions[i_bh].y +
                                     Ms      * sys->positions[curr].y) / Mtot;
                                sys->positions[i_bh].z =
                                    (Mbh_old * sys->positions[i_bh].z +
                                     Ms      * sys->positions[curr].z) / Mtot;

                                sys->masses[i_bh] = Mtot;

                                /* Mark star for removal. */
                                sys->masses[curr] = 0.0;

                                /* Unlink curr from this cell's list. */
                                if (prev == -1) head[cell] = next_p;
                                else            next[prev] = next_p;

                                p = next_p;
                                continue;
                            }
                        }
                        else if (t_p == TYPE_BH) {
                            /* BH-BH collision: avoid double counting (curr > i_bh). */
                            if (curr > i_bh && r2 <= r2_bh) {
                                int host  = i_bh;
                                int guest = curr;

                                if (sys->masses[curr] > sys->masses[i_bh]) {
                                    host  = curr;
                                    guest = i_bh;
                                }

                                double M1   = sys->masses[host];
                                double M2   = sys->masses[guest];
                                double Mtot = M1 + M2;

                                Vector3 v1 = sys->velocities[host];
                                Vector3 v2 = sys->velocities[guest];

                                /* Momentum conservation */
                                sys->velocities[host].x =
                                    (M1 * v1.x + M2 * v2.x) / Mtot;
                                sys->velocities[host].y =
                                    (M1 * v1.y + M2 * v2.y) / Mtot;
                                sys->velocities[host].z =
                                    (M1 * v1.z + M2 * v2.z) / Mtot;

                                /* Center-of-mass position */
                                sys->positions[host].x =
                                    (M1 * sys->positions[host].x +
                                     M2 * sys->positions[guest].x) / Mtot;
                                sys->positions[host].y =
                                    (M1 * sys->positions[host].y +
                                     M2 * sys->positions[guest].y) / Mtot;
                                sys->positions[host].z =
                                    (M1 * sys->positions[host].z +
                                     M2 * sys->positions[guest].z) / Mtot;

                                sys->masses[host]  = Mtot;
                                sys->masses[guest] = 0.0;   /* guest BH removed */

                                /* We do not try to unlink guest from its own
                                 * cell list here (it may live in a different cell).
                                 * It will be skipped later because mass=0 and
                                 * removed by compaction at the end.
                                 */
                            }
                        }

                        prev = curr;
                        p    = next_p;
                    }
                }
            }
        }
    }

    free(head);
    free(next);

    /* Now physically remove all particles with mass <= 0.0,
       keeping bh_indices[] and *n_bh consistent. */
    compact_particle_system(sys, bh_indices, n_bh);
}
