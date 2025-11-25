#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "structs.h"
#include "bh_collision.h"

/* ----------------- small helpers ----------------- */

/* Periodic minimum-image separation in box [0, L). */
static inline float periodic_delta(float dx)
{
    if (dx >  0.5f * L) dx -= L;
    if (dx < -0.5f * L) dx += L;
    return dx;
}

/* Map (x,y,z) to cell index on NMESH^3 grid with box size L. */
static int cell_index(float x, float y, float z,
                      int *ix, int *iy, int *iz)
{
    const int   N = NMESH;
    const float h = (float)(L / (float)NMESH);

    float xg = x / h;
    float yg = y / h;
    float zg = z / h;

    int i = (int)floorf(xg);
    int j = (int)floorf(yg);
    int k = (int)floorf(zg);

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
static float distance2_periodic(const Vector3 *a, const Vector3 *b)
{
    float dx = periodic_delta(b->x - a->x);
    float dy = periodic_delta(b->y - a->y);
    float dz = periodic_delta(b->z - a->z);
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
 *   - N_BH is tiny (~2¨C10), so effectively O(N).
 * -------------------------------------------------- */
static void compact_particle_system(ParticleSystem *sys,
                                    int           *bh_indices,
                                    int           *n_bh)
{
    int i = 0;
    while (i < sys->N) {
        if (sys->masses[i] <= 0.0f) {
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

                /* Move last particle into slot i. */
                sys->positions[i]  = sys->positions[last];
                sys->velocities[i] = sys->velocities[last];
                sys->masses[i]     = sys->masses[last];
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
 * -------------------------------------------------- */
void bh_collision_step(ParticleSystem *sys,
                       int           *bh_indices,
                       int           *n_bh)
{
    if (!sys || !bh_indices || !n_bh) return;
    if (*n_bh <= 0) return;

    const int   N_cells       = NMESH * NMESH * NMESH;
    const float h             = (float)(L / (float)NMESH);
    const float search_radius = (BH_STAR_COLLISION_RADIUS > BH_BH_COLLISION_RADIUS)
                                ? BH_STAR_COLLISION_RADIUS
                                : BH_BH_COLLISION_RADIUS;
    const float r2_star = BH_STAR_COLLISION_RADIUS * BH_STAR_COLLISION_RADIUS;
    const float r2_bh   = BH_BH_COLLISION_RADIUS   * BH_BH_COLLISION_RADIUS;

    /* --------- Build cell-linked list: head[cell], next[particle] --------- */

    int *head = (int*)malloc((size_t)N_cells * sizeof(int));
    int *next = (int*)malloc((size_t)sys->N   * sizeof(int));

    if (!head || !next) {
        if (head) free(head);
        if (next) free(next);
        return;
    }

    for (int c = 0; c < N_cells; ++c)
        head[c] = -1;

    for (int p = 0; p < sys->N; ++p) {
        if (sys->masses[p] <= 0.0f) {
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

        next[p]   = head[cell];
        head[cell] = p;
    }

    /* --------- Loop over BHs and process collisions --------- */

    for (int bi = 0; bi < *n_bh; ++bi) {
        int i_bh = bh_indices[bi];

        if (i_bh < 0 || i_bh >= sys->N) continue;
        if (sys->masses[i_bh] <= 0.0f)  continue;
        if (sys->types[i_bh] != TYPE_BH) continue;

        Vector3 pos_bh = sys->positions[i_bh];

        int ixBH, iyBH, izBH;
        (void)cell_index(pos_bh.x, pos_bh.y, pos_bh.z, &ixBH, &iyBH, &izBH);

        int nCellR = (int)ceilf(search_radius / h);

        for (int ix = ixBH - nCellR; ix <= ixBH + nCellR; ++ix) {
            if (ix < 0 || ix >= NMESH) continue;
            for (int iy = iyBH - nCellR; iy <= iyBH + nCellR; ++iy) {
                if (iy < 0 || iy >= NMESH) continue;
                for (int iz = izBH - nCellR; iz <= izBH + nCellR; ++iz) {
                    if (iz < 0 || iz >= NMESH) continue;

                    int cell = ix + NMESH * (iy + NMESH * iz);

                    for (int p = head[cell]; p != -1; p = next[p]) {
                        if (p == i_bh) continue;
                        if (sys->masses[p] <= 0.0f)  continue;

                        int t_p = sys->types[p];
                        if (t_p == TYPE_DM)          continue;

                        float r2 = distance2_periodic(&pos_bh, &sys->positions[p]);

                        if (t_p == TYPE_STAR) {
                            /* BH¨Cstar collision */
                            if (r2 <= r2_star) {
                                float Mbh_old = sys->masses[i_bh];
                                float Ms      = sys->masses[p];
                                float Mtot    = Mbh_old + Ms;

                                Vector3 v_bh = sys->velocities[i_bh];
                                Vector3 v_s  = sys->velocities[p];

                                sys->velocities[i_bh].x =
                                    (Mbh_old * v_bh.x + Ms * v_s.x) / Mtot;
                                sys->velocities[i_bh].y =
                                    (Mbh_old * v_bh.y + Ms * v_s.y) / Mtot;
                                sys->velocities[i_bh].z =
                                    (Mbh_old * v_bh.z + Ms * v_s.z) / Mtot;

                                sys->positions[i_bh].x =
                                    (Mbh_old * sys->positions[i_bh].x +
                                     Ms      * sys->positions[p].x) / Mtot;
                                sys->positions[i_bh].y =
                                    (Mbh_old * sys->positions[i_bh].y +
                                     Ms      * sys->positions[p].y) / Mtot;
                                sys->positions[i_bh].z =
                                    (Mbh_old * sys->positions[i_bh].z +
                                     Ms      * sys->positions[p].z) / Mtot;

                                sys->masses[i_bh] = Mtot;
                                /* Mark star for removal. */
                                sys->masses[p]    = 0.0f;
                            }
                        }
                        else if (t_p == TYPE_BH) {
                            /* BH¨CBH collision: avoid double counting (p > i_bh). */
                            if (p <= i_bh) continue;

                            if (r2 <= r2_bh) {
                                int host  = i_bh;
                                int guest = p;

                                if (sys->masses[p] > sys->masses[i_bh]) {
                                    host  = p;
                                    guest = i_bh;
                                }

                                float M1   = sys->masses[host];
                                float M2   = sys->masses[guest];
                                float Mtot = M1 + M2;

                                Vector3 v1 = sys->velocities[host];
                                Vector3 v2 = sys->velocities[guest];

                                sys->velocities[host].x =
                                    (M1 * v1.x + M2 * v2.x) / Mtot;
                                sys->velocities[host].y =
                                    (M1 * v1.y + M2 * v2.y) / Mtot;
                                sys->velocities[host].z =
                                    (M1 * v1.z + M2 * v2.z) / Mtot;

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
                                /* guest BH marked for removal */
                                sys->masses[guest] = 0.0f;
                            }
                        }
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
