#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "structs.h"
#include "snapshot.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* Wrap coordinate into [0, L) */
static inline double wrap_box(double x)
{
    x = fmod(x, L);
    if (x < 0.0) x += L;
    if (x >= L)  x -= L;
    return x;
}

/* ---------- grayscale version (old) ---------- */

int write_xy_density_pgm(const ParticleSystem *sys, int imgN, const char *filename)
{
    if (!sys || imgN <= 0 || !filename) {
        fprintf(stderr, "write_xy_density_pgm: invalid arguments\n");
        return -1;
    }

    const int Np = sys->N;
    if (Np <= 0) {
        fprintf(stderr, "write_xy_density_pgm: no particles\n");
        return -1;
    }

    double *img = (double*)calloc((size_t)imgN * imgN, sizeof(double));
    unsigned char *buf = (unsigned char*)malloc((size_t)imgN * imgN);
    if (!img || !buf) {
        fprintf(stderr, "write_xy_density_pgm: allocation failed\n");
        free(img);
        free(buf);
        return -1;
    }

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

    double max_val = 0.0;
    for (int i = 0; i < imgN * imgN; ++i)
        if (img[i] > max_val) max_val = img[i];
    if (max_val <= 0.0) max_val = 1.0;

    const double LOG_SCALE = 9.0;
    const double LOG_DEN   = log(1.0 + LOG_SCALE);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < imgN * imgN; ++i) {
        double n = img[i] / max_val;     /* 0..1 */
        if (n < 0.0) n = 0.0;
        if (n > 1.0) n = 1.0;
        double v = log(1.0 + LOG_SCALE * n) / LOG_DEN;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        buf[i] = (unsigned char)(255.0 * v + 0.5);
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("write_xy_density_pgm: fopen");
        free(img);
        free(buf);
        return -1;
    }

    fprintf(f, "P5\n%d %d\n255\n", imgN, imgN);
    size_t written = fwrite(buf, 1, (size_t)imgN * imgN, f);
    if (written != (size_t)imgN * imgN) {
        fprintf(stderr, "write_xy_density_pgm: fwrite wrote %zu bytes\n", written);
    }

    fclose(f);
    free(img);
    free(buf);
    return 0;
}

/* ---------- color PPM version: star=white, BH=red ---------- */

/* simple log stretch 0..1 -> 0..1 */
static inline double log_stretch(double x)
{
    if (x <= 0.0) return 0.0;
    if (x > 1.0)  x = 1.0;
    const double S = 9.0;
    const double D = 1.0 / log(1.0 + S);
    return log(1.0 + S * x) * D;
}

/* BH membership test (N_BH is tiny, O(N_BH) is fine) */
static inline int is_bh_particle(int p, const int *bh_indices, int n_bh)
{
    for (int b = 0; b < n_bh; ++b) {
        if (bh_indices[b] == p) return 1;
    }
    return 0;
}

int write_xy_density_ppm_color(const ParticleSystem *sys,
                               const int *bh_indices,
                               int n_bh,
                               int imgN,
                               const char *filename)
{
    if (!sys || imgN <= 0 || !filename) {
        fprintf(stderr, "write_xy_density_ppm_color: invalid arguments\n");
        return -1;
    }

    const int Np = sys->N;
    if (Np <= 0) {
        fprintf(stderr, "write_xy_density_ppm_color: no particles\n");
        return -1;
    }

    size_t npix = (size_t)imgN * imgN;

    /* separate 2D ¡°densities¡±: stars, DM, BH */
    double *star_img = (double*)calloc(npix, sizeof(double));
    double *dm_img   = (double*)calloc(npix, sizeof(double));
    double *bh_img   = (double*)calloc(npix, sizeof(double));
    unsigned char *rgb = (unsigned char*)malloc(3 * npix);

    if (!star_img || !dm_img || !bh_img || !rgb) {
        fprintf(stderr, "write_xy_density_ppm_color: allocation failed\n");
        free(star_img);
        free(dm_img);
        free(bh_img);
        free(rgb);
        return -1;
    }

    /* deposit mass onto pixels */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < Np; ++p) {
        double x = wrap_box(sys->positions[p].x);
        double y = wrap_box(sys->positions[p].y);
        double m = sys->masses[p];
        if (m <= 0.0) continue;

        int ix = (int)(x / L * imgN);
        int iy = (int)(y / L * imgN);

        if (ix < 0)        ix = 0;
        if (ix >= imgN)    ix = imgN - 1;
        if (iy < 0)        iy = 0;
        if (iy >= imgN)    iy = imgN - 1;

        size_t idx = (size_t)iy * imgN + ix;

        int type = sys->types ? sys->types[p] : 0;

        /* choose channel: BH > DM > star */
        if ( (bh_indices && n_bh > 0 && is_bh_particle(p, bh_indices, n_bh))
#ifdef TYPE_BH
             || (type == TYPE_BH)
#endif
           ) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            bh_img[idx] += m;
#ifdef TYPE_DM
        } else if (type == TYPE_DM) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            dm_img[idx] += m;
#endif
        } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
            star_img[idx] += m;
        }
    }

    /* find maxima for scaling */
    double max_star = 0.0, max_dm = 0.0, max_bh = 0.0;
    for (size_t i = 0; i < npix; ++i) {
        if (star_img[i] > max_star) max_star = star_img[i];
        if (dm_img[i]   > max_dm)   max_dm   = dm_img[i];
        if (bh_img[i]   > max_bh)   max_bh   = bh_img[i];
    }
    if (max_star <= 0.0) max_star = 1.0;
    if (max_dm   <= 0.0) max_dm   = 1.0;
    if (max_bh   <= 0.0) max_bh   = 1.0;

    /* map to RGB:
     *  - stars: white (R=G=B)
     *  - DM: purple (R and B, little green)
     *  - BH: red
     *  - background: black (no mass)
     */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < imgN; ++iy) {
        for (int ix = 0; ix < imgN; ++ix) {
            size_t idx = (size_t)iy * imgN + ix;
            double s = star_img[idx] / max_star;   /* 0..1 */
            double d = dm_img[idx]   / max_dm;     /* 0..1 */
            double b = bh_img[idx]   / max_bh;     /* 0..1 */

            s = log_stretch(s);
            d = log_stretch(d);
            b = log_stretch(b);

            double red, green, blue;

            if (s <= 0.0 && d <= 0.0 && b <= 0.0) {
                /* pure background */
                red = green = blue = 0.0;
            } else {
                /* base star field: soft white */
                double star_red   = s;
                double star_green = s;
                double star_blue  = s;

                /* DM: purple (more red+blue, little green) */
                double dm_red   = 0.8 * d;
                double dm_green = 0.2 * d;
                double dm_blue  = 0.8 * d;

                /* BH: red */
                double bh_red   = b;
                double bh_green = 0.0;
                double bh_blue  = 0.0;

                red   = star_red   + dm_red   + bh_red;
                green = star_green + dm_green + bh_green;
                blue  = star_blue  + dm_blue  + bh_blue;

                /* if BH present, force red dominance */
                if (b > 0.0) {
                    if (red < 1.0) red = 1.0; /* saturate red */
                    green *= 0.3;            /* darken green */
                    blue  *= 0.3;            /* darken blue */
                }

                /* clamp */
                if (red   > 1.0) red   = 1.0;
                if (green > 1.0) green = 1.0;
                if (blue  > 1.0) blue  = 1.0;
                if (red   < 0.0) red   = 0.0;
                if (green < 0.0) green = 0.0;
                if (blue  < 0.0) blue  = 0.0;
            }

            rgb[3*idx + 0] = (unsigned char)(255.0 * red   + 0.5);
            rgb[3*idx + 1] = (unsigned char)(255.0 * green + 0.5);
            rgb[3*idx + 2] = (unsigned char)(255.0 * blue  + 0.5);
        }
    }

    /* Overpaint BHs with larger red dots so they are clearly visible */
    if (bh_indices && n_bh > 0) {
        /* radius in pixels: scale with resolution, min ~2¨C3 */
        int r_pix = imgN / 128;   /* e.g. 512 -> 4, 256 -> 2 */
        if (r_pix < 2) r_pix = 2;

        for (int bidx = 0; bidx < n_bh; ++bidx) {
            int p = bh_indices[bidx];
            if (p < 0 || p >= sys->N) continue;

            double x = wrap_box(sys->positions[p].x);
            double y = wrap_box(sys->positions[p].y);

            int ix = (int)(x / L * imgN);
            int iy = (int)(y / L * imgN);

            if (ix < 0)      ix = 0;
            if (ix >= imgN)  ix = imgN - 1;
            if (iy < 0)      iy = 0;
            if (iy >= imgN)  iy = imgN - 1;

            for (int dy = -r_pix; dy <= r_pix; ++dy) {
                int jy = iy + dy;
                if (jy < 0 || jy >= imgN) continue;
                for (int dx = -r_pix; dx <= r_pix; ++dx) {
                    int jx = ix + dx;
                    if (jx < 0 || jx >= imgN) continue;
                    /* optional: make it circular instead of square */
                    int ddx = dx;
                    int ddy = dy;
                    if (ddx*ddx + ddy*ddy > r_pix*r_pix) continue;

                    size_t idx_pix = (size_t)jy * imgN + jx;
                    rgb[3*idx_pix + 0] = 255;
                    rgb[3*idx_pix + 1] = 0;
                    rgb[3*idx_pix + 2] = 0;
                }
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("write_xy_density_ppm_color: fopen");
        free(star_img);
        free(dm_img);
        free(bh_img);
        free(rgb);
        return -1;
    }

    fprintf(f, "P6\n%d %d\n255\n", imgN, imgN);
    size_t written = fwrite(rgb, 3, npix, f);
    if (written != 3 * npix) {
        fprintf(stderr,
                "write_xy_density_ppm_color: fwrite wrote %zu bytes\n", written);
    }

    fclose(f);
    free(star_img);
    free(dm_img);
    free(bh_img);
    free(rgb);
    return 0;
}
