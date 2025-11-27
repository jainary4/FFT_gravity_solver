#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "structs.h"

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

/* 
 * Project particles onto x¨Cy plane, bin into imgN¡ÁimgN grid, and write a PGM image.
 *
 * - Uses particle mass as weight (so denser regions are brighter).
 * - Simple log scaling for dynamic range.
 * - Treats all particle types the same (stars, DM, BH all contribute to density).
 *
 * Returns 0 on success, -1 on error.
 */
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

    /* allocate image buffers */
    double *img = (double*)calloc((size_t)imgN * imgN, sizeof(double));
    unsigned char *buf = (unsigned char*)malloc((size_t)imgN * imgN * sizeof(unsigned char));
    if (!img || !buf) {
        fprintf(stderr, "write_xy_density_pgm: allocation failed\n");
        free(img);
        free(buf);
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

    /* find max density for scaling */
    double max_val = 0.0;
    for (int i = 0; i < imgN * imgN; ++i) {
        if (img[i] > max_val) max_val = img[i];
    }
    if (max_val <= 0.0) max_val = 1.0;

    /* convert to 8-bit grayscale with log stretch */
    const double LOG_BASE = 10.0;
    const double SCALE    = 9.0;    /* controls ¡°contrast¡± */

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < imgN * imgN; ++i) {
        double n = img[i] / max_val;      /* 0..1 */
        if (n < 0.0) n = 0.0;
        if (n > 1.0) n = 1.0;
        double v = log(1.0 + SCALE * n) / log(1.0 + SCALE);  /* 0..1 */
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        buf[i] = (unsigned char)(255.0 * v + 0.5);
    }

    /* write PGM (binary, P5) */
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
