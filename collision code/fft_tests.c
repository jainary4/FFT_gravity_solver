#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"   // for PI, maybe IDX, etc.
#include "fft_solver.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* 
 * Standalone 3D array allocator for tests.
 * Layout: contiguous block of N*N*N doubles, with triple-pointer indexing.
 * NOTE: This is used only for rhs and phi_exact. The phi returned by
 * solve_poisson_fftw is allocated via mesh.c (allocate_3d_array).
 */
static double*** alloc_3d(int N)
{
    double ***a  = (double***)malloc(N * sizeof(double**));
    if (!a) {
        fprintf(stderr, "alloc_3d: failed to allocate level-1\n");
        return NULL;
    }

    double **planes = (double**)malloc(N * N * sizeof(double*));
    if (!planes) {
        fprintf(stderr, "alloc_3d: failed to allocate level-2\n");
        free(a);
        return NULL;
    }

    double *data = (double*)calloc((size_t)N * N * N, sizeof(double));
    if (!data) {
        fprintf(stderr, "alloc_3d: failed to allocate data block\n");
        free(planes);
        free(a);
        return NULL;
    }

    for (int i = 0; i < N; ++i) {
        a[i] = &planes[i * N];
        for (int j = 0; j < N; ++j) {
            a[i][j] = &data[((size_t)i * N + j) * N];
        }
    }

    return a;
}

static void free_3d(double ***a, int N)
{
    (void)N;  // unused
    if (!a) return;
    double *data   = a[0][0];
    double **plane = a[0];
    free(data);
    free(plane);
    free(a);
}

/* Discrete Laplacian eigenvalue k^2 for a single mode (nx,ny,nz) 
 * with FFT grid size N and spacing h. This matches the implementation
 * in solve_poisson_fftw.
 */
static double mode_k2(int nx, int ny, int nz, int N, double h)
{
    int ki = nx;
    int kj = ny;
    int kk = nz;

    double kx = (2.0 * PI * ki) / (N * h);
    double ky = (2.0 * PI * kj) / (N * h);
    double kz = (2.0 * PI * kk) / (N * h);

    double sx = sin(0.5 * kx * h);
    double sy = sin(0.5 * ky * h);
    double sz = sin(0.5 * kz * h);

    double k2 = 4.0 * (sx*sx + sy*sy + sz*sz) / (h*h);
    return k2;
}

/* Compute relative L2 error and max abs error between two N^3 fields */
static void field_error(double ***phi_num, double ***phi_exact, int N,
                        double *err_L2_rel, double *err_max_abs)
{
    double num = 0.0;
    double den = 0.0;
    double max_abs = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:num,den) reduction(max:max_abs) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double exact = phi_exact[i][j][k];
                double diff  = phi_num[i][j][k] - exact;
                num += diff * diff;
                den += exact * exact;
                double adiff = fabs(diff);
                if (adiff > max_abs) max_abs = adiff;
            }
        }
    }

    *err_L2_rel  = (den > 0.0) ? sqrt(num / den) : sqrt(num);
    *err_max_abs = max_abs;
}

/* ---------------- Test 1: pure 1D cosine mode ---------------- */
/* phi(i,j,k) = cos(2¦Ð i / N), RHS = ?^2 phi = -k2 * phi
 * where k2 is the discrete eigenvalue for mode (1,0,0).
 */
static void test_fft_single_mode()
{
    printf("TEST: FFT Poisson solver - single cosine mode\n");

    const int N = 16;
    const double h = 1.0;  /* arbitrary; must be consistent with mode_k2 */

    double ***rhs       = alloc_3d(N);
    double ***phi_exact = alloc_3d(N);

    if (!rhs || !phi_exact) {
        fprintf(stderr, "Allocation failed in test_fft_single_mode\n");
        return;
    }

    int nx = 1, ny = 0, nz = 0;
    double k2 = mode_k2(nx, ny, nz, N, h);

    if (k2 == 0.0) {
        fprintf(stderr, "ERROR: k2=0 for mode (%d,%d,%d)\n", nx,ny,nz);
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double phase_x = 2.0 * PI * nx * i / (double)N;
        double cosx = cos(phase_x);
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double phi = cosx;  // depends only on i
                phi_exact[i][j][k] = phi;
                rhs[i][j][k]       = -k2 * phi;
            }
        }
    }

    /* Solve ?? ¦Õ = rhs */
    double ***phi_num = solve_poisson_fftw(rhs, N, h);
    if (!phi_num) {
        fprintf(stderr, "solve_poisson_fftw returned NULL in test_fft_single_mode\n");
        free_3d(rhs, N);
        free_3d(phi_exact, N);
        return;
    }

    double err_L2_rel, err_max_abs;
    field_error(phi_num, phi_exact, N, &err_L2_rel, &err_max_abs);

    printf("  k2 (mode 1,0,0) = %.6e\n", k2);
    printf("  L2 relative error = %.3e\n", err_L2_rel);
    printf("  max abs error     = %.3e\n", err_max_abs);

    /* Tolerances: double-precision FFT + trig gives ~1e-8 here;
       demanding <1e-6 is comfortably strict but realistic. */
    if (err_L2_rel < 1e-6 && err_max_abs < 1e-6) {
        printf("  PASS (FFT solver reproduces single-mode solution)\n");
    } else {
        printf("  FAIL (FFT solver error too large for single mode)\n");
    }

    free_3d(rhs, N);
    free_3d(phi_exact, N);
    /* phi_num was allocated by allocate_3d_array in mesh.c; we rely on
       program exit to clean it up here (small test, one-shot). */
}

/* ---------------- Test 2: superposition of two modes ---------------- */
/* phi = cos(2¦Ð i / N) + 0.5 cos(4¦Ð j / N)
 * mode1 = (1,0,0), mode2 = (0,2,0)
 */
static void test_fft_two_modes()
{
    printf("TEST: FFT Poisson solver - superposition of two modes\n");

    const int N = 16;
    const double h = 1.0;

    double ***rhs       = alloc_3d(N);
    double ***phi_exact = alloc_3d(N);

    if (!rhs || !phi_exact) {
        fprintf(stderr, "Allocation failed in test_fft_two_modes\n");
        return;
    }

    int n1x=1, n1y=0, n1z=0;
    int n2x=0, n2y=2, n2z=0;

    double k2_1 = mode_k2(n1x,n1y,n1z,N,h);
    double k2_2 = mode_k2(n2x,n2y,n2z,N,h);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double phase1_x = 2.0 * PI * n1x * i / (double)N;
        double cos1 = cos(phase1_x);

        for (int j = 0; j < N; ++j) {
            double phase2_y = 2.0 * PI * n2y * j / (double)N;
            double cos2 = cos(phase2_y);

            for (int k = 0; k < N; ++k) {
                double phi1 = cos1;
                double phi2 = 0.5 * cos2;
                double phi  = phi1 + phi2;

                phi_exact[i][j][k] = phi;

                double lap1 = -k2_1 * phi1;
                double lap2 = -k2_2 * phi2;
                rhs[i][j][k] = lap1 + lap2;
            }
        }
    }

    double ***phi_num = solve_poisson_fftw(rhs, N, h);
    if (!phi_num) {
        fprintf(stderr, "solve_poisson_fftw returned NULL in test_fft_two_modes\n");
        free_3d(rhs, N);
        free_3d(phi_exact, N);
        return;
    }

    double err_L2_rel, err_max_abs;
    field_error(phi_num, phi_exact, N, &err_L2_rel, &err_max_abs);

    printf("  L2 relative error = %.3e\n", err_L2_rel);
    printf("  max abs error     = %.3e\n", err_max_abs);

    if (err_L2_rel < 1e-6 && err_max_abs < 1e-6) {
        printf("  PASS (FFT solver reproduces two-mode superposition)\n");
    } else {
        printf("  FAIL (FFT solver error too large for two modes)\n");
    }

    free_3d(rhs, N);
    free_3d(phi_exact, N);
}

/* ---------------- Test 3: 3D mode ---------------- */
/* phi = cos(2¦Ð i / N) * cos(2¦Ð j / N) * cos(4¦Ð k / N)
 * mode = (1,1,2)
 */
static void test_fft_3d_mode()
{
    printf("TEST: FFT Poisson solver - full 3D mode\n");

    const int N = 16;
    const double h = 1.0;

    double ***rhs       = alloc_3d(N);
    double ***phi_exact = alloc_3d(N);

    if (!rhs || !phi_exact) {
        fprintf(stderr, "Allocation failed in test_fft_3d_mode\n");
        return;
    }

    int nx=1, ny=1, nz=2;
    double k2 = mode_k2(nx,ny,nz,N,h);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        double phase_x = 2.0 * PI * nx * i / (double)N;
        double cosx = cos(phase_x);
        for (int j = 0; j < N; ++j) {
            double phase_y = 2.0 * PI * ny * j / (double)N;
            double cosy = cos(phase_y);
            for (int k = 0; k < N; ++k) {
                double phase_z = 2.0 * PI * nz * k / (double)N;
                double cosz = cos(phase_z);

                double phi = cosx * cosy * cosz;
                phi_exact[i][j][k] = phi;
                rhs[i][j][k]       = -k2 * phi;
            }
        }
    }

    double ***phi_num = solve_poisson_fftw(rhs, N, h);
    if (!phi_num) {
        fprintf(stderr, "solve_poisson_fftw returned NULL in test_fft_3d_mode\n");
        free_3d(rhs, N);
        free_3d(phi_exact, N);
        return;
    }

    double err_L2_rel, err_max_abs;
    field_error(phi_num, phi_exact, N, &err_L2_rel, &err_max_abs);

    printf("  k2 (mode 1,1,2) = %.6e\n", k2);
    printf("  L2 relative error = %.3e\n", err_L2_rel);
    printf("  max abs error     = %.3e\n", err_max_abs);

    if (err_L2_rel < 1e-6 && err_max_abs < 1e-6) {
        printf("  PASS (FFT solver reproduces 3D mode)\n");
    } else {
        printf("  FAIL (FFT solver error too large for 3D mode)\n");
    }

    free_3d(rhs, N);
    free_3d(phi_exact, N);
}

/* ---------------- Test 4: DC mode (constant RHS) ---------------- */
/* A constant RHS should produce a potential whose Laplacian is that constant.
 * In a periodic domain, the true Poisson problem is ill-posed for non-zero mean,
 * so the solver should effectively kill the k=0 mode (set potential to 0 up to
 * numerical noise).
 */
static void test_fft_dc_mode()
{
    printf("TEST: FFT Poisson solver - DC (constant) RHS\n");

    const int N = 16;
    const double h = 1.0;
    const double rhs_val = 1.0;

    double ***rhs       = alloc_3d(N);
    double ***phi_exact = alloc_3d(N);

    if (!rhs || !phi_exact) {
        fprintf(stderr, "Allocation failed in test_fft_dc_mode\n");
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                rhs[i][j][k]       = rhs_val;
                phi_exact[i][j][k] = 0.0;   // expected gauge choice
            }
        }
    }

    double ***phi_num = solve_poisson_fftw(rhs, N, h);
    if (!phi_num) {
        fprintf(stderr, "solve_poisson_fftw returned NULL in test_fft_dc_mode\n");
        free_3d(rhs, N);
        free_3d(phi_exact, N);
        return;
    }

    double err_L2_rel, err_max_abs;
    field_error(phi_num, phi_exact, N, &err_L2_rel, &err_max_abs);

    printf("  L2 relative error (vs zero) = %.3e\n", err_L2_rel);
    printf("  max abs potential           = %.3e\n", err_max_abs);

    /* Here we just require potential to be numerically close to zero. */
    if (err_max_abs < 1e-10) {
        printf("  PASS (FFT solver correctly handles DC mode by zeroing potential)\n");
    } else {
        printf("  FAIL (FFT solver leaves large DC component in potential)\n");
    }

    free_3d(rhs, N);
    free_3d(phi_exact, N);
}

/* ---------------- main: run all tests ---------------- */

int main(void)
{
#ifdef _OPENMP
    printf("Running FFT tests with OpenMP, max threads = %d\n", omp_get_max_threads());
#endif

    test_fft_single_mode();
    printf("\n");

    test_fft_two_modes();
    printf("\n");

    test_fft_3d_mode();
    printf("\n");

    test_fft_dc_mode();

    return 0;
}
