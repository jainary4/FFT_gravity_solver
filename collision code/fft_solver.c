#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>
#include <float.h>     
#include "constants.h"
#include "mesh.h"

// Solve Poisson equation using FFTW (complex<->complex variant).
// Input:  laplacian_phi_pad[i][j][k] = 4πGρ on padded N^3 grid
// Output: phi_pad[i][j][k] = gravitational potential (same N)
// Note: This implementation uses full complex->complex FFTs so the Fourier
//       arrays are a full N^3 complex grid (no r2c half-storage).
double*** solve_poisson_fftw(double ***laplacian_phi_pad, int N, double h) {

    int N3 = N * N * N;  // total number of cells

    // Allocate output potential array
    double ***phi_pad = allocate_3d_array(N);
    if (!phi_pad) {
        fprintf(stderr, "allocate_3d_array failed for phi_pad\n");
        return NULL;
    }

    // Allocate full complex arrays for FFTW (size N^3)
    fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (size_t)N3);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (size_t)N3);
    if (in == NULL || out == NULL) {
        fprintf(stderr, "FFTW malloc failed\n");
        if (in) fftw_free(in);
        if (out) fftw_free(out);
        return NULL;
    }

    // Copy real input (laplacian_phi_pad) into complex input array (imag = 0)
    // Parallelized
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int idx = IDX(i,j,k,N);
                in[idx][0] = laplacian_phi_pad[i][j][k]; // real part
                in[idx][1] = 0.0;                         // imag part
            }
        }
    }

    // Initialize FFTW threads and use maximum available OpenMP threads
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    // Create forward and backward complex-to-complex plans
    // Forward: in -> out (FFTW_FORWARD)
    // Backward: out -> in (FFTW_BACKWARD) so 'in' will hold inverse result
    fftw_plan forward  = fftw_plan_dft_3d(N, N, N, in, out, FFTW_FORWARD,  FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_dft_3d(N, N, N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (!forward || !backward) {
        fprintf(stderr, "FFTW plan creation failed\n");
        if (forward) fftw_destroy_plan(forward);
        if (backward) fftw_destroy_plan(backward);
        fftw_free(in);
        fftw_free(out);
        fftw_cleanup_threads();
        return NULL;
    }

    // Execute forward FFT: out = FFT(in)
    fftw_execute(forward);

    // Precompute a scale-aware small threshold for k^2 to protect division by near-zero.
    // We scale with machine epsilon and an estimate of maximum discrete k^2.
    double k2_max_est = 12.0 / (h * h);                 // approx upper bound for discrete k^2
    double threshold  = DBL_EPSILON * k2_max_est * 100.0; // safety factor 100

    // Apply Green's function in Fourier space for each mode.
    // out[idx] currently contains (4πGρ)^(k); overwrite it with φ^(k) = -(4πGρ)^(k) / k^2_discrete
    //
    // We iterate over all grid indices (0..N-1) and map them to signed frequency indices
    // via ki = (i <= N/2)  i : i - N, etc.
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int idx = IDX(i,j,k,N);

                // Map array indices to signed frequency indices (handles FFTW ordering)
                int ki = (i <= N/2) ? i : i - N;
                int kj = (j <= N/2) ? j : j - N;
                int kk = (k <= N/2) ? k : k - N;

                // Physical wave numbers (units: 1 / length)
                double kx = (2.0 * PI * ki) / (N * h);
                double ky = (2.0 * PI * kj) / (N * h);
                double kz = (2.0 * PI * kk) / (N * h);

                // Discrete Laplacian eigenvalue consistent with finite-difference stencil:
                // k^2_discrete = 4/h^2 [ sin^2(kx h / 2) + sin^2(ky h / 2) + sin^2(kz h / 2) ]
                double sx = sin(0.5 * kx * h);
                double sy = sin(0.5 * ky * h);
                double sz = sin(0.5 * kz * h);
                double k2 = 4.0 * (sx*sx + sy*sy + sz*sz) / (h * h);

                if (k2 > threshold) {
                    // Regular mode: invert discrete Laplacian
                    double factor = -1.0 / k2;
                    double re = out[idx][0];
                    double im = out[idx][1];
                    out[idx][0] = factor * re;
                    out[idx][1] = factor * im;
                } else {
                    // DC or near-DC: set to zero (fix gauge; potential has arbitrary additive constant)
                    out[idx][0] = 0.0;
                    out[idx][1] = 0.0;
                }
            }
        }
    }

    // Inverse FFT: in = IFFT(out)
    fftw_execute(backward);

    // Normalize inverse (FFTW does not normalize inverse transform)
    double norm = 1.0 / (double)N3;

    // Copy resulting real potential (real part of 'in') into phi_pad
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int idx = IDX(i,j,k,N);
                phi_pad[i][j][k] = in[idx][0] * norm; // use real part
            }
        }
    }

    // Cleanup FFTW resources
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(in);
    fftw_free(out);
    fftw_cleanup_threads();

    return phi_pad;
}
