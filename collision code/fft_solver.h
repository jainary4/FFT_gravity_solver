#ifndef FFT_SOLVER_H
#define FFT_SOLVER_H


double*** solve_poisson_fftw(double ***laplacian_phi_pad, int N, double h);

#endif