
complex* IFFT_iterative(complex input_seq[], int N, complex WN[]) {
    complex* return_seq = new complex[N];
    complex* current_seq = input_seq;

    // Iteratively compute FFT
    for (int stage = 1; stage <= log2(N); ++stage) {
        int m = pow(2, stage);  // Size of the current stage
        int half_m = m / 2;

        // Parallel for each group of m elements
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < half_m; ++j) {
                int idx1 = k + j;
                int idx2 = k + j + half_m;
                complex W = WN[(j * N) / m];  
                complex t = ComplexMul(current_seq[idx2], W);
                complex temp1 = ComplexAdd(current_seq[idx1], t);
                complex temp2 = ComplexSub(current_seq[idx1], t);
                return_seq[idx1] = temp1;
                return_seq[idx2] = temp2;
            }
        }
       
        std::swap(current_seq, return_seq);
    }

    return current_seq;
}



complex* DIT_IFFT_reordered(complex input_seq[], int N) {
    complex* reordered_seq = new complex[N];

    
    complex* WN_inverse = Calc_Inverse_WN(N);


    reordered_seq = reorder_seq(input_seq, N);


    // Perform Inverse FFT
    reordered_seq = IFFT_iterative(reordered_seq, N, WN_inverse);


    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        reordered_seq[i].re /= N;
        reordered_seq[i].im /= N;
    }
    return reordered_seq;
    delete[] WN_inverse;
}