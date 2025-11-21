complex* DIT_FFT_reordered(complex input_seq[], int N);
complex* DIT_FFT_iterative(complex input_seq[], int N, complex WN[]);

complex* DIT_FFT_reordered(complex input_seq[], int N) {
 
    complex* reordered_seq = reorder_seq(input_seq, N);

    complex* WN = Calc_WN(N);


    reordered_seq = DIT_FFT_iterative(reordered_seq, N, WN);

    return reordered_seq;
}  

complex* DIT_FFT_iterative(complex input_seq[], int N, complex WN[]) {
    complex* return_seq = new complex[N];
    complex* current_seq = input_seq;

    int stages = static_cast<int>(log2(N));
    
    for (int stage = 1; stage <= stages; ++stage) {
        int m = 1 << stage;  // 2^stage
        int half_m = m / 2;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N / 2; ++i) {
            int k = (i / half_m) * m;
            int j = i % half_m;
            int idx1 = k + j;
            int idx2 = k + j + half_m;

            complex W = WN[(j * N) / m];
            complex t = ComplexMul(current_seq[idx2], W);
            return_seq[idx1] = ComplexAdd(current_seq[idx1], t);
            return_seq[idx2] = ComplexSub(current_seq[idx1], t);
        }

        std::swap(current_seq, return_seq);
    }

    return current_seq;
}