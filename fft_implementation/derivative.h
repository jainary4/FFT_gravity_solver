
using namespace std;

complex* compute_derivative_using_fft(complex input_seq[], int N) {
    complex* fft_result = DIT_FFT_reordered(input_seq, N);

    // Step 2: Create a sequence of frequencies for multiplying in the frequency domain
    complex* derivative_fft_result = new complex[N];
    for (int k = 0; k < N; ++k) {
        // Multiply by j * 2 * pi * k (this is the frequency-domain equivalent of differentiation)
        complex multiplier;
        multiplier.re = 0;
        multiplier.im = (2 * PI * k);  // j * 2 * pi * k is represented by 0 + j * 2 * pi * k
        derivative_fft_result[k] = ComplexMul(fft_result[k], multiplier);
    }

    // Step 3: Perform the inverse DIT-FFT to obtain the derivative in the time domain
    complex* derivative_seq = DIT_IFFT_reordered(derivative_fft_result, N);

    // Clean up the memory
    delete[] fft_result;
    delete[] derivative_fft_result;

    return derivative_seq;
}