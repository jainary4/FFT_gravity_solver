#include "complex.h"         // Definition of struct complex, Calculation of WN[]
#include "dit_fft.h"         // DIT-FFT
#include "inverse_fft.h"     // Inverse DIT-FFT
#include "denoise.h"
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <sys/time.h>
# define PI 3.1415926
using namespace std;

void tick ( struct timeval * t ) {
 gettimeofday (t , NULL ) ;
 }

double tock ( struct timeval * t ) {
struct timeval now ;
gettimeofday (& now , NULL ) ;
 return ( double ) ( now.tv_sec - t-> tv_sec ) +
((double ) ( now.tv_usec - t-> tv_usec ) /1000000.) ;
}

void save_to_file(const char* filename, const complex* data, int N, const string& label) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    file << "# " << label << " Results\n";
    file << "# Format: Index\tReal Part\tImaginary Part\n";
    for (int i = 0; i < N; ++i) {
        file << i << "\t" << data[i].re << "\t" << data[i].im << "\n";
    }
    file.close();
}

int main(int argc, char **argv) {
    // Ensure correct number of arguments
    if (argc !=2) {
        cout << "Usage: ./fft [k] [validate_or_evaluate]\n";
        cout << "Example:\n";
        cout << "  ./fft 10 1  # Validate DIT and Inverse DIT for N = 2^10\n";
        cout << "  ./fft 10 0  # Evaluate performance for DIT and Inverse DIT for N = 2^10\n";
        return 1;
    }

    // Get input arguments
    int k = atoi(argv[1]);                // Exponent of 2 for N = 2^k
    int mode = atoi(argv[2]);             // 1 for validate, 0 for evaluate
    int N = 1 << k;                       // Compute N = 2^k

    // Generate the input sequence
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);
    double T = 1.0;

    complex* input_seq = new complex[N];

    // Generate sequence: sin(50t) + cos(120t)
      #pragma omp parallel for default(none) shared(N, T, input_seq)
    for (int i = 0; i < N; i++) {
        double t_i = (i * T) / N;
        double real = sin(2 * PI * 50 * t_i) + cos(2 * PI * 120 * t_i);
        input_seq[i] = complex(real, 0);
    }
    if (mode == 1) { 
        cout << "\nValidating for N = " << N << "...\n";    
        complex* dit_fft_result = DIT_FFT_reordered(input_seq, N);
        save_to_file("fft_output.txt", dit_fft_result, N, "FFT");
        cout << "FFT results saved to fft_output.txt\n";
        complex* inverse_result = DIT_IFFT_reordered(dit_fft_result, N);
        save_to_file("ifft_output.txt", inverse_result, N, "Inverse FFT");

        delete[] dit_fft_result;
        delete[] inverse_result;
    }
}