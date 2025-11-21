CXX = g++  
CXXFLAGS = -std=c++14 -O2 -Wall -fopenmp  
TARGET = fft  
  
all: $(TARGET)  
  
$(TARGET): fft.cpp complex.h dit_fft.h inverse_dit.h denoise.h derivative.h validate_n_evaluate.h  
	$(CXX) $(CXXFLAGS) fft.cpp -o $(TARGET)  
  
run: $(TARGET)  
	export OMP_NUM_THREADS=8 && ./$(TARGET) 10 1  
  
clean:  
	rm -f $(TARGET) fft_output.txt ifft_output.txt