CC = clang

# Homebrew libomp prefix (keg-only). Make picks it up at build time.
LIBOMP_PREFIX := $(shell brew --prefix libomp)

# Homebrew FFTW prefix (fftw formula)
FFTW_PREFIX := $(shell brew --prefix fftw)

# Compiler flags: enable OpenMP for clang and point to libomp headers
CFLAGS = -O3 -march=native -std=c11 -Wall \
		 -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include \
		 -I$(FFTW_PREFIX)/include

# Linker flags: point to libomp library
LDFLAGS = -L$(LIBOMP_PREFIX)/lib -lomp \
		 -L$(FFTW_PREFIX)/lib -lfftw3 -lfftw3_threads

SRCS = main.c particles.c mesh.c random.c fft_solver.c poission.c force.c
OBJS = $(SRCS:.c=.o)
TARGET = sim
SUPLEMENT1= force_x.txt 
SUPLEMENT2= force_y.txt 
SUPLEMENT3= force_z.txt

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) $(SUPLEMENT1) $(SUPLEMENT2) $(SUPLEMENT3)

.PHONY: all clean