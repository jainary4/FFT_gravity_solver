# Makefile for N-body / BH collision project (Intel icc)

# ----------------------------------------------------------------------
# Compiler and flags
# ----------------------------------------------------------------------
CC      = icc
CFLAGS  = -std=c99 -O3 -Wall -Wextra -openmp
# For debugging, you can add:
# CFLAGS += -g

# Libraries common to all programs
LDLIBS  = -lm

# ----------------------------------------------------------------------
# FFTW installation
# (you said it's here: $(HOME)/A4FFTSolver/fftw-install)
# ----------------------------------------------------------------------
FFTW_PREFIX = $(HOME)/A4FFTSolver/fftw-install
FFTW_INC    = -I$(FFTW_PREFIX)/include
FFTW_LIBS   = -L$(FFTW_PREFIX)/lib -lfftw3 -lfftw3_omp

# ----------------------------------------------------------------------
# Object lists
# ----------------------------------------------------------------------

# Core BH collision code
BH_COLLISION_OBJS = bh_collision.o

# Simple BH example binary
BH_EXAMPLE_OBJS   = bh_collision_example.o $(BH_COLLISION_OBJS)

PM_DEBUG_OBJS = pm_energy_debug.o \
                mesh.o            \
                particles.o       \
                random.o          \
                poisson.o         \
                fft_solver.o      \
                force.o

# Unit-test style BH tests (uses particles + mesh + random)
BH_TESTS_OBJS     = bh_collision_tests.o \
                    bh_collision.o      \
                    particles.o         \
                    mesh.o              \
                    random.o

# Full main simulation (FFT solver etc.)
MAIN_OBJS         = main.o      \
                    mesh.o      \
                    particles.o \
                    random.o    \
                    poisson.o   \
                    fft_solver.o \
                    force.o     \
                    bh_collision.o \
                    integrator.o \
                    snapshot.o

# Symplectic integrator tests (no FFTW needed)
INTEGRATOR_TEST_OBJS = integrator_tests.o integrator.o

# ----------------------------------------------------------------------
# Default target: build everything useful
# ----------------------------------------------------------------------
.PHONY: all clean tests

all: main bh_collision_example bh_collision_tests integrator_tests

# ----------------------------------------------------------------------
# Executable targets
# ----------------------------------------------------------------------

bh_collision_example: $(BH_EXAMPLE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

bh_collision_tests: $(BH_TESTS_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Symplectic integrator tests (harmonic + Kepler, etc.)
integrator_tests: $(INTEGRATOR_TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

pm_energy_debug: $(PM_DEBUG_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(FFTW_LIBS)

# FFT tests (uses FFTW)
fft_tests: fft_tests.o fft_solver.o mesh.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(FFTW_LIBS)

# Full simulation (uses FFTW)
main: $(MAIN_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(FFTW_LIBS)

# ----------------------------------------------------------------------
# Object build rules
# ----------------------------------------------------------------------

# Generic rule for most .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# fft_solver needs FFTW headers
fft_solver.o: fft_solver.c
	$(CC) $(CFLAGS) $(FFTW_INC) -c $< -o $@

# ----------------------------------------------------------------------
# Housekeeping
# ----------------------------------------------------------------------

tests: bh_collision_tests integrator_tests
	./bh_collision_tests
	./integrator_tests

clean:
	rm -f *.o \
	      bh_collision_example \
	      bh_collision_tests \
	      integrator_tests \
	      pm_energy_debug \
	      fft_tests \
	      main
