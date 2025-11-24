CC = clang

# Homebrew libomp prefix (keg-only). Make picks it up at build time.
LIBOMP_PREFIX := $(shell brew --prefix libomp)

# Compiler flags: enable OpenMP for clang and point to libomp headers
CFLAGS = -O3 -march=native -std=c11 -Wall \
		 -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include

# Linker flags: point to libomp library
LDFLAGS = -L$(LIBOMP_PREFIX)/lib -lomp

SRCS = main.c particles.c mesh.c random.c
OBJS = $(SRCS:.c=.o)
TARGET = sim

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
