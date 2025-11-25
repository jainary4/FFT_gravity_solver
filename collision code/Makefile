CC      = icc
CFLAGS  = -O3 -march=native -fp-model precise -std=c11 -qopenmp -Wall
LDFLAGS = -qopenmp -lm

# Adjust these to your actual filenames if needed
SRCS = \
    bh_collision_example.c \
    bh_collision.c

OBJS = $(SRCS:.c=.o)

all: bh_test

bh_test: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o bh_test
