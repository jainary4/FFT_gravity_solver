CC      = icc
CFLAGS  = -O3 -march=native -fp-model precise -std=c11 -qopenmp -Wall
LDFLAGS = -qopenmp -lm

SRCS_TEST = \
    bh_collision_tests.c \
    bh_collision.c \
    particles.c \
    mesh.c \
    random.c

OBJS_TEST = $(SRCS_TEST:.c=.o)

bh_tests: $(OBJS_TEST)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o bh_tests
