CC = icx

CFLAGS = -I../src -std=gnu17 -fast -xHost
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -DMKL_DIRECT_CALL_SEQ
# CFLAGS += -DUSE_CPLX  # make clean and uncomment to use complex numbers

LDFLAGS += -lhdf5 -lhdf5_hl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

all: test_greens bench_linalg

bench_linalg: mem.o bench_linalg.o
	@echo linking $@
	@${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

test_greens: greens.o mem.o test_greens.o
	@echo linking $@
	@${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

vpath %.c ../src ../test

%.o: %.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} $<

clean:
	rm -f *.o *.optrpt *.seq *.par
