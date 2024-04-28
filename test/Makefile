CC = icx

CFLAGS = -std=gnu17 -fast -xCORE-AVX2 -axCOMMON-AVX512
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -DMKL_DIRECT_CALL_SEQ
CFLAGS += -I../src
# CFLAGS += -DUSE_CPLX  # make clean and uncomment to use complex numbers

OPTFLAGS = -fargument-noalias

LDFLAGS +=  -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group
# LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

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
	@${CC} -c ${CFLAGS} ${OPTFLAGS} $<

clean:
	rm -f *.o *.optrpt *.yaml *.tmp
