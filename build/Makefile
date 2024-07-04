CC = icx
CFLAGS = -std=gnu17 -O3 -xCORE-AVX2 -axCOMMON-AVX512
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -DMKL_DIRECT_CALL_SEQ
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
# CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
# CFLAGS += -qopenmp # to enable 2x threading, use -qopenmp

OPTFLAGS = -fargument-noalias

LDFLAGS += -lhdf5 -lhdf5_hl
LDFLAGS +=  -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group
# LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

OBJS = data.o dqmc.o greens.o meas.o mem.o prof.o sig.o updates.o

all: dqmc_1 dqmc_stack

dqmc_1: ${OBJS} main_1.o
	@echo linking $@
	@${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

dqmc_stack: ${OBJS} main_stack.o
	@echo linking $@
	@${CC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

%.o: ../src/%.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} ${OPTFLAGS} $<

clean:
	rm -f *.o *.optrpt *.yaml *.tmp
