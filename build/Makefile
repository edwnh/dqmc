CC = icx

CFLAGS = -std=gnu17 -fast -xHost
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -DMKL_DIRECT_CALL_SEQ
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
# CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -qopenmp-stubs # to enable 2x threading, use -qopenmp

LDFLAGS += -lhdf5 -lhdf5_hl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

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
	@${CC} -c ${CFLAGS} $<

clean:
	rm -f *.o *.optrpt *.seq *.par
