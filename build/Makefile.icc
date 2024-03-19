CC = icc

CFLAGS = -std=gnu11 -Wall -Wextra -Ofast -xHost
CFLAGS += -DMKL_DIRECT_CALL_SEQ -mkl=sequential
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
# CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -qopenmp  # to disable openmp, use -qopenmp-stubs

LDFLAGS += -lhdf5 -lhdf5_hl

SRCFILES = data.o dqmc.o greens.o meas.o prof.o sig.o updates.o

all: one stack

one: ${SRCFILES} main_1.o
	@echo linking dqmc_1
	@${CC} ${CFLAGS} -o dqmc_1 $? ${LDFLAGS}

stack: ${SRCFILES} main_stack.o
	@echo linking dqmc_stack
	@${CC} ${CFLAGS} -o dqmc_stack $? ${LDFLAGS}

%.o: ../src/%.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} $<

clean:
	rm -f *.o *.optrpt *.seq *.par
