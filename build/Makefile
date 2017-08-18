CC = icc
CC_MPI = mpiicc

CFLAGS = -std=gnu11 -Wall -Wextra -Ofast -xHost
CFLAGS += -DMKL_DIRECT_CALL_SEQ -mkl=sequential
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -qopenmp  # to disable openmp, use -qopenmp-stubs

LDFLAGS += -lhdf5 -lhdf5_hl

SRCFILES = data.o dqmc.o greens.o meas.o prof.o sig.o updates.o

all: one mpi

one: ${SRCFILES} main_1.o
	@echo linking dqmc_1
	@${CC} ${CFLAGS} -o dqmc_1 $? ${LDFLAGS}

mpi: ${SRCFILES} main_mpi.o
	@echo linking dqmc_mpi
	@${CC_MPI} ${CFLAGS} -o dqmc_mpi $? ${LDFLAGS}

%.o: ../src/%.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} $<

main_mpi.o: ../src/main_mpi.c
	@echo compiling $<
	@${CC_MPI} -c ${CFLAGS} $<

clean:
	rm -f *.o *.optrpt *.seq *.par
