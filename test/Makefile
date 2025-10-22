UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	CC = icx
	MKLROOT ?= $(CONDA_PREFIX)
	CFLAGS += -I$(MKLROOT)/include
	CFLAGS += -DMKL_DIRECT_CALL_SEQ
	CFLAGS += -mauto-arch=CORE-AVX2,CORE-AVX512,COMMON-AVX512 -fargument-noalias

	# statically link MKL libraries
	LDFLAGS += -Wl,--start-group \
		$(MKLROOT)/lib/libmkl_intel_lp64.a \
		$(MKLROOT)/lib/libmkl_sequential.a \
		$(MKLROOT)/lib/libmkl_core.a \
		-Wl,--end-group \
		-lpthread

	# dynamically link MKL libraries
	# LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
else ifeq ($(UNAME), Darwin)
	CC = clang
	CFLAGS += -DACCELERATE_NEW_LAPACK
	CFLAGS += -Wno-format -Wno-incompatible-pointer-types-discards-qualifiers

	LDFLAGS += -framework Accelerate
endif

CFLAGS += -I../src

CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -std=gnu17 -O3
CFLAGS += -Wall -Wextra -Wno-unused-variable
CFLAGS += -MMD -MP

all: test_greens bench_linalg

bench_linalg: bench_linalg.o mem.o
	@echo linking $@
	@${CC} -o $@ $^ ${LDFLAGS}

test_greens: greens.o test_greens.o mem.o
	@echo linking $@
	@${CC} -o $@ $^ ${LDFLAGS}

vpath %.c ../src

%.o: %.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} $<

clean:
	rm -f *.o *.d test_greens bench_linalg

-include $(wildcard *.d)
