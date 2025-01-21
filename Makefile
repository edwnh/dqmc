BUILD_DIR = build
TARGET = $(BUILD_DIR)/dqmc

CC = icx
MKLROOT ?= $(CONDA_PREFIX)
HDF_PREFIX = HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5/

SRCS = \
	src/data.c \
	src/dqmc.c \
	src/greens.c \
	src/main_1.c \
	src/meas.c \
	src/mem.c \
	src/prof.c \
	src/sig.c \
	src/updates.c

OBJS_REAL = $(SRCS:%.c=$(BUILD_DIR)/real/%.o)
OBJS_CPLX = $(SRCS:%.c=$(BUILD_DIR)/cplx/%.o)

CFLAGS += -I$(MKLROOT)/include -I$(HDF_PREFIX)/include

CFLAGS += -DMKL_DIRECT_CALL_SEQ
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
# CFLAGS += -qopenmp # uncomment to enable 2x threading

CFLAGS += -std=gnu17 -O3 -mauto-arch=CORE-AVX2,CORE-AVX512,COMMON-AVX512 -fargument-noalias
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -MMD -MP

# statically link HDF5 libraries
LDFLAGS += -Wl,--start-group \
	$(HDF_PREFIX)/lib/libhdf5.a \
	$(HDF_PREFIX)/lib/libhdf5_hl.a \
	$(HDF_PREFIX)/lib/libaec.a \
	$(HDF_PREFIX)/lib/libszaec.a \
	$(HDF_PREFIX)/lib/libzlib-static.a \
	-Wl,--end-group

# dynamically link HDF5 libraries
# LDFLAGS += -lhdf5 -lhdf5_hl

# statically link MKL libraries
LDFLAGS += -Wl,--start-group \
	$(MKLROOT)/lib/libmkl_intel_lp64.a \
	$(MKLROOT)/lib/libmkl_sequential.a \
	$(MKLROOT)/lib/libmkl_core.a \
	-Wl,--end-group \
    -lpthread

# dynamically link MKL libraries
# LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

.PHONY: all clean

all: $(TARGET).real $(TARGET).cplx

$(TARGET).real: $(OBJS_REAL)
	@echo LD $@
	@$(CC) $^ -o $@ $(LDFLAGS)

$(TARGET).cplx: $(OBJS_CPLX)
	@echo LD $@
	@$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/real/%.o : %.c
	@echo CC $@
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $< -c -o $@

$(BUILD_DIR)/cplx/%.o : %.c
	@echo CC $@
	@mkdir -p $(@D)
	@$(CC) -DUSE_CPLX $(CFLAGS) $< -c -o $@

clean:
	$(RM) -r $(BUILD_DIR)

-include $(OBJS_REAL:%.o=%.d) $(OBJS_CPLX:%.o=%.d)
