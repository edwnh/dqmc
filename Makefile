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

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:%.o=%.d)

CFLAGS += -I$(MKLROOT)/include -I$(HDF_PREFIX)/include

CFLAGS += -DMKL_DIRECT_CALL_SEQ
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE
# CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
# CFLAGS += -qopenmp # to enable 2x threading, use -qopenmp

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

all: $(TARGET)

$(BUILD_DIR)/%.o : %.c
	@echo CC $@
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $< -c -o $@

$(TARGET): $(OBJS)
	@echo LD $@
	@$(CC) $^ -o $@ $(LDFLAGS)

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)
