BUILD_DIR = build
TARGET = $(BUILD_DIR)/dqmc

UNAME := $(shell uname)
HDF_VERSION = 2.0.0

ifeq ($(UNAME), Linux)
	CC = icx
	MKLROOT ?= $(CONDA_PREFIX)
	HDF_DIST = hdf5-$(HDF_VERSION)-ubuntu-2404_intel.tar.gz
else ifeq ($(UNAME), Darwin)
	CC = clang
	HDF_DIST = hdf5-$(HDF_VERSION)-macos14_clang.tar.gz
endif

HDF_URL = https://github.com/HDFGroup/hdf5/releases/download/$(HDF_VERSION)/$(HDF_DIST)
HDF_PREFIX ?= HDF5-$(HDF_VERSION)-$(UNAME)/HDF_Group/HDF5/$(HDF_VERSION)

SRCS = \
	src/main_1.c \
	src/mem.c \
	src/prof.c \
	src/sig.c \
	src/wrapper.c

# compiled twice, once for real and once for complex
SRCS_RC = \
	src/rc/data.c \
	src/rc/dqmc.c \
	src/rc/greens.c \
	src/rc/meas.c \
	src/rc/updates.c

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
OBJS_REAL = $(SRCS_RC:%.c=$(BUILD_DIR)/real/%.o)
OBJS_CPLX = $(SRCS_RC:%.c=$(BUILD_DIR)/cplx/%.o)

CFLAGS += -Isrc -I$(HDF_PREFIX)/include

CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
CFLAGS += -DPROFILE_ENABLE

CFLAGS += -std=gnu17 -O3
CFLAGS += -Wall -Wextra -Wno-unused-variable
CFLAGS += -MMD -MP

ifeq ($(UNAME), Linux)
	CFLAGS += -I$(MKLROOT)/include
	CFLAGS += -DMKL_DIRECT_CALL_SEQ
	CFLAGS += -mauto-arch=CORE-AVX2,CORE-AVX512,COMMON-AVX512 -fargument-noalias
	# CFLAGS += -qopenmp # uncomment to enable 2x threading

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
else ifeq ($(UNAME), Darwin)
	CFLAGS += -DACCELERATE_NEW_LAPACK
	CFLAGS += -Wno-format -Wno-incompatible-pointer-types-discards-qualifiers

	LDFLAGS += -framework Accelerate

	# statically link HDF5 libraries
	LDFLAGS += \
		$(HDF_PREFIX)/lib/libhdf5.a \
		$(HDF_PREFIX)/lib/libhdf5_hl.a \
		$(HDF_PREFIX)/lib/libaec.a \
		$(HDF_PREFIX)/lib/libszaec.a \
		$(HDF_PREFIX)/lib/libzlib-static.a

	# dynamically link HDF5 libraries
	# LDFLAGS += -lhdf5 -lhdf5_hl
endif

.PHONY: all clean deps

all: $(TARGET)

$(TARGET): $(OBJS) $(OBJS_REAL) $(OBJS_CPLX)
	@echo LD $@
	@$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o : %.c
	@echo CC $@
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $< -c -o $@

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

deps:
	@curl -fL "$(HDF_URL)" -o "$(HDF_DIST)" && \
	tar -xf "$(HDF_DIST)" && \
	tar -xf hdf5/HDF5-$(HDF_VERSION)-$(UNAME).tar.gz && \
	rm -rf hdf5/ "$(HDF_DIST)" && \
	echo "done"

-include $(OBJS:%.o=%.d) $(OBJS_REAL:%.o=%.d) $(OBJS_CPLX:%.o=%.d)
