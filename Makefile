APP = pulp_llama2

TRAIN_LIB= /home/andrea/PULP-TrainLib-Tutorial/pulp-trainlib/lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources

APP_SRCS = main.c

APP_SRCS += net.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_rmsnorm_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c

APP_LDFLAGS += -lm

NUM_CORES ?= 8
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)

# enable performance counters:
#APP_CFLAGS += -DSTATS

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3	
APP_CFLAGS += -DDATA_TYPE=$(DATA_TYPE)


get_header:
	gcc genHeader.c -o genHeader
	./genHeader

include $(RULES_DIR)/pmsis_rules.mk
