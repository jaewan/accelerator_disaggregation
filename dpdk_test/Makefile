CC = g++
CUDA_CC = nvcc

# Directories
CUDA_PATH = /usr/local/cuda-11.8
GDRCOPY_PATH = $(HOME)/gdrcopy

# DPDK flags
DPDK_FLAGS = $(shell pkg-config --cflags --libs libdpdk)

# CUDA flags
CUDA_CFLAGS = -I$(CUDA_PATH)/include
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart

# GDRCopy flags
GDRCOPY_CFLAGS = -I$(GDRCOPY_PATH)/include
GDRCOPY_LDFLAGS = -L$(GDRCOPY_PATH)/lib -lgdrapi

# Common flags
COMMON_FLAGS = -Wall -O3 -std=c++11 -pthread

# Server CUDA kernel
SERVER_CUDA_SRC = server_cuda_kernel.cu
SERVER_CUDA_OBJ = server_cuda_kernel.o

# Main targets
server: server.o $(SERVER_CUDA_OBJ)
	$(CC) $(COMMON_FLAGS) -o $@ $^ $(DPDK_FLAGS) $(CUDA_LDFLAGS) $(GDRCOPY_LDFLAGS)

client: client.o
	$(CC) $(COMMON_FLAGS) -o $@ $^ $(DPDK_FLAGS)

# Object files
server.o: server.cpp
	$(CC) $(COMMON_FLAGS) $(DPDK_FLAGS) $(CUDA_CFLAGS) $(GDRCOPY_CFLAGS) -c -o $@ $<

client.o: client.cpp
	$(CC) $(COMMON_FLAGS) $(DPDK_FLAGS) -c -o $@ $<

$(SERVER_CUDA_OBJ): $(SERVER_CUDA_SRC)
	$(CUDA_CC) --compile -o $@ $< -I$(CUDA_PATH)/include

# Additional targets
all: server client

clean:
	rm -f server client *.o

.PHONY: all clean
