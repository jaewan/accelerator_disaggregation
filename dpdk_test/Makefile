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
GDRCOPY_LDFLAGS = -L$(GDRCOPY_PATH)/lib 

# Common flags
COMMON_FLAGS = -Wall -O0 -std=c++17 -pthread -g

# Server CUDA kernel
SERVER_CUDA_SRC = server_cuda_kernel.cu
SERVER_CUDA_OBJ = server_cuda_kernel.o

# Main targets
server: server.o rpc-service.o $(SERVER_CUDA_OBJ)
	$(CC) $(COMMON_FLAGS) -o $@ $^ $(DPDK_FLAGS) $(CUDA_LDFLAGS) $(GDRCOPY_LDFLAGS)

client: client.o rpc-client.o
	$(CC) $(COMMON_FLAGS) -o $@ $^ $(DPDK_FLAGS)

# Object files
server.o: server.cpp future.hpp
	$(CC) $(COMMON_FLAGS) $(DPDK_FLAGS) $(CUDA_CFLAGS) $(GDRCOPY_CFLAGS) -c -o $@ $<

rpc-service.o: rpc-service.cpp
	$(CC) $(COMMON_FLAGS) $(DPDK_FLAGS) $(CUDA_CFLAGS) $(GDRCOPY_CFLAGS) -c -o $@ $<

client.o: client.cpp
	$(CC) $(COMMON_FLAGS) $(DPDK_FLAGS) -c -o $@ $<

rpc-client.o: rpc-client.cpp
	$(CC) $(COMMON_FLAGS) $(DPDK_FLAGS) $(CUDA_CFLAGS) $(GDRCOPY_CFLAGS) -c -o $@ $<

$(SERVER_CUDA_OBJ): $(SERVER_CUDA_SRC)
	$(CUDA_CC) --compile -o $@ $< -I$(CUDA_PATH)/include

# Additional targets
all: server client

clean:
	rm -f server client *.o

.PHONY: all clean
