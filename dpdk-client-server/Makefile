# DPDK paths - adjust these according to your installation
DPDK_INC_PATH ?= /usr/local/include
DPDK_LIB_PATH ?= /usr/local/lib

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -g -Wall -Wextra $(shell pkg-config --cflags libdpdk)
LDFLAGS = $(shell pkg-config --libs libdpdk)

# Source files
SRCS = src/server.c src/client.c
OBJS = $(SRCS:.c=.o)

# Target executables
TARGETS = server client

all: $(TARGETS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

server: src/server.o
	$(CC) $^ -o $@ $(LDFLAGS)

client: src/client.o
	$(CC) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGETS)

.PHONY: all clean 