#!/bin/bash

# Create a test file
dd if=/dev/urandom of=testdata.bin bs=1M count=10

# Get server MAC address from the server output
# or by running on the server: cat /sys/class/net/ens6/address
SERVER_MAC="XX:XX:XX:XX:XX:XX"

# Run client
sudo ./client -l 2-3 --proc-type=primary --file-prefix=client \
  -a 0000:a1:00.0 -- $SERVER_MAC testdata.bin
