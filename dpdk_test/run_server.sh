#!/bin/bash

# Run the server (GPU receiver)
sudo ./server -l 0-1 --proc-type=primary --file-prefix=server \
  -a 0000:a1:00.0 -- -p 0
