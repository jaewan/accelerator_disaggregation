#!/bin/bash

# Setup GDRCopy
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make
sudo ./insmod.sh

# ----------- Configure Hugepages for DPDK ----------- 
# # Reserve 4GB of hugepages (adjust as needed)
sudo echo 2048 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# ----------- Create mount point if it doesn't exist ----------- 
sudo mkdir -p /mnt/huge
sudo mount -t hugetlbfs nodev /mnt/huge

# ----------- Bind the Mellanox interface to DPDK ----------- 
# Unbind the interface from the kernel driver
sudo ifconfig ens6 down

# Bind it to the DPDK driver
sudo dpdk-devbind.py --bind=mlx5_core 0000:a1:00.0
