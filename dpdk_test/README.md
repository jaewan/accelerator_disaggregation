# Testing Instructions for DPDK GPU Communication Code

## Prerequisites and Setup

1. **Install GDRCopy** (if not already installed):
   ```bash
   make setup-gdrcopy
   ```

2. **Configure hugepages for DPDK**:
   ```bash
   # Use the setup target in the makefile
   make setup
   ```

3. **Bind the Mellanox ConnectX-5 interface to DPDK**:
   ```bash
   # Use the bind-mlx target in the makefile
   make bind-mlx
   ```

4. **Verify the binding**:
   ```bash
   python3 dpdk-devbind.py --status
   ```
   You should see your Mellanox card listed under a DPDK-compatible driver.

## Building the Code

Or just make it compilable from bazel

1. **Update the Makefile paths** if necessary:
   - Make sure `CUDA_PATH` points to your CUDA 11.8 installation. **I hardcoded it in Make. Probably need to reference path in .venv**
   - Make sure `GDRCOPY_PATH` points to your GDRCopy installation

2. **Build the code**:
   ```bash
   make all
   ```

3. **Generate test data**:
   ```bash
   make testdata
   ```

## Running the Single-Node Test

For testing on a single node, you'll run both the server and client components on the same machine.

### Terminal 1 (Server):

```bash
sudo ./server -l 0-1 --proc-type=primary --file-prefix=server \
  -a 0000:a1:00.0 -- -p 12346 -i 192.168.1.20
```

Notes:
- Replace the IP address with one that's valid on your network
- `-l 0-1` assigns CPU cores 0 and 1 to DPDK processes
- `-a 0000:a1:00.0` specifies the PCI address of your Mellanox card

### Terminal 2 (Client):

First, get the MAC address of the server:
```bash
# Check the MAC address the DPDK port is using
# Look at the output of the server when it starts
```

Then run the client:
```bash
sudo ./client -l 2-3 --proc-type=primary --file-prefix=client \
  -a 0000:a1:00.0 -- XX:XX:XX:XX:XX:XX testdata.bin 192.168.1.20 12346 192.168.1.10 12345
```

Replace:
- `XX:XX:XX:XX:XX:XX` with the MAC address of the server
- The IP addresses with valid ones for your network
- You may need to adjust the core assignments with `-l` based on your system

## Testing Between Two Physical Machines

### On the Server Machine:
```bash
sudo ./server -l 0-1 --proc-type=primary --file-prefix=server \
  -a <server_NIC_PCI_address> -- -p 12346 -i <server_ip>
```

### On the Client Machine:
```bash
sudo ./client -l 0-1 --proc-type=primary --file-prefix=client \
  -a <client_NIC_PCI_address> -- <server_mac> testdata.bin <server_ip> 12346 <client_ip> 12345
```

## Verifying Functionality

The code implements the following reliability features that you should observe working:

1. **Proper Network Layering**:
   - The packets now have proper Ethernet, IPv4, and UDP headers
   - The custom header is placed after these standard network layers

2. **Sequence Numbers and ACKs**:
   - Each packet has a unique sequence number
   - The receiver sends acknowledgments for received packets

3. **Sliding Window Flow Control**:
   - The `MAX_WINDOW_SIZE` constant controls the window size
   - The client tracks unacknowledged packets within this window
   - The server maintains a receive window

4. **Retransmission Timer**:
   - Lost packets are retransmitted
   - The client uses a dedicated timer thread

5. **Congestion Control**:
   - Exponential backoff for retransmissions

6. **Direct NIC-to-GPU Transfer**:
   - Packets are processed on the GPU
   - The mbuf reference counting ensures packets aren't freed before GPU processing completes
   - A separate GPU worker thread handles GPU operations asynchronously

## Debugging Tips

1. **Run in non-promiscuous mode first**:
   ```bash
   # Add this line to server.cpp after port initialization
   ret = rte_eth_promiscuous_disable(port_id);
   ```

2. **Check for mbuf leaks**:
   ```bash
   # Before exiting, add debug print in server.cpp
   printf("Mbuf pool count: %u\n", rte_mempool_avail_count(mbuf_pool));
   ```

3. **Use DPDK debug logs**:
   ```bash
   export RTE_LOG_LEVEL=DEBUG
   ```

4. **Use CUDA debugging**:
   ```bash
   # Add these lines to server_cuda_kernel.cu
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess) {
       printf("CUDA error: %s\n", cudaGetErrorString(error));
   }
   ```

5. **Initial testing without the GPU**:
   You can modify the server.cpp code to skip GPU processing initially by commenting out the call to `cuda_process_packets` in `process_gpu_batch`.

## Known Limitations

1. **IP Address Configuration**:
   - The code uses fixed IP addresses that need to be specified on the command line
   - In a production environment, you'd use address discovery

2. **Simplified Connection Management**:
   - The code doesn't handle all edge cases for connection setup/teardown
   - A production application would need more robust connection management

3. **Error Handling**:
   - The code has basic error handling, but a production application would need more comprehensive error handling

4. **Performance Tuning**:
   - The code is optimized for correctness and demonstration, not for maximum performance
   - Production code would need additional tuning for specific hardware

sudo ./server -l 0 --no-pci --huge-dir=/mnt/huge --vdev="net_pcap1,iface=ens6,rx_pcap=dpdk_server_rx.pcap,tx_pcap=dpdk_server_tx.pcap" --socket-mem=1024 --file-prefix=server

sudo ./client -l 1 -n 4 --no-pci --huge-dir=/mnt/huge --vdev="net_pcap0,iface=ens6,rx_pcap=dpdk_server_tx.pcap,tx_pcap=dpdk_server_rx.pcap" --socket-mem=1024 --file-prefix=client -- 42:01:0a:0a:01:0a testdata.bin