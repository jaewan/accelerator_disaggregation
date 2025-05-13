# DPDK Client-Server with GPU Acceleration - Architecture Overview

## 1. Purpose

This project implements a high-performance, reliable data transfer system between a client and a server using the Data Plane Development Kit (DPDK) for fast packet I/O. The server offloads packet payload processing to a NVIDIA GPU using CUDA for acceleration. The key goal is to achieve efficient data transfer from the network interface card (NIC) directly to the GPU on the server, minimizing CPU involvement and memory copies, particularly utilizing CUDA's pinned memory features for Zero-Copy access.

## 2. Architecture

The system consists of three main components:

- **Client (client.cpp)**: A DPDK application responsible for reading data from a file, segmenting it, establishing a reliable connection, sending the data over UDP/IP using a custom protocol, and handling acknowledgments and retransmissions.

- **Server (server.cpp)**: A DPDK application that receives packets from the client. It manages connection state, handles the custom reliability protocol (sending ACKs, handling packet order), and identifies data packets. Crucially, it uses a DPDK memory pool backed by CUDA-pinned host memory. Data packet payloads (residing in these pinned mbufs) are queued and handed off to a dedicated worker thread for asynchronous GPU processing.

- **GPU Kernel (server_cuda_kernel.cu)**: Contains the CUDA C++ code, including the kernel that runs on the GPU to process batches of packet payloads. It assumes it can directly access the payload data in the server's pinned host memory (Zero-Copy).

## 3. Key Technologies

### DPDK (Data Plane Development Kit)
Used on both client and server to bypass the kernel's network stack, allowing for fast packet reception and transmission directly from user space. Provides libraries for memory management (mempools), device interaction, and packet manipulation.

### CUDA (Compute Unified Device Architecture)
NVIDIA's platform for general-purpose computing on GPUs. Used on the server to write the packet processing kernel and manage GPU resources.

### Pinned (Page-Locked) Host Memory
Host memory allocated via cudaHostAlloc (with cudaHostAllocMapped) is locked in physical RAM and made directly accessible by the GPU over the PCIe bus. This enables:
- **Zero-Copy Access**: The GPU kernel can directly read/write mapped pinned host memory without explicit cudaMemcpy calls for the payload data itself.
- **High-Speed DMA**: If explicit copies were needed, transfers involving pinned memory are significantly faster than with standard pageable memory.

### Custom Reliable Protocol
Layered on top of UDP/IP. It provides basic TCP-like reliability features:
- Connection Setup (SYN / SYN-ACK)
- Connection Teardown (FIN / FIN-ACK)
- Sequenced Data Transfer
- Cumulative Acknowledgments (ACKs)
- Flow Control (Sliding Window)
- Retransmission on Timeout

## 4. Workflow

### Initialization:
- Client initializes DPDK, network port, parses arguments (server MAC/IP/Port, data file).
- Server initializes DPDK, network port, GPU (cuda_init), creates the special pinned memory mbuf pool, parses arguments (server IP/Port), and starts the gpu_worker_thread.

### Connection:
Client sends a SYN packet (CustomHeader+UDP+IP+Eth). Server receives SYN, creates connection state, sends SYN-ACK. Client receives SYN-ACK, marks connection as established.

### Data Transfer (Client):
1. Reads data chunk from file (up to MAX_PAYLOAD_SIZE).
2. Checks if the number of unacknowledged packets is within the window limit.
3. Calls create_packet to build packet (Eth+IP+UDP+CustomHeader+Payload).
4. Sends packet using rte_eth_tx_burst.
5. Stores packet info (mbuf, seq, time) in unacked_pkts map.
6. Updates next_seq_num.
7. Periodically calls process_acks to handle incoming ACKs.

### Data Reception (Server - Main Thread):
1. Receives packet burst using rte_eth_rx_burst.
2. For each packet: parses headers (Eth/IP/UDP/Custom).
3. Calls process_packet:
   - Handles SYN/FIN logic.
   - For DATA packets:
     - Checks sequence number.
     - If in-order: Updates expected sequence number, adds mbuf pointer to a temporary gpu_batch vector (for this burst), sends ACK. Processes any newly contiguous packets from the out-of-order buffer, adding them to gpu_batch.
     - If out-of-order (future): Buffers the mbuf in conn.out_of_order_pkts, sends duplicate ACK.
     - If duplicate (past): Sends duplicate ACK, frees mbuf.
   - Frees mbufs for SYN, FIN, duplicates immediately after handling.
4. After processing the burst, if the gpu_batch vector is not empty, creates batch_metadata, moves the vector into it, and enqueues the metadata for the GPU worker thread.

### GPU Processing (Server - GPU Worker Thread):
1. Waits for and dequeues batch_metadata from the batch_q.
2. Calls process_gpu_batch:
   - Allocates temporary host arrays for payload pointers and sizes.
   - Iterates through mbufs in the batch, extracts pointers to payloads (custom_hdr + 1) and sizes, populating the temporary arrays.
   - Calls cuda_process_packets (from .cu file), passing the host arrays of pointers/sizes.
   - Waits for cuda_process_packets to return (it synchronizes internally).
   - Frees all mbufs associated with the completed batch.
   - Frees the temporary host arrays.

### CUDA Kernel Execution (server_cuda_kernel.cu):
1. cuda_process_packets copies the host pointer/size arrays to GPU memory.
2. Launches process_packets_kernel.
3. Kernel threads access the GPU pointer array to get the host pointer for their assigned packet payload.
4. Kernel threads directly access/modify the payload data in pinned host memory using the retrieved host pointer (Zero-Copy).
5. cuda_process_packets synchronizes the stream and returns.

### ACK Processing (Client):
1. process_acks receives packets.
2. process_received_packet parses ACKs, updates last_ack_received and remote_window.
3. Removes acknowledged packets from unacked_pkts, freeing their mbufs.

### Retransmission (Client - Timer Thread):
1. Periodically checks unacked_pkts.
2. If a packet timeout expires, retransmits it and applies exponential backoff.

### Teardown:
Client sends FIN. Server receives FIN, sends FIN-ACK. Client waits for final ACKs/timeout.

### Cleanup:
Both client and server stop threads, free remaining mbufs/buffers, clean up CUDA resources (server), close DPDK ports, free mempools (including handling the pinned memory via cudaFreeHost on the server), and clean up EAL.

## 5. Performance Considerations

### Memory Bottlenecks
- The zero-copy approach eliminates PCIe transfers for payload data, but pointer arrays still require copying.
- Performance will be limited by PCIe bandwidth when GPU computation is minimal.

### CPU Overhead
- DPDK polling can consume significant CPU cycles - consider tuning the polling frequency.
- The server's asynchronous GPU offload reduces CPU involvement in payload processing.

### GPU Utilization
- For small payloads, GPU overhead might exceed benefits; batch processing mitigates this.
- Kernel efficiency depends on the actual processing workload implemented.

## 6. Assumptions

- CUDA toolkit and compatible NVIDIA drivers are installed on the server.
- DPDK is installed, configured (e.g., hugepages setup), and appropriate NIC drivers (like vfio-pci) are bound.
- The server GPU supports accessing mapped pinned host memory (most modern NVIDIA GPUs do).
- Basic network connectivity (Client/Server can reach each other via IP/MAC) is established. The current code assumes client and server are on the same L2 segment for MAC addressing to work directly.
- Sufficient hugepage memory is allocated for DPDK.

## 7. System Diagram
Client                                      Server
+------------------+                      +------------------+
| DPDK Application |                      | DPDK Application |
|                  |                      |                  |
| +-------------+  |                      | +-------------+  |
| | File Reader |  |                      | | Connection  |  |
| +-------------+  |                      | | Management  |  |
|       |          |                      | +-------------+  |
|       v          |                      |       |          |
| +-------------+  |  Network (UDP/IP)    | +-------------+  |
| | Reliability |  | <----------------->  | | Reliability |  |
| | Protocol    |  |                      | | Protocol    |  |
| +-------------+  |                      | +-------------+  |
|       |          |                      |       |          |
|       v          |                      |       v          |
| +-------------+  |                      | +-------------+  |
| | DPDK        |  |                      | | DPDK w/     |  |
| | Networking  |  |                      | | Pinned Pool |  |
| +-------------+  |                      | +-------------+  |
|       |          |                      |       |          |
|       v          |                      |       v          |
| +-------------+  |                      | +-------------+  |
| | NIC Driver  |  |                      | | NIC Driver  |  |
| +-------------+  |                      | +-------------+  |
|       |          |                      |       |          |
+-------|---------+                      +-------|---------+
|                                        |
v                                        v
[Network]                                [Network]
|
v
+-------------+
| GPU Worker  |
| Thread      |
+-------------+
|
v
+-------------+
| CUDA Kernel |
+-------------+
