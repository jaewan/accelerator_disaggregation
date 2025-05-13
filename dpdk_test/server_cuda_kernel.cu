// server_cuda_kernel.cu - CUDA kernel for processing packets on GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <vector>

// Flag definitions (must match the ones in server.cpp)
#define PKT_FLAG_DATA   0x0001  // Data packet
#define PKT_FLAG_ACK    0x0002  // Acknowledgement
#define PKT_FLAG_SYN    0x0004  // Synchronize sequence numbers
#define PKT_FLAG_FIN    0x0008  // Finish connection
#define PKT_FLAG_RST    0x0010  // Reset connection

// Structure containing information about packet batches
struct packet_batch_info {
    void **packet_addrs;    // Array of packet addresses
    int num_packets;        // Number of packets in the batch
    uint32_t *quit_flag;    // Flag indicating whether to quit processing
};

//// CUDA kernel for processing packets
//__global__ void process_packets_kernel(packet_batch_info *batch_info) {
//    // Skip if quit flag is set
//    if (*batch_info->quit_flag != 0) {
//        return;
//    }
//    
//    // Get thread ID
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    
//    // Process packets in parallel
//    if (tid < batch_info->num_packets) {
//        // Get packet address
//        unsigned char *pkt_addr = (unsigned char *)batch_info->packet_addrs[tid];
//        
//        // Skip Ethernet header (typically 14 bytes)
//        // In a real implementation, you would parse the full headers correctly
//        packet_header *hdr = (packet_header *)(pkt_addr + 14);
//        
//        // Only process DATA packets (ignore ACKs, SYNs, etc.)
//        if ((hdr->flags & PKT_FLAG_DATA) != 0) {
//            // Get pointer to payload (after header)
//            unsigned char *payload = (unsigned char *)(hdr + 1);
//            
//            // Example processing: Invert all bytes in the payload
//            // In a real application, you would perform actual data processing
//            for (uint32_t i = 0; i < hdr->payload_size; i++) {
//                payload[i] = ~payload[i];
//            }
//        }
//    }
//}

__global__ void kernel_add(const float* a, const float* b, float* out, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void kernel_relu(const float* a, const float* b, float* out, size_t n) {
    (void)b; // Unused.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) out[idx] = fmaxf(0.0f, a[idx]);
}

// Simplified matrix multiplication (1D indexing for 2D matmul)
__global__ void kernel_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = 0;
        for (int i = 0; i < K; ++i) {
            val += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = val;
    }
}

void kernel_add_wrapper(std::vector<void*> parents, float* C, size_t n) {
    dim3 threadsPerBlock(32, 32);  // 1024 threads in a block
    dim3 numBlocks((n + threadsPerBlock.x * threadsPerBlock.y - 1) /
                   (threadsPerBlock.x * threadsPerBlock.y));
    if (parents.size() != 2)
        return;
    kernel_add<<<numBlocks, threadsPerBlock>>>((float*) parents.at(0), (float*) parents.at(1), C, n);
}

void kernel_relu_wrapper(std::vector<void*> parents, float* C, size_t n) {
    dim3 threadsPerBlock(32, 32);  // 1024 threads in a block
    dim3 numBlocks((n + threadsPerBlock.x * threadsPerBlock.y - 1) /
                   (threadsPerBlock.x * threadsPerBlock.y));

    if (parents.size() != 1)
        return;
    kernel_add<<<numBlocks, threadsPerBlock>>>((float*) parents.at(0), nullptr, C, n);
}

void kernel_matmul_wrapper(std::vector<void*> parents, float* C, size_t n) {
    dim3 threadsPerBlock(32, 32);  // 1024 threads in a block
    dim3 numBlocks(1, 1);          // Only one block needed
    if (parents.size() != 2)
        return;
    kernel_matmul<<<numBlocks, threadsPerBlock>>>((float*) parents.at(0), (float*) parents.at(1), C, n, n, n);
}

//// This function will be called by the server to process a batch of packets
//extern "C" void cuda_process_packets(void *packets_info) {
//    packet_batch_info *host_info = (packet_batch_info *)packets_info;
//    
//    // Allocate device memory for batch info
//    packet_batch_info *dev_info;
//    cudaMalloc(&dev_info, sizeof(packet_batch_info));
//    
//    // Allocate device memory for packet address array
//    void **dev_packet_addrs;
//    cudaMalloc(&dev_packet_addrs, host_info->num_packets * sizeof(void *));
//    
//    // Copy packet addresses to device
//    cudaMemcpy(dev_packet_addrs, host_info->packet_addrs, 
//               host_info->num_packets * sizeof(void *), 
//               cudaMemcpyHostToDevice);
//    
//    // Allocate and copy quit flag
//    uint32_t *dev_quit_flag;
//    cudaMalloc(&dev_quit_flag, sizeof(uint32_t));
//    cudaMemcpy(dev_quit_flag, host_info->quit_flag, 
//               sizeof(uint32_t), cudaMemcpyHostToDevice);
//    
//    // Set up device batch info
//    packet_batch_info host_dev_info;
//    host_dev_info.packet_addrs = dev_packet_addrs;
//    host_dev_info.num_packets = host_info->num_packets;
//    host_dev_info.quit_flag = dev_quit_flag;
//    
//    // Copy batch info to device
//    cudaMemcpy(dev_info, &host_dev_info, 
//               sizeof(packet_batch_info), cudaMemcpyHostToDevice);
//    
//    // Launch kernel with enough threads to process all packets
//    // Using 256 threads per block
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (host_info->num_packets + threadsPerBlock - 1) / threadsPerBlock;
//    process_packets_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_info);
//    
//    // Wait for kernel to complete
//    cudaDeviceSynchronize();
//    
//    // Free device memory
//    cudaFree(dev_packet_addrs);
//    cudaFree(dev_quit_flag);
//    cudaFree(dev_info);
//}
//