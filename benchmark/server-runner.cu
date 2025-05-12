#include <unistd.h> 
#include <iostream>
#include <cstring>    
#include <sys/socket.h> 
#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"

#define BLOCK_SIZE 16

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void* cuda_host_alloc(size_t data_size) {
    // Allocate CUDA pinned memory
    // Copypasta from orogiinal code
    void *host_ptr;
    CUDA_CHECK(cudaHostAlloc(&host_ptr, data_size, cudaHostAllocMapped));
    std::cout << "Allocated CUDA pinned memory at " << host_ptr << std::endl;
    return host_ptr;
}

void* cuda_device_alloc(size_t data_size) {
    // Allocate GPU memory
    void *device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, DATA_SIZE));
    std::cout << "Allocated GPU memory at " << device_ptr << std::endl;
    return device_ptr;
}


// Copypasta from this dude's code:
//https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

void matmul_rpc(int client_fd) {

    void* a = cuda_host_alloc(DATA_SIZE);
    recvAll(client_fd, a, DATA_SIZE);

    std::cout << "First matrix received" << std::endl;

    void* b = cuda_host_alloc(DATA_SIZE);
    recvAll(client_fd, b, DATA_SIZE);

    std::cout << "Second matrix received" << std::endl;

    // Copypasta from the same source above
    unsigned int grid_rows = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Copypasta
    // Create CUDA stream for asynchronous transfers
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    //// Transfer data from pinned memory to GPU
    //CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, DATA_SIZE, cudaMemcpyHostToDevice, stream));
    //std::cout << "Started asynchronous transfer to GPU" << std::endl;

    // Synchronize the stream
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    //std::cout << "GPU transfer completed" << std::endl;

    //CUDA_CHECK(cudaStreamDestroy(stream));

    void* a_dev = cuda_device_alloc(DATA_SIZE);
    void* b_dev = cuda_device_alloc(DATA_SIZE);
    void* c_dev = cuda_device_alloc(DATA_SIZE);

     // I actually want a blocking interface for a more reliable performance metric.
    CUDA_CHECK(cudaMemcpy(a_dev, a, DATA_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev, b, DATA_SIZE, cudaMemcpyHostToDevice));

    gpu_square_matrix_mult<<<dimGrid, dimBlock>>>((int*) a_dev, (int*)b_dev, (int*)c_dev, DIM / 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "Computation done" << std::endl;

    void* c = cuda_host_alloc(DATA_SIZE); // Assume the matrix is square
    CUDA_CHECK(cudaMemcpy(c, c_dev, DATA_SIZE, cudaMemcpyDeviceToHost));

    sendAll(client_fd, c, DATA_SIZE);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}

int run_server(int client_fd) {

    
    char command = 1; // Debugging

    //ssize_t bytes_received = recv(client_fd, &command, 1, 0);

    //if (bytes_received <= 0) {
    //    perror("What happen ed");
    //    exit(1);
    //}

    //std::cout << command << "is the command";


    switch (command) {
        case 1:
            matmul_rpc(client_fd);
            break;
        default:
            exit(1);
            break;
    }

    return 0;
}