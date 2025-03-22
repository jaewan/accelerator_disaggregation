#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_memzone.h>
#include <rte_extmem.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <unistd.h> // for sleep()

// DPDK configuration
#define RX_RING_SIZE 128
#define TX_RING_SIZE 512
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

// CUDA configuration
#define DATA_SIZE (1024 * 1024) // 1 MB of data
#define MBUF_DATA_SIZE RTE_MBUF_DEFAULT_BUF_SIZE

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define RTE_CHECK(call) \
    do { \
        int ret = call; \
        if (ret != 0) { \
            std::cerr << "DPDK error: " << rte_strerror(-ret) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(int argc, char *argv[]) {
    // Initialize DPDK EAL
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        std::cerr << "EAL initialization failed: " << rte_strerror(-ret) << std::endl;
        return -1;
    }
    argc -= ret;
    argv += ret;

    // Check available ports
    unsigned nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        std::cerr << "No Ethernet ports available" << std::endl;
        return -1;
    }
    std::cout << "Found " << nb_ports << " Ethernet ports" << std::endl;

    // Allocate CUDA pinned memory
    void *host_ptr;
    CUDA_CHECK(cudaHostAlloc(&host_ptr, DATA_SIZE, cudaHostAllocMapped));
    std::cout << "Allocated CUDA pinned memory at " << host_ptr << std::endl;

    // Register CUDA pinned memory as external memory for DPDK
    struct rte_memseg_list *msl;
    struct rte_memseg *ms;
    msl = rte_mem_virt2memseg_list(host_ptr);
    ms = rte_mem_virt2memseg(host_ptr, msl);
    RTE_CHECK(rte_extmem_register(host_ptr, DATA_SIZE, nullptr, 0, rte_socket_id()));
    std::cout << "Registered CUDA pinned memory with DPDK" << std::endl;

    // Create memory pool using external memory
    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create_extbuf(
        "MBUF_POOL", NUM_MBUFS, MBUF_CACHE_SIZE, 0, MBUF_DATA_SIZE, rte_socket_id(),
        host_ptr, DATA_SIZE, nullptr, 0, nullptr, nullptr);
    if (mbuf_pool == nullptr) {
        std::cerr << "Cannot create mbuf pool" << std::endl;
        CUDA_CHECK(cudaFreeHost(host_ptr));
        return -1;
    }
    std::cout << "Created mbuf pool with external CUDA pinned memory" << std::endl;

    // Configure the first port
    unsigned port_id = 0;
    struct rte_eth_conf port_conf = {0};
    port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;

    RTE_CHECK(rte_eth_dev_configure(port_id, 1, 1, &port_conf));

    // Setup RX queue with external memory pool
    struct rte_eth_rxconf rx_conf = {0};
    rx_conf.rx_drop_en = 1; // Drop packets if queue is full
    RTE_CHECK(rte_eth_rx_queue_setup(port_id, 0, RX_RING_SIZE, rte_eth_dev_socket_id(port_id), &rx_conf, mbuf_pool));

    // Setup TX queue (not used in this example, but required for port configuration)
    RTE_CHECK(rte_eth_tx_queue_setup(port_id, 0, TX_RING_SIZE, rte_eth_dev_socket_id(port_id), nullptr));

    // Start the port
    RTE_CHECK(rte_eth_dev_start(port_id));

    // Enable promiscuous mode
    RTE_CHECK(rte_eth_promiscuous_enable(port_id));

    // Allocate GPU memory
    void *device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, DATA_SIZE));
    std::cout << "Allocated GPU memory at " << device_ptr << std::endl;

    // Create CUDA stream for asynchronous transfers
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Receive packets with DPDK (NIC writes directly to CUDA pinned buffer)
    size_t received_bytes = 0;
    while (received_bytes < DATA_SIZE) {
        struct rte_mbuf *pkts[BURST_SIZE];
        uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, pkts, BURST_SIZE);

        for (uint16_t i = 0; i < nb_rx; i++) {
            size_t pkt_len = rte_pktmbuf_pkt_len(pkts[i]);
            char *pkt_data = rte_pktmbuf_mtod(pkts[i], char*);

            // NIC has already written data to CUDA pinned buffer (no memcpy needed)
            received_bytes += pkt_len;

            rte_pktmbuf_free(pkts[i]);
        }

        // Optional: Add a small sleep to avoid busy-waiting
        if (nb_rx == 0) {
            usleep(10);
        }
    }

    std::cout << "Received " << received_bytes << " bytes of data" << std::endl;

    // Transfer data from pinned memory to GPU
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, DATA_SIZE, cudaMemcpyHostToDevice, stream));
    std::cout << "Started asynchronous transfer to GPU" << std::endl;

    // Synchronize the stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "GPU transfer completed" << std::endl;

    // Cleanup CUDA resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_ptr));
    RTE_CHECK(rte_extmem_unregister(host_ptr, DATA_SIZE));
    CUDA_CHECK(cudaFreeHost(host_ptr));

    // Cleanup DPDK resources
    RTE_CHECK(rte_eth_dev_stop(port_id));
    RTE_CHECK(rte_eth_dev_close(port_id));
    rte_mempool_free(mbuf_pool);

    // Cleanup DPDK EAL
    rte_eal_cleanup();

    std::cout << "Cleanup completed successfully" << std::endl;
    return 0;
}
