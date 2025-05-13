// server.cpp - DPDK GPU Server with direct NIC to GPU transfer
// Compile with: g++ -o server server.cpp $(pkg-config --cflags --libs libdpdk) -lcuda -lgdrapi

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <rte_gpu.h>
#include <rte_gpu_comm.h>
#include <rte_timer.h>
#include <rte_lcore.h>
#include <rte_cycles.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_ether.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <signal.h>
#include <arpa/inet.h>
#include <map>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>

// Configuration constants
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define MAX_BURST_SIZE 32
#define MAX_PAYLOAD_SIZE 1400  // Max payload size to keep packet under MTU
#define MAX_WINDOW_SIZE 64     // Flow control window size
#define ACK_TIMEOUT_MS 10      // Timeout for ACK in milliseconds
#define MAX_RETRIES 5          // Maximum retransmission attempts
#define GPU_PAGE_SIZE (1UL << 16)
#define GPU_BATCH_SIZE 32      // Number of packets processed in one GPU batch

// Use pinned memory for DPDK mbuf pool
#define USE_PINNED_MEMORY_POOL 1
#define MEMPOOL_DATAROOM_SIZE (RTE_MBUF_DEFAULT_BUF_SIZE)

// Default port settings
#define DEFAULT_SERVER_PORT 12346
#define DEFAULT_SERVER_IP "192.168.1.20"

// DPDK port configuration
static struct rte_eth_conf port_conf = {
    .rxmode = {
        .mq_mode = RTE_ETH_MQ_RX_RSS,
        .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
    },
    .txmode = {
        .mq_mode = RTE_ETH_MQ_TX_NONE,
        .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM,
    },
};

// Custom packet header for reliability
struct packet_header {
    uint32_t seq_num;        // Sequence number
    uint32_t ack_num;        // Acknowledgement number 
    uint16_t flags;          // Control flags
    uint16_t window;         // Flow control window
    uint32_t payload_size;   // Size of payload in bytes
    uint32_t total_size;     // Total size of data being transferred
    uint32_t offset;         // Offset of this chunk in the total data
};

// Flag definitions
#define PKT_FLAG_DATA   0x0001  // Data packet
#define PKT_FLAG_ACK    0x0002  // Acknowledgement
#define PKT_FLAG_SYN    0x0004  // Synchronize sequence numbers
#define PKT_FLAG_FIN    0x0008  // Finish connection
#define PKT_FLAG_RST    0x0010  // Reset connection

// Connection identifier
struct connection_id {
    uint32_t client_ip;      // Client IP address
    uint16_t client_port;    // Client port
    
    bool operator<(const connection_id &other) const {
        if (client_ip != other.client_ip)
            return client_ip < other.client_ip;
        return client_port < other.client_port;
    }
};

// State structure for tracking connections
struct connection_state {
    uint32_t next_seq_num;            // Next expected sequence number
    uint32_t last_ack_sent;           // Last acknowledgement sent
    uint16_t recv_window;             // Current receive window size
    std::map<uint32_t, rte_mbuf*> out_of_order_pkts;  // Buffer for out-of-order packets
    std::mutex mutex;                 // Mutex for thread safety
    bool active;                      // Connection status
    struct rte_ether_addr client_mac; // Client MAC address
    uint32_t client_ip;               // Client IP address
    uint16_t client_port;             // Client port
    
    connection_state() : next_seq_num(0), last_ack_sent(0), 
                        recv_window(MAX_WINDOW_SIZE), active(true) {
        memset(&client_mac, 0, sizeof(client_mac));
    }
};

// Structure for batch metadata to synchronize between main thread and GPU worker
struct batch_metadata {
    std::vector<struct rte_mbuf *> mbufs;  // Mbufs in this batch
    bool processing_complete;              // Flag indicating GPU processing is complete
    
    batch_metadata() : processing_complete(false) {}
};

// Packet batch queue for communication between main thread and GPU worker
struct batch_queue {
    std::queue<std::shared_ptr<batch_metadata>> queue;  // Queue of batches
    std::mutex mutex;                                  // Mutex for thread safety
    std::condition_variable cv;                        // Condition variable for signaling
    std::atomic<bool> stop;                            // Stop flag for worker threads
    
    batch_queue() : stop(false) {}
};

// Global variables
static volatile bool force_quit = false;
static uint16_t port_id = 0;
static struct rte_mempool *mbuf_pool = NULL;
static int16_t gpu_dev_id = 0;
static std::map<connection_id, connection_state> connections;  // Client -> connection state
static std::mutex conn_mutex;
static uint32_t server_ip;            // Server IP address (network byte order)
static uint16_t server_port;          // Server UDP port (network byte order)
static std::thread gpu_worker_thread;  // GPU worker thread
static batch_queue batch_q;            // Batch queue for GPU processing
static std::atomic<uint64_t> processed_batches(0);  // Counter for batches processed

// Function prototypes
static void signal_handler(int signum);
static int init_eal(int argc, char **argv);
static int init_gpu(void);
static struct rte_mempool *create_pinned_mempool(const char *name, unsigned int n, 
                                               unsigned int cache_size, uint16_t priv_size,
                                               uint16_t data_room_size, int socket_id);
static int init_port(uint16_t port_id, struct rte_mempool *mbuf_pool);
static void process_packet(struct rte_mbuf *pkt, std::vector<struct rte_mbuf *> &gpu_batch);
static int send_ack(const connection_state &conn, uint32_t ack_num, uint16_t window, uint16_t flags);
static void gpu_worker_function(void);
static void process_gpu_batch(std::shared_ptr<batch_metadata> batch);
static void cleanup(void);

// CUDA functions imported from server_cuda_kernel.cu
extern "C" {
    // Processes a batch of packets on the GPU
    // Returns 0 on success, non-zero on error
    int cuda_process_packets(void **packet_data_ptrs, uint32_t *packet_sizes, int num_packets);
    
    // Initialize CUDA resources
    // Returns 0 on success, non-zero on error
    int cuda_init(int device_id);
    
    // Cleanup CUDA resources
    void cuda_cleanup(void);
}

// Main function
int main(int argc, char **argv) {
    int ret;
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse command line arguments
    uint16_t port = DEFAULT_SERVER_PORT;
    const char *ip_addr = DEFAULT_SERVER_IP;
    
    // Initialize EAL
    ret = init_eal(argc, argv);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error initializing EAL\n");
    }
    
    // Skip DPDK arguments
    argc -= ret;
    argv += ret;
    
    // Parse remaining arguments
    int opt;
    while ((opt = getopt(argc, argv, "p:i:")) != -1) {
        switch (opt) {
            case 'p':
                port = atoi(optarg);
                break;
            case 'i':
                ip_addr = optarg;
                break;
            default:
                printf("Usage: %s [EAL args] -- [-p port] [-i ip_address]\n", argv[0]);
                return -1;
        }
    }
    
    // Set server IP and port
    server_ip = inet_addr(ip_addr);
    server_port = htons(port);
    
    printf("Server initialized with:\n");
    printf("  IP: %s, Port: %d\n", ip_addr, port);
    
    // Create mempool for mbufs with pinned memory
#if USE_PINNED_MEMORY_POOL
    mbuf_pool = create_pinned_mempool("PINNED_MBUF_POOL", NUM_MBUFS,
                                   MBUF_CACHE_SIZE, 0, MEMPOOL_DATAROOM_SIZE, 
                                   rte_socket_id());
    if (mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create pinned mbuf pool\n");
    }
    printf("Using CUDA-pinned mempool for zero-copy access\n");
#else
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, 
        rte_socket_id());
    if (mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");
    }
    printf("Using standard DPDK mempool\n");
#endif
    
    // Initialize GPU
    ret = init_gpu();
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error initializing GPU\n");
    }
    
    // Initialize port
    ret = init_port(port_id, mbuf_pool);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error initializing port %d\n", port_id);
    }
    
    printf("Server initialized. Waiting for packets...\n");
    
    // Start GPU worker thread
    gpu_worker_thread = std::thread(gpu_worker_function);
    
    // Main packet processing loop
    while (!force_quit) {
        // Vector to collect packets for GPU processing
        std::vector<struct rte_mbuf *> gpu_batch;
        gpu_batch.reserve(MAX_BURST_SIZE);
        
        // Receive packets in burst
        struct rte_mbuf *rx_mbufs[MAX_BURST_SIZE];
        const uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, rx_mbufs, MAX_BURST_SIZE);
        
        if (nb_rx > 0) {
            // Process each received packet
            for (uint16_t i = 0; i < nb_rx; i++) {
                process_packet(rx_mbufs[i], gpu_batch);
            }
            
            // If we have packets for GPU processing, queue them
            if (!gpu_batch.empty()) {
                std::shared_ptr<batch_metadata> batch = std::make_shared<batch_metadata>();
                batch->mbufs = std::move(gpu_batch);
                
                // Add batch to queue for GPU processing
                {
                    std::lock_guard<std::mutex> lock(batch_q.mutex);
                    batch_q.queue.push(batch);
                }
                
                // Notify GPU worker thread
                batch_q.cv.notify_one();
            }
        }
        
        // Print statistics occasionally
        static uint64_t last_printed_batches = 0;
        uint64_t current_batches = processed_batches.load();
        if (current_batches >= last_printed_batches + 100) {
            printf("Processed %lu GPU batches\n", current_batches);
            last_printed_batches = current_batches;
        }
        
        // Give other processes a chance to run
        usleep(100);
    }
    
    // Cleanup and exit
    cleanup();
    
    return 0;
}

// Signal handler for clean termination
static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

// Initialize EAL
static int init_eal(int argc, char **argv) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
    }
    return ret;
}

// Create a mempool with CUDA-pinned memory
static struct rte_mempool *create_pinned_mempool(const char *name, unsigned int n, 
                                               unsigned int cache_size, uint16_t priv_size,
                                               uint16_t data_room_size, int socket_id) {
    struct rte_mempool *mp;
    size_t elt_size;
    void *addr;
    
    // Calculate the size of each element in the mempool
    elt_size = sizeof(struct rte_mbuf) + priv_size + data_room_size;
    
    // Round up to cache line size
    elt_size = RTE_ALIGN_CEIL(elt_size, RTE_CACHE_LINE_SIZE);
    
    // Create a mempool with external memory
    mp = rte_mempool_create_empty(name, n, elt_size, cache_size, priv_size,
                                socket_id, 0);
    if (mp == NULL) {
        RTE_LOG(ERR, USER1, "Cannot create mempool\n");
        return NULL;
    }
    
    // Allocate memory using cudaHostAlloc with the cudaHostAllocPortable flag
    // This ensures the memory is accessible by any GPU in the system
    cudaError_t cuda_err = cudaHostAlloc(&addr, n * elt_size,
                                      cudaHostAllocPortable | cudaHostAllocMapped);
    if (cuda_err != cudaSuccess) {
        RTE_LOG(ERR, USER1, "CUDA host alloc failed: %s\n", cudaGetErrorString(cuda_err));
        rte_mempool_free(mp);
        return NULL;
    }
    
    // Populate the mempool with the pinned memory
    if (rte_mempool_populate_default_bulk(mp, addr, n) < 0) {
        RTE_LOG(ERR, USER1, "Cannot populate mempool\n");
        cudaFreeHost(addr);
        rte_mempool_free(mp);
        return NULL;
    }
    
    // Set the mempool ops
    rte_mempool_set_ops_byname(mp, "ring_mp_mc", NULL);
    
    // Initialize the mbuf pool
    rte_pktmbuf_pool_init(mp, data_room_size);
    
    // Initialize each mbuf in the pool
    if (rte_mempool_obj_iter(mp, rte_pktmbuf_init, NULL) != 0) {
        RTE_LOG(ERR, USER1, "Cannot init mbufs\n");
        cudaFreeHost(addr);
        rte_mempool_free(mp);
        return NULL;
    }
    
    return mp;
}

// Initialize GPU and set up GPU memory
static int init_gpu(void) {
    int ret;
    
    // Get the number of available GPUs
    int16_t num_gpus = rte_gpu_count_avail();
    if (num_gpus <= 0) {
        RTE_LOG(ERR, USER1, "No GPU devices found\n");
        return -1;
    }
    
    // Use the first available GPU
    gpu_dev_id = 0;
    
    // Get the GPU device
    struct rte_gpu *gpu = rte_gpu_get_by_id(gpu_dev_id);
    if (gpu == NULL) {
        RTE_LOG(ERR, USER1, "Cannot get GPU device\n");
        return -1;
    }
    
    // Initialize CUDA resources
    ret = cuda_init(gpu_dev_id);
    if (ret != 0) {
        RTE_LOG(ERR, USER1, "Failed to initialize CUDA resources\n");
        return -1;
    }
    
    printf("GPU initialized: device_id=%d\n", gpu_dev_id);
    return 0;
}

// Initialize network port
static int init_port(uint16_t port_id, struct rte_mempool *mbuf_pool) {
    int ret;
    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf txconf;
    
    // Get device info
    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        RTE_LOG(ERR, USER1, "Error getting device info: %s\n", strerror(-ret));
        return ret;
    }
    
    // Configure TX
    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    
    // Configure the Ethernet device
    ret = rte_eth_dev_configure(port_id, 1, 1, &port_conf);
    if (ret != 0) {
        return ret;
    }
    
    // Adjust ring sizes if necessary
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rxd, &nb_txd);
    if (ret != 0) {
        return ret;
    }
    
    // Setup RX queue
    ret = rte_eth_rx_queue_setup(port_id, 0, nb_rxd,
                            rte_eth_dev_socket_id(port_id),
                            NULL, mbuf_pool);
    if (ret < 0) {
        return ret;
    }
    
    // Setup TX queue
    ret = rte_eth_tx_queue_setup(port_id, 0, nb_txd,
                            rte_eth_dev_socket_id(port_id),
                            &txconf);
    if (ret < 0) {
        return ret;
    }
    
    // Start the Ethernet port
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        return ret;
    }
    
    // Enable promiscuous mode
    ret = rte_eth_promiscuous_enable(port_id);
    if (ret < 0) {
        return ret;
    }
    
    return 0;
}

// Process a received packet
static void process_packet(struct rte_mbuf *pkt, std::vector<struct rte_mbuf *> &gpu_batch) {
    if (pkt == NULL) {
        return;
    }
    
    // Parse Ethernet header
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    if (eth_hdr->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
        // Not an IPv4 packet, ignore
        rte_pktmbuf_free(pkt);
        return;
    }
    
    // Parse IP header
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    if (ip_hdr->next_proto_id != IPPROTO_UDP) {
        // Not a UDP packet, ignore
        rte_pktmbuf_free(pkt);
        return;
    }
    
    // Parse UDP header
    struct rte_udp_hdr *udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    if (udp_hdr->dst_port != server_port) {
        // Not destined for our server port, ignore
        rte_pktmbuf_free(pkt);
        return;
    }
    
    // Parse packet header
    struct packet_header *hdr = (struct packet_header *)(udp_hdr + 1);
    
    // Create connection ID from client information
    connection_id conn_id;
    conn_id.client_ip = ip_hdr->src_addr;
    conn_id.client_port = udp_hdr->src_port;
    
    // Lock the connections map
    std::lock_guard<std::mutex> lock(conn_mutex);
    
    // Get or create connection state
    connection_state &conn = connections[conn_id];
    
    // Initialize connection if new
    if (!conn.active) {
        conn.active = true;
        conn.client_ip = ip_hdr->src_addr;
        conn.client_port = udp_hdr->src_port;
        rte_ether_addr_copy(&eth_hdr->src_addr, &conn.client_mac);
    }
    
    // Process based on packet flags
    if (hdr->flags & PKT_FLAG_SYN) {
        // New connection setup
        conn.next_seq_num = hdr->seq_num + 1;
        conn.last_ack_sent = hdr->seq_num;
        conn.active = true;
        
        // Send SYN-ACK
        send_ack(conn, conn.next_seq_num, conn.recv_window, PKT_FLAG_SYN | PKT_FLAG_ACK);
        printf("New connection from client %s:%d, initialized seq_num: %u\n",
               inet_ntoa(*(struct in_addr *)&conn.client_ip), 
               ntohs(conn.client_port), 
               conn.next_seq_num);
        
        // Free the SYN packet
        rte_pktmbuf_free(pkt);
    }
    else if (hdr->flags & PKT_FLAG_FIN) {
        // Connection termination
        conn.active = false;
        
        // Send FIN-ACK
        send_ack(conn, hdr->seq_num + 1, 0, PKT_FLAG_FIN | PKT_FLAG_ACK);
        printf("Connection closed from client %s:%d\n",
               inet_ntoa(*(struct in_addr *)&conn.client_ip), 
               ntohs(conn.client_port));
        
        // Free any buffered packets
        for (auto &pair : conn.out_of_order_pkts) {
            rte_pktmbuf_free(pair.second);
        }
        conn.out_of_order_pkts.clear();
        
        // Free the FIN packet
        rte_pktmbuf_free(pkt);
    }
    else if (hdr->flags & PKT_FLAG_DATA) {
        // Data packet
        std::lock_guard<std::mutex> conn_lock(conn.mutex);
        
        if (hdr->seq_num == conn.next_seq_num) {
            // In-order packet, process immediately
            conn.next_seq_num += hdr->payload_size;
            conn.last_ack_sent = conn.next_seq_num;
            
            // Add this packet to the GPU batch - IMPORTANT: DO NOT FREE THE PACKET YET!
            // Increase reference count to prevent it from being freed after GPU processing
            rte_pktmbuf_refcnt_update(pkt, 1);
            gpu_batch.push_back(pkt);
            
            // Check for any buffered packets that can now be processed
            auto it = conn.out_of_order_pkts.begin();
            while (it != conn.out_of_order_pkts.end()) {
                struct rte_mbuf *buf_pkt = it->second;
                struct rte_ether_hdr *buf_eth = rte_pktmbuf_mtod(buf_pkt, struct rte_ether_hdr *);
                struct rte_ipv4_hdr *buf_ip = (struct rte_ipv4_hdr *)(buf_eth + 1);
                struct rte_udp_hdr *buf_udp = (struct rte_udp_hdr *)(buf_ip + 1);
                struct packet_header *buf_hdr = (struct packet_header *)(buf_udp + 1);
                
                if (buf_hdr->seq_num == conn.next_seq_num) {
                    // This packet is now in order
                    conn.next_seq_num += buf_hdr->payload_size;
                    conn.last_ack_sent = conn.next_seq_num;
										// Add this packet to the GPU batch - DO NOT FREE YET!
                    gpu_batch.push_back(buf_pkt);

                    // Remove from buffer but don't free (it's now in the GPU batch)
                    auto to_erase = it;
                    ++it;
                    conn.out_of_order_pkts.erase(to_erase);
                } else if (buf_hdr->seq_num < conn.next_seq_num) {
                    // Duplicate packet, can be discarded
                    rte_pktmbuf_free(buf_pkt);
                    auto to_erase = it;
                    ++it;
                    conn.out_of_order_pkts.erase(to_erase);
                } else {
                    // Still out of order
                    ++it;
                }
            }

            // Send ACK for the latest processed packet
            send_ack(conn, conn.last_ack_sent, conn.recv_window, PKT_FLAG_ACK);
        } else if (hdr->seq_num > conn.next_seq_num) {
            // Out-of-order packet, buffer it
            conn.out_of_order_pkts[hdr->seq_num] = pkt;

            // Send duplicate ACK for the last in-order packet received
            send_ack(conn, conn.last_ack_sent, conn.recv_window, PKT_FLAG_ACK);
            printf("Out-of-order packet received from %s:%d, seq: %u, expected: %u\n",
                   inet_ntoa(*(struct in_addr *)&conn.client_ip),
                   ntohs(conn.client_port),
                   hdr->seq_num, conn.next_seq_num);

            // Don't free the packet as we've buffered it
        } else {
            // Duplicate packet, ignore but send ACK
            send_ack(conn, conn.last_ack_sent, conn.recv_window, PKT_FLAG_ACK);
            printf("Duplicate packet received from %s:%d, seq: %u\n",
                   inet_ntoa(*(struct in_addr *)&conn.client_ip),
                   ntohs(conn.client_port),
                   hdr->seq_num);

            // Free the duplicate packet
            rte_pktmbuf_free(pkt);
        }
    } else {
        // Other packet type (e.g., ACK)
        // Just free it as we don't need it for GPU processing
        rte_pktmbuf_free(pkt);
    }
}

// Send an acknowledgement packet
static int send_ack(const connection_state &conn, uint32_t ack_num, uint16_t window, uint16_t flags) {
    struct rte_mbuf *ack_pkt = rte_pktmbuf_alloc(mbuf_pool);
    if (ack_pkt == NULL) {
        RTE_LOG(ERR, USER1, "Failed to allocate mbuf for ACK\n");
        return -1;
    }

    // Calculate total packet size
    uint16_t total_length = sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr) +
                           sizeof(struct packet_header);

    // 1. Set up Ethernet header
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(ack_pkt, struct rte_ether_hdr *);
    rte_ether_addr_copy(&conn.client_mac, &eth_hdr->dst_addr);
    rte_eth_macaddr_get(port_id, &eth_hdr->src_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    // 2. Set up IP header
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    memset(ip_hdr, 0, sizeof(struct rte_ipv4_hdr));
    ip_hdr->version_ihl = RTE_IPV4_VHL_DEF;  // IP version 4, header length 5 (20 bytes)
    ip_hdr->total_length = rte_cpu_to_be_16(total_length);
    ip_hdr->time_to_live = 64;  // TTL
    ip_hdr->next_proto_id = IPPROTO_UDP;
    ip_hdr->src_addr = server_ip;
    ip_hdr->dst_addr = conn.client_ip;

    // Let hardware calculate IP checksum
    ack_pkt->ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM;

    // 3. Set up UDP header
    struct rte_udp_hdr *udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    udp_hdr->src_port = server_port;
    udp_hdr->dst_port = conn.client_port;
    udp_hdr->dgram_len = rte_cpu_to_be_16(sizeof(struct rte_udp_hdr) + sizeof(struct packet_header));

    // Let hardware calculate UDP checksum
    ack_pkt->ol_flags |= RTE_MBUF_F_TX_UDP_CKSUM;
    udp_hdr->dgram_cksum = 0;  // Set to 0 for hardware to fill in

    // 4. Set up packet header
    struct packet_header *hdr = (struct packet_header *)(udp_hdr + 1);
    memset(hdr, 0, sizeof(struct packet_header));
    hdr->ack_num = ack_num;
    hdr->flags = flags;
    hdr->window = window;

    // 5. Set the packet length
    ack_pkt->data_len = sizeof(struct rte_ether_hdr) + total_length;
    ack_pkt->pkt_len = ack_pkt->data_len;

    // Send the ACK packet
    uint16_t sent = rte_eth_tx_burst(port_id, 0, &ack_pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send ACK packet\n");
        rte_pktmbuf_free(ack_pkt);
        return -1;
    }

    return 0;
}

// GPU worker thread function
static void gpu_worker_function(void) {
    printf("GPU worker thread started\n");

    while (!force_quit) {
        // Wait for a batch to process
        std::shared_ptr<batch_metadata> batch;

        {
            std::unique_lock<std::mutex> lock(batch_q.mutex);
            batch_q.cv.wait(lock, [&]() {
                return !batch_q.queue.empty() || force_quit || batch_q.stop.load();
            });

            if (force_quit || batch_q.stop.load()) {
                break;
            }

            if (!batch_q.queue.empty()) {
                batch = batch_q.queue.front();
                batch_q.queue.pop();
            }
        }

        if (batch) {
            // Process the batch on the GPU
            process_gpu_batch(batch);

            // Increment the batch counter
            processed_batches++;
        }
    }

    printf("GPU worker thread exiting\n");
}

// Process a batch of packets on the GPU
static void process_gpu_batch(std::shared_ptr<batch_metadata> batch) {
    if (batch->mbufs.empty()) {
        return;
    }

    int num_packets = batch->mbufs.size();

    // Allocate arrays for packet data pointers and sizes
    void **packet_data_ptrs = (void **)malloc(num_packets * sizeof(void *));
    uint32_t *packet_sizes = (uint32_t *)malloc(num_packets * sizeof(uint32_t));

    if (!packet_data_ptrs || !packet_sizes) {
        RTE_LOG(ERR, USER1, "Failed to allocate memory for GPU batch\n");
        free(packet_data_ptrs);
        free(packet_sizes);

        // Free all mbufs in the batch
        for (auto &pkt : batch->mbufs) {
            rte_pktmbuf_free(pkt);
        }

        return;
    }

    // Extract packet data pointers and sizes
    for (int i = 0; i < num_packets; i++) {
        struct rte_mbuf *pkt = batch->mbufs[i];
        struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
        struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
        struct rte_udp_hdr *udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
        struct packet_header *custom_hdr = (struct packet_header *)(udp_hdr + 1);

        // Point to the payload data (after the custom header)
        packet_data_ptrs[i] = (void *)(custom_hdr + 1);
        packet_sizes[i] = custom_hdr->payload_size;
    }

    // Process packets on the GPU - with pinned memory, no explicit registration needed
    int ret = cuda_process_packets(packet_data_ptrs, packet_sizes, num_packets);
    if (ret != 0) {
        RTE_LOG(ERR, USER1, "GPU packet processing failed\n");
    }

    // Free all packet mbufs now that GPU processing is complete
    for (auto &pkt : batch->mbufs) {
        rte_pktmbuf_free(pkt);
    }

    // Mark the batch as processed
    batch->processing_complete = true;

    // Clean up
    free(packet_data_ptrs);
    free(packet_sizes);
}

// Cleanup all resources
static void cleanup(void) {
    // Stop the GPU worker thread
    batch_q.stop.store(true);
    batch_q.cv.notify_all();

    if (gpu_worker_thread.joinable()) {
        gpu_worker_thread.join();
    }

    // Clean up any pending GPU batches
    {
        std::lock_guard<std::mutex> lock(batch_q.mutex);
        while (!batch_q.queue.empty()) {
            auto batch = batch_q.queue.front();
            batch_q.queue.pop();

            // Free all mbufs in the batch
            for (auto &pkt : batch->mbufs) {
                rte_pktmbuf_free(pkt);
            }
        }
    }

    // Cleanup CUDA resources
    cuda_cleanup();

    // Free all buffered packets
    std::lock_guard<std::mutex> lock(conn_mutex);
    for (auto &conn_pair : connections) {
        std::lock_guard<std::mutex> conn_lock(conn_pair.second.mutex);
        for (auto &pkt_pair : conn_pair.second.out_of_order_pkts) {
            rte_pktmbuf_free(pkt_pair.second);
        }
        conn_pair.second.out_of_order_pkts.clear();
    }

    // Close the Ethernet port
    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);

    // Print mempool stats before freeing
    printf("Mbuf pool count: %u/%u available\n",
           rte_mempool_avail_count(mbuf_pool),
           rte_mempool_in_use_count(mbuf_pool) + rte_mempool_avail_count(mbuf_pool));

    // Get and free the pinned memory buffer if we used a pinned memory pool
#if USE_PINNED_MEMORY_POOL
    if (mbuf_pool != NULL) {
        // Get the mempool memory pointer from the mempool private data
        struct rte_mempool_memhdr *memhdr = STAILQ_FIRST(&mbuf_pool->mem_list);
        if (memhdr != NULL) {
            void *addr = memhdr->addr;

            // Free the mempool first (but this doesn't free the CUDA-pinned memory)
            rte_mempool_free(mbuf_pool);

            // Now free the CUDA-pinned memory
            cudaError_t cuda_err = cudaFreeHost(addr);
            if (cuda_err != cudaSuccess) {
                printf("CUDA error freeing host memory: %s\n", cudaGetErrorString(cuda_err));
            } else {
                printf("Freed CUDA-pinned mempool memory\n");
            }
        } else {
            rte_mempool_free(mbuf_pool);
        }
    }
#else
    // Free memory pool
    if (mbuf_pool != NULL) {
        rte_mempool_free(mbuf_pool);
    }
#endif

    // Clean up EAL
    rte_eal_cleanup();

    printf("Cleanup complete\n");
}
