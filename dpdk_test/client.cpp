// client.cpp - DPDK Client with reliable communication layer
// Compile with: g++ -o client client.cpp $(pkg-config --cflags --libs libdpdk)

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <rte_timer.h>
#include <rte_lcore.h>
#include <rte_cycles.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_ether.h>
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
#include <chrono>
#include <queue>
#include <algorithm>

// Configuration constants
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define MAX_BURST_SIZE 32
#define MAX_PAYLOAD_SIZE 1400    // Max payload size to keep packet under MTU
#define MAX_WINDOW_SIZE 64       // Flow control window size
#define ACK_TIMEOUT_MS 10        // Timeout for ACK in milliseconds
#define MAX_RETRIES 5            // Maximum retransmission attempts

// Default port settings
#define DEFAULT_CLIENT_PORT 12345
#define DEFAULT_SERVER_PORT 12346
#define DEFAULT_CLIENT_IP "192.168.1.10"
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

// Packet tracking structure
struct packet_info {
    uint32_t seq_num;                // Sequence number
    std::chrono::steady_clock::time_point send_time;  // Time when packet was sent
    uint8_t retries;                 // Number of retransmission attempts
    struct rte_mbuf *pkt;            // Packet mbuf
    uint32_t payload_size;           // Size of packet payload

    packet_info(uint32_t seq, struct rte_mbuf *p, uint32_t size)
        : seq_num(seq), send_time(std::chrono::steady_clock::now()),
          retries(0), pkt(p), payload_size(size) {}
};

// Connection state
struct connection_state {
    uint32_t next_seq_num;           // Next sequence number to use
    uint32_t last_ack_received;      // Last acknowledgement received
    uint16_t remote_window;          // Remote flow control window
    std::map<uint32_t, packet_info> unacked_pkts;  // Unacknowledged packets
    std::mutex mutex;                // Mutex for thread safety
    bool connected;                  // Connection established
    bool fin_sent;                   // FIN sent

    connection_state() : next_seq_num(0), last_ack_received(0),
                       remote_window(MAX_WINDOW_SIZE), connected(false),
                       fin_sent(false) {}
};

// Global variables
static volatile bool force_quit = false;
static uint16_t port_id = 0;
static struct rte_mempool *mbuf_pool = NULL;
static struct rte_ether_addr server_mac;  // Server MAC address
static uint32_t client_ip;       // Client IP address (network byte order)
static uint32_t server_ip;       // Server IP address (network byte order)
static uint16_t client_port;     // Client UDP port (network byte order)
static uint16_t server_port;     // Server UDP port (network byte order)
static connection_state conn;
static std::thread timer_thread;

// Function prototypes
static void signal_handler(int signum);
static int init_eal(int argc, char **argv);
static int init_port(uint16_t port_id, struct rte_mempool *mbuf_pool);
static int establish_connection();
static int send_data(const void *data, size_t len);
static int create_packet(struct rte_mbuf **pkt_ptr, uint32_t seq_num, uint16_t flags,
                         const void *payload, uint32_t payload_size, uint32_t total_size, uint32_t offset);
static void process_received_packet(struct rte_mbuf *pkt);
static void process_acks();
static void retransmission_timer();
static void close_connection();
static void cleanup();

// Main function
int main(int argc, char **argv) {
    int ret;

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (argc < 3) {
        printf("Usage: %s <server_mac> <data_file> [server_ip] [server_port] [client_ip] [client_port]\n", argv[0]);
        return -1;
    }

    // Parse server MAC address
    if (rte_ether_unformat_addr(argv[1], &server_mac) < 0) {
        printf("Invalid server MAC address\n");
        return -1;
    }

    // Set IP addresses and ports (default or from command line)
    if (argc >= 4) {
        server_ip = inet_addr(argv[3]);
    } else {
        server_ip = inet_addr(DEFAULT_SERVER_IP);
    }

    if (argc >= 5) {
        server_port = htons(atoi(argv[4]));
    } else {
        server_port = htons(DEFAULT_SERVER_PORT);
    }

    if (argc >= 6) {
        client_ip = inet_addr(argv[5]);
    } else {
        client_ip = inet_addr(DEFAULT_CLIENT_IP);
    }

    if (argc >= 7) {
        client_port = htons(atoi(argv[6]));
    } else {
        client_port = htons(DEFAULT_CLIENT_PORT);
    }

    // Initialize EAL
    ret = init_eal(argc, argv);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error initializing EAL\n");
    }

    // Create mempool for mbufs
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id());
    if (mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");
    }

    // Initialize port
    ret = init_port(port_id, mbuf_pool);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error initializing port %d\n", port_id);
    }

    printf("Client initialized with:\n");
    printf("  Server MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
           server_mac.addr_bytes[0], server_mac.addr_bytes[1],
           server_mac.addr_bytes[2], server_mac.addr_bytes[3],
           server_mac.addr_bytes[4], server_mac.addr_bytes[5]);
    printf("  Server IP: %s, Port: %d\n",
           inet_ntoa(*(struct in_addr *)&server_ip), ntohs(server_port));
    printf("  Client IP: %s, Port: %d\n",
           inet_ntoa(*(struct in_addr *)&client_ip), ntohs(client_port));
    printf("Connecting to server...\n");

    // Start timer thread for handling retransmissions
    timer_thread = std::thread(retransmission_timer);

    // Establish connection with server
    ret = establish_connection();
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Failed to establish connection\n");
    }

    printf("Connected to server. Sending data...\n");

    // Read data from file
    FILE *fp = fopen(argv[2], "rb");
    if (fp == NULL) {
        printf("Failed to open data file: %s\n", argv[2]);
        close_connection();
        cleanup();
        return -1;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Allocate buffer for file data
    uint8_t *file_data = (uint8_t *)malloc(file_size);
    if (file_data == NULL) {
        printf("Failed to allocate memory for file data\n");
        fclose(fp);
        close_connection();
        cleanup();
        return -1;
    }

    // Read file data
    size_t bytes_read = fread(file_data, 1, file_size, fp);
    fclose(fp);

    if (bytes_read != file_size) {
        printf("Failed to read file data\n");
        free(file_data);
        close_connection();
        cleanup();
        return -1;
    }

    // Send file data
    printf("Sending %zu bytes of data...\n", file_size);
    ret = send_data(file_data, file_size);
    free(file_data);

    if (ret < 0) {
        printf("Failed to send data\n");
        close_connection();
        cleanup();
        return -1;
    }

    printf("Data sent successfully. Closing connection...\n");

    // Close connection
    close_connection();

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

    return 0;
}

// Establish connection with server
static int establish_connection() {
    struct rte_mbuf *syn_pkt;
    int ret;

    // Create SYN packet
    ret = create_packet(&syn_pkt, conn.next_seq_num, PKT_FLAG_SYN,
                        NULL, 0, 0, 0);
    if (ret < 0) {
        return -1;
    }

    // Send the SYN packet
    uint16_t sent = rte_eth_tx_burst(port_id, 0, &syn_pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send SYN packet\n");
        rte_pktmbuf_free(syn_pkt);
        return -1;
    }

    // Store the sent packet for potential retransmission
    {
        std::lock_guard<std::mutex> lock(conn.mutex);
        conn.unacked_pkts[conn.next_seq_num] = packet_info(conn.next_seq_num, syn_pkt, 0);
        conn.next_seq_num++;
    }

    // Wait for connection to be established (SYN-ACK received)
    int timeout_ms = 1000;  // 1 second
    auto start_time = std::chrono::steady_clock::now();

    while (!conn.connected && !force_quit) {
        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > timeout_ms) {
            printf("Connection timeout\n");
            return -1;
        }

        // Process ACKs
        process_acks();

        // Sleep a bit to avoid busy waiting
        usleep(1000);  // 1ms
    }

    if (force_quit) {
        return -1;
    }

    return 0;
}

// Create a packet with proper headers
static int create_packet(struct rte_mbuf **pkt_ptr, uint32_t seq_num, uint16_t flags,
                        const void *payload, uint32_t payload_size, uint32_t total_size, uint32_t offset) {

    struct rte_mbuf *pkt = rte_pktmbuf_alloc(mbuf_pool);
    if (pkt == NULL) {
        RTE_LOG(ERR, USER1, "Failed to allocate mbuf for packet\n");
        return -1;
    }

    // Calculate packet size
    uint16_t total_length = sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr) +
                           sizeof(struct packet_header) + payload_size;

    // 1. Set up Ethernet header
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    rte_ether_addr_copy(&server_mac, &eth_hdr->dst_addr);
    rte_eth_macaddr_get(port_id, &eth_hdr->src_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    // 2. Set up IP header
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    memset(ip_hdr, 0, sizeof(struct rte_ipv4_hdr));
    ip_hdr->version_ihl = RTE_IPV4_VHL_DEF;  // IP version 4, header length 5 (20 bytes)
    ip_hdr->total_length = rte_cpu_to_be_16(total_length);
    ip_hdr->time_to_live = 64;  // TTL
    ip_hdr->next_proto_id = IPPROTO_UDP;
    ip_hdr->src_addr = client_ip;
    ip_hdr->dst_addr = server_ip;

    // Let hardware calculate IP checksum
    pkt->ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM;

    // 3. Set up UDP header
    struct rte_udp_hdr *udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    udp_hdr->src_port = client_port;
    udp_hdr->dst_port = server_port;
    udp_hdr->dgram_len = rte_cpu_to_be_16(sizeof(struct rte_udp_hdr) +
                                         sizeof(struct packet_header) + payload_size);

    // Let hardware calculate UDP checksum
    pkt->ol_flags |= RTE_MBUF_F_TX_UDP_CKSUM;
    udp_hdr->dgram_cksum = 0;  // Set to 0 for hardware to fill in

    // 4. Set up packet header
    struct packet_header *hdr = (struct packet_header *)(udp_hdr + 1);
    memset(hdr, 0, sizeof(struct packet_header));
    hdr->seq_num = seq_num;
    hdr->flags = flags;
    hdr->window = MAX_WINDOW_SIZE;
    hdr->payload_size = payload_size;
    hdr->total_size = total_size;
    hdr->offset = offset;

    // 5. Copy payload if any
    if (payload != NULL && payload_size > 0) {
        uint8_t *payload_ptr = (uint8_t *)(hdr + 1);
        memcpy(payload_ptr, payload, payload_size);
    }

    // 6. Set the packet length
    pkt->data_len = sizeof(struct rte_ether_hdr) + total_length;
    pkt->pkt_len = pkt->data_len;

    // Return the packet
    *pkt_ptr = pkt;
    return 0;
}

// Send data to server
static int send_data(const void *data, size_t len) {
    const uint8_t *data_ptr = (const uint8_t *)data;
    size_t bytes_sent = 0;
    size_t total_size = len;

    while (bytes_sent < len && !force_quit) {
        // Process ACKs
        process_acks();

        // Calculate available window
        uint16_t available_window;
        {
            std::lock_guard<std::mutex> lock(conn.mutex);
            available_window = std::min(conn.remote_window, (uint16_t)conn.unacked_pkts.size());
            available_window = MAX_WINDOW_SIZE - available_window;
        }

        // If window is full, wait for ACKs
        if (available_window == 0) {
            usleep(1000);  // 1ms
            continue;
        }

        // Calculate number of bytes to send in this packet
        uint32_t bytes_to_send = std::min((size_t)MAX_PAYLOAD_SIZE, len - bytes_sent);

        // Create packet with data payload
        struct rte_mbuf *pkt;
        uint32_t seq_num;
        {
            std::lock_guard<std::mutex> lock(conn.mutex);
            seq_num = conn.next_seq_num;
            conn.next_seq_num += bytes_to_send;
        }

        int ret = create_packet(&pkt, seq_num, PKT_FLAG_DATA,
                               data_ptr + bytes_sent, bytes_to_send,
                               total_size, bytes_sent);
        if (ret < 0) {
            return -1;
        }

        // Send the packet
        uint16_t sent = rte_eth_tx_burst(port_id, 0, &pkt, 1);
        if (sent != 1) {
            RTE_LOG(WARNING, USER1, "Failed to send data packet\n");
            rte_pktmbuf_free(pkt);
            return -1;
        }

        // Store the sent packet for potential retransmission
        {
            std::lock_guard<std::mutex> lock(conn.mutex);
            conn.unacked_pkts[seq_num] = packet_info(seq_num, pkt, bytes_to_send);
        }

        bytes_sent += bytes_to_send;

        // Progress indicator
        if (bytes_sent % (1024 * 1024) == 0 || bytes_sent == len) {
            printf("Progress: %.2f%% (%zu/%zu bytes)\n",
                   (double)bytes_sent / len * 100, bytes_sent, len);
        }
    }

    if (force_quit) {
        return -1;
    }

    // Wait for all packets to be acknowledged
    int timeout_ms = 5000;  // 5 seconds
    auto start_time = std::chrono::steady_clock::now();

    while (!force_quit) {
        // Check if all packets are acknowledged
        {
            std::lock_guard<std::mutex> lock(conn.mutex);
            if (conn.unacked_pkts.empty()) {
                break;
            }
        }

        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > timeout_ms) {
            printf("Timeout waiting for ACKs\n");
            return -1;
        }

        // Process ACKs
        process_acks();

        // Sleep a bit to avoid busy waiting
        usleep(1000);  // 1ms
    }

    if (force_quit) {
        return -1;
    }

    return 0;
}

// Process a received packet
static void process_received_packet(struct rte_mbuf *pkt) {
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
    if (udp_hdr->dst_port != client_port) {
        // Not destined for our client port, ignore
        rte_pktmbuf_free(pkt);
        return;
    }

    // Parse packet header
    struct packet_header *hdr = (struct packet_header *)(udp_hdr + 1);

    // Process based on packet flags
    if (hdr->flags & PKT_FLAG_ACK) {
        std::lock_guard<std::mutex> lock(conn.mutex);

        // If this is a SYN-ACK, mark the connection as established
        if (hdr->flags & PKT_FLAG_SYN) {
            conn.connected = true;
        }

        // Update last ACK received
        if (hdr->ack_num > conn.last_ack_received) {
            conn.last_ack_received = hdr->ack_num;
        }

        // Update remote window
        conn.remote_window = hdr->window;

        // Remove acknowledged packets
        auto it = conn.unacked_pkts.begin();
        while (it != conn.unacked_pkts.end()) {
            if (it->first + it->second.payload_size <= conn.last_ack_received) {
                // This packet is fully acknowledged
                rte_pktmbuf_free(it->second.pkt);
                it = conn.unacked_pkts.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Free the received mbuf
    rte_pktmbuf_free(pkt);
}

// Process acknowledgements
static void process_acks() {
    struct rte_mbuf *rx_mbufs[MAX_BURST_SIZE];
    const uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, rx_mbufs, MAX_BURST_SIZE);

    if (nb_rx == 0) {
        return;
    }

    for (uint16_t i = 0; i < nb_rx; i++) {
        process_received_packet(rx_mbufs[i]);
    }
}

// Timer thread for handling retransmissions
static void retransmission_timer() {
    while (!force_quit) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ACK_TIMEOUT_MS));

        std::lock_guard<std::mutex> lock(conn.mutex);

        // Current time
        auto now = std::chrono::steady_clock::now();

        // Check for packets that need retransmission
        for (auto &pair : conn.unacked_pkts) {
            packet_info &info = pair.second;

            // Check if packet timeout has expired
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - info.send_time).count();

            if (elapsed > ACK_TIMEOUT_MS) {
                // Increment retries
                info.retries++;

                if (info.retries > MAX_RETRIES) {
                    // Too many retries, connection may be lost
                    printf("Packet seq %u exceeded maximum retries\n", info.seq_num);
                    continue;
                }

                // Update send time
                info.send_time = now;

                // Retransmit the packet
                uint16_t sent = rte_eth_tx_burst(port_id, 0, &info.pkt, 1);
                if (sent != 1) {
                    RTE_LOG(WARNING, USER1, "Failed to retransmit packet seq %u\n", info.seq_num);
                }

                // Apply exponential backoff to the timeout for this packet
                // This is a simple form of congestion control
                if (info.retries > 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(
                        (1 << (info.retries - 1)) * ACK_TIMEOUT_MS / 10));
                }
            }
        }
    }
}

// Close connection
static void close_connection() {
    if (conn.fin_sent) {
        return;
    }

    // Create FIN packet
    struct rte_mbuf *fin_pkt;
    int ret = create_packet(&fin_pkt, conn.next_seq_num, PKT_FLAG_FIN,
                          NULL, 0, 0, 0);
    if (ret < 0) {
        return;
    }

    // Send the FIN packet
    uint16_t sent = rte_eth_tx_burst(port_id, 0, &fin_pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send FIN packet\n");
        rte_pktmbuf_free(fin_pkt);
        return;
    }

    conn.fin_sent = true;

    // Wait for FIN-ACK (simplified)
    int timeout_ms = 1000;  // 1 second
    auto start_time = std::chrono::steady_clock::now();

		while (!force_quit) {
        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed > timeout_ms) {
            printf("FIN-ACK timeout\n");
            break;
        }

        // Process ACKs
        process_acks();

        // Sleep a bit to avoid busy waiting
        usleep(1000);  // 1ms
    }

    // Free the FIN packet
    rte_pktmbuf_free(fin_pkt);
}

// Cleanup all resources
static void cleanup() {
    // Stop the timer thread
    force_quit = true;
    if (timer_thread.joinable()) {
        timer_thread.join();
    }

    // Free all unacknowledged packets
    {
        std::lock_guard<std::mutex> lock(conn.mutex);
        for (auto &pair : conn.unacked_pkts) {
            rte_pktmbuf_free(pair.second.pkt);
        }
        conn.unacked_pkts.clear();
    }

    // Close the Ethernet port
    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);

    // Free memory pool
    if (mbuf_pool != NULL) {
        rte_mempool_free(mbuf_pool);
    }

    // Clean up EAL
    rte_eal_cleanup();

    printf("Cleanup complete\n");
}
