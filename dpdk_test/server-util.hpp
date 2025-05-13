#include <map>

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

#define SEQ_ZERO 0
#define SEQ_ZERO_PLUS_ACK (SEQ_ZERO + 1)

// Use pinned memory for DPDK mbuf pool
#define USE_PINNED_MEMORY_POOL 1
#define MEMPOOL_DATAROOM_SIZE (RTE_MBUF_DEFAULT_BUF_SIZE)

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

    bool operator==(const connection_id &other) const {
        return client_ip == other.client_ip && client_port == other.client_port;
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
                        recv_window(MAX_WINDOW_SIZE), active(false) {
        memset(&client_mac, 0, sizeof(client_mac));
    }
};

int send_data(const connection_state &conn, const void *payload, uint32_t payload_size);

void rpc_materialize_svc(struct rte_mbuf* pkt, const connection_state &conn);
void rpc_init_svc(struct rte_mbuf* pkt, const connection_state &conn);
void rpc_operation_svc(struct rte_mbuf* pkt, const connection_state &conn);
void rpc_reserve_svc(struct rte_mbuf* pkt, const connection_state &conn);