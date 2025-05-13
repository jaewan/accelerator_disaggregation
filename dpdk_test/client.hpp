#include <unordered_map>
#include <unistd.h>
#include <map>
#include "rpc-common.hpp"

// Configuration constants
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define MAX_BURST_SIZE 32
#define MAX_PAYLOAD_SIZE 1400    // Max payload size to keep packet under MTU
#define MAX_WINDOW_SIZE 64       // Flow control window size
#define ACK_TIMEOUT_MS 1000        // Timeout for ACK in milliseconds
#define MAX_RETRIES 5            // Maximum retransmission attempts


struct Request {
    bool fulfilled;
    uint64_t result; 
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


uint64_t rpc_init_clnt(const connection_state &conn);
uint64_t rpc_operation_clnt(const connection_state &conn, uint64_t, uint64_t, OpCode);
uint64_t rpc_reserve_clnt(const connection_state &conn);
void* rpc_materialize_clnt(const connection_state &conn, uint64_t key);
int create_rpc_request(struct rte_mbuf **pkt_ptr, const void *payload,
    uint32_t payload_size, uint32_t rpc_id, uint64_t rpc_num);