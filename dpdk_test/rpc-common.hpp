#include <unistd.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <rte_memzone.h>

#define RPC_SERVER_MIN         0
#define RPC_SERVER_MATERIALIZE 1
#define RPC_SERVER_INIT        2
#define RPC_SERVER_RESERVE     3
#define RPC_SERVER_OPERATION   4
#define RPC_SERVER_MAX         5

typedef uint64_t TensorKey;

// Custom packet header for reliability
struct packet_header {
    uint32_t seq_num;        // Sequence number
    uint32_t ack_num;        // Acknowledgement number 
    uint16_t flags;          // Control flags
    uint16_t window;         // Flow control window
    uint32_t payload_size;   // Size of payload in bytes
    uint32_t total_size;     // Total size of data being transferred
    uint32_t offset;         // Offset of this chunk in the total data
    uint32_t rpc_id;
    uint64_t rpc_num;
};

struct rpc_header_materialize_request {
    TensorKey tensor_key;
} __attribute__((packed));

// struct rpc_header_materialize_reply {
//     
// } __attribute__((packed));

enum class InitType : uint8_t {
    CONSTANT,
    ZERO,
    IDENTITY,
    RANDOM,   // With different distributions later.
    DATA
};

struct rpc_header_init_request {
    InitType type;
    uint64_t shape[4];
    uint64_t length;
} __attribute__((packed));

struct rpc_header_init_reply {
    TensorKey tensor_key;
    uint8_t error;
} __attribute__((packed));


// Empty request.
struct rpc_header_reserve_request {
} __attribute__((packed));

struct rpc_header_reserve_reply {
    TensorKey tensor_key;
} __attribute__((packed));


enum class OpCode : uint8_t {
    Add    = 0,
    Sub    = 1,
    Mul    = 2,
    Div    = 3,
    Matmul = 4,
    Relu   = 5
};

struct rpc_header_operation_request {
    TensorKey operand1;
    TensorKey operand2;
    OpCode op_type;
} __attribute__((packed));

struct rpc_header_operation_reply {
    uint8_t error;
} __attribute__((packed));