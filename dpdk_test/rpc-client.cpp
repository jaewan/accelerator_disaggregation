#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <map>
#include "client.hpp"

std::atomic<uint64_t> reqgen{0};
std::atomic<uint64_t> keygen{0};

uint64_t rpc_reserve_clnt(const connection_state &conn) {
    struct rte_mbuf *pkt;
    int ret;

    struct rpc_header_init_request req;

    ret = create_rpc_request(&pkt, &req, sizeof(struct rpc_header_init_request), RPC_SERVER_RESERVE, reqgen++);
    if (ret < 0) {
        return -1;
    }

    uint16_t sent = rte_eth_tx_burst(0, 0, &pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send packet\n");
        rte_pktmbuf_free(pkt);
        return -1;
    }

    return keygen++; // Bad.
}

uint64_t rpc_init_clnt(const connection_state &conn) {
    struct rte_mbuf *pkt;
    int ret;

    struct rpc_header_init_request req;

    ret = create_rpc_request(&pkt, &req, sizeof(struct rpc_header_init_request), RPC_SERVER_INIT, reqgen++);
    if (ret < 0) {
        return -1;
    }

    uint16_t sent = rte_eth_tx_burst(0, 0, &pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send packet\n");
        rte_pktmbuf_free(pkt);
        return -1;
    }

    return keygen++; // Bad.
}


uint64_t rpc_operation_clnt(const connection_state &conn, uint64_t t1, uint64_t t2, OpCode op) {
    struct rte_mbuf *pkt;
    int ret;

    struct rpc_header_operation_request req;
    req.operand1 = t1;
    req.operand2 = t2;
    req.op_type = op;

    ret = create_rpc_request(&pkt, &req, sizeof(struct rpc_header_operation_request) + 100, RPC_SERVER_OPERATION, reqgen++);
    if (ret < 0) {
        return -1;
    }

    uint16_t sent = rte_eth_tx_burst(0, 0, &pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send packet\n");
        rte_pktmbuf_free(pkt);
        return -1;
    }

    return keygen++; // Bad.
}


void* rpc_materialize_clnt(const connection_state &conn, uint64_t key) {
    struct rte_mbuf *pkt;
    int ret;

    struct rpc_header_materialize_request req;
    req.tensor_key = key;

    ret = create_rpc_request(&pkt, &req, sizeof(struct rpc_header_materialize_request) + 100, 1, reqgen++);
    if (ret < 0) {
        return NULL;
    }

    uint16_t sent = rte_eth_tx_burst(0, 0, &pkt, 1);
    if (sent != 1) {
        RTE_LOG(WARNING, USER1, "Failed to send packet\n");
        rte_pktmbuf_free(pkt);
        return NULL;
    }

    return NULL;
}