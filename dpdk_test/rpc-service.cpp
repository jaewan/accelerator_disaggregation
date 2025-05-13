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
#include "rpc-common.hpp"
#include "future.hpp"
#include "server-util.hpp"
#include "server_cuda_kernel.hpp"

#define DATA_SIZE 8100

std::unordered_map<TensorKey, FutureResult> tensor_store;

std::atomic<uint64_t> keygen{0};

//void* req_from_pkt(struct rte_mbuf* pkt) {
//    return ((void *) pkt) + all_header_size;
//}
//
//void* data_from_pkt(struct rte_mbuf* pkt) {
//    return ((void *) pkt) + all_header_size;
//}

void rpc_init_svc(struct rte_mbuf* pkt, const connection_state &conn) {
    void* req_ = req_from_pkt(pkt);
    struct rpc_header_init_request* req = (struct rpc_header_init_request *) req_;
    TensorKey new_key = keygen++;

    tensor_store.emplace(new_key, FutureResult(new_key, pkt));
    printf("Registered tensor id: %d\n", new_key);

    struct rpc_header_init_reply reply;
    reply.error = 0;
    send_data(conn, &reply, sizeof(struct rpc_header_init_reply));
}


void rpc_reserve_svc(struct rte_mbuf* pkt, const connection_state &conn) {
    TensorKey new_key = keygen++;

    struct rpc_header_reserve_reply reply;
    reply.tensor_key = new_key;
    send_data(conn, &reply, sizeof(struct rpc_header_reserve_reply));
}


void rpc_operation_svc(struct rte_mbuf* pkt, const connection_state &conn) {
    void* req_ = req_from_pkt(pkt);
    struct rpc_header_operation_request* req = (struct rpc_header_operation_request *) req_;

    bool second_operand_present = true;

    FutureResult::ComputeFunc compute_func;

    switch (req->op_type) {
        case OpCode::Add:
            compute_func = kernel_add_wrapper;
            break;
        case OpCode::Relu:
            compute_func = kernel_relu_wrapper;
            second_operand_present = false;
            break;
        case OpCode::Matmul: {
            compute_func = kernel_matmul_wrapper;
            break;
        }
        default:
            throw std::runtime_error("Unimplemented operation");
    }

    std::vector<FutureResult::Ptr> parents;

    auto it = tensor_store.find(req->operand1);
    printf("Operand1: %d\n", req->operand1);
    if (it == tensor_store.end()) {
        std::cerr << "Missing operand 1 tensor: " << req->operand1 << std::endl;
        return;
    }
    parents.push_back(std::make_shared<FutureResult>(it->second));

    if (second_operand_present) {
        auto it = tensor_store.find(req->operand2);
        if (it == tensor_store.end()) {
            std::cerr << "Missing operand 2 tensor: " << req->operand2 << std::endl;
            return;
        }
        parents.push_back(std::make_shared<FutureResult>(it->second));
        printf("Operand2: %d\n", req->operand2);
    }

    TensorKey result_key = keygen++;

    FutureResult res(result_key, std::move(parents), compute_func);
    tensor_store.emplace(result_key, std::move(res));

    struct rpc_header_operation_reply reply;
    reply.error = 0;
    send_data(conn, &reply, sizeof(struct rpc_header_operation_reply));
}


void rpc_materialize_svc(struct rte_mbuf* pkt, const connection_state &conn) {
    void* req_ = req_from_pkt(pkt);
    struct rpc_header_materialize_request* req = (struct rpc_header_materialize_request *) req_;

    auto it = tensor_store.find(req->tensor_key);
    if (it == tensor_store.end()) {
        std::cerr << "Invalid Tensor Key: " << req->tensor_key << std::endl;
        return;
    }

    FutureResult& res = it->second;

    void* data = res.evaluate();

    send_data(conn, data, 1000);
}
