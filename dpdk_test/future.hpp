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
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <unistd.h>

static size_t all_header_size =
    sizeof(struct rte_ether_hdr)
  + sizeof(struct rte_ipv4_hdr)
  + sizeof(struct rte_udp_hdr)
  + sizeof(struct packet_header);

  
static void* req_from_pkt(struct rte_mbuf* pkt) {
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    return ((void *) eth_hdr) + all_header_size;
}

static void* data_from_pkt(struct rte_mbuf* pkt) {
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    return ((void *) eth_hdr) + all_header_size;
}


typedef uint64_t TensorKey;

class FutureResult : public std::enable_shared_from_this<FutureResult> {
public:
    using Ptr = std::shared_ptr<FutureResult>;
    using ComputeFunc = std::function<void (std::vector<void*>&, float*, size_t)>;

    FutureResult(TensorKey tensor_key, struct rte_mbuf* pkt)
        : tensor_key_(tensor_key), original_packet_(std::move(pkt)), cached_(false) {}

    FutureResult(TensorKey tensor_key, std::vector<Ptr> parents, ComputeFunc compute_func)
        : tensor_key_(tensor_key),
          parents_(std::move(parents)),
          compute_func_(std::move(compute_func)),
          cached_(false) {}

    const TensorKey tensor_key() const { return tensor_key_; }

    // Evaluate the tensor result (with lazy computation)
    void* evaluate() {
        if (cached_) return data_;

        if (parents_.empty()) {
            data_ = data_from_pkt(original_packet_);
            cached_ = true;
            return data_;
        }

        std::vector<void*> parent_data;
        for (const auto& parent : parents_) {
            parent_data.push_back(parent->evaluate());
        }

        compute_func_(parent_data, (float*) data_, length_);
        cached_ = true;
        return data_;
    }

    const uint64_t length() const { return length_; }

private:
    TensorKey tensor_key_;
    std::vector<Ptr> parents_;
    ComputeFunc compute_func_;
    struct rte_mbuf* original_packet_;
    void* data_;
    bool cached_ = false;
    uint64_t shape_[4];
    uint64_t length_;
};