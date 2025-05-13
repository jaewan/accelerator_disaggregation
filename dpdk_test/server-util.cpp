#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <rte_memzone.h>
#include <rte_gpudev.h>
//#include <rte_gpu_comm.h>
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
#include <rte_arp.h>
#include <rte_byteorder.h>
#include "server-util.hpp"
#include "rpc-handlers.hpp"

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
        RTE_LOG(ERR, USER1, "Promiscuous mode not enabled\n");
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

    if (eth_hdr->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_ARP)) {
        struct rte_arp_hdr *arp_hdr = (struct rte_arp_hdr *)(eth_hdr + 1);

        uint32_t source_ip = arp_hdr->arp_data.arp_sip;
        uint32_t target_ip = arp_hdr->arp_data.arp_tip;

        printf("%d\n", target_ip);

        if (arp_hdr->arp_opcode == rte_cpu_to_be_16(RTE_ARP_OP_REQUEST) &&
            target_ip == 134254346) {
            printf("Got an arp request here %d \n", arp_hdr->arp_data.arp_tip);

            // Extra nice efficiency by just modifying this packet
            struct rte_ether_addr sender_mac = eth_hdr->src_addr;

            // Set Ethernet header
            rte_ether_addr_copy(&sender_mac, &eth_hdr->dst_addr);
            rte_ether_addr_copy(&server_mac, &eth_hdr->src_addr);
            eth_hdr->ether_type = RTE_ETHER_TYPE_ARP;

        

            // Set ARP reply
            arp_hdr->arp_opcode = RTE_ARP_OP_REPLY;
            arp_hdr->arp_data.arp_sip = target_ip;
            arp_hdr->arp_data.arp_tip = source_ip;
            rte_ether_addr_copy(&server_mac, &arp_hdr->arp_data.arp_sha);
            rte_ether_addr_copy(&sender_mac, &arp_hdr->arp_data.arp_tha);

            // Send it back
            uint16_t nb_tx = rte_eth_tx_burst(port_id, 0, &pkt, 1);
            if (nb_tx < 1)
                rte_pktmbuf_free(pkt);
            return;
        }
    } else if (eth_hdr->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
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

    //printf("Got an IP Packet with IP %d and port %d\n", ip_hdr->src_addr, udp_hdr->src_port);
    
    
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