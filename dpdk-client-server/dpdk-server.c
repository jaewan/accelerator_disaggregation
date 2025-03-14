#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>
#include <signal.h>
#include <unistd.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32
#define PORT_ID 0

static volatile bool force_quit;
static struct rte_mempool *mbuf_pool;
static uint64_t total_bytes_received = 0;
static uint64_t total_packets_received = 0;

static const struct rte_eth_conf port_conf_default = {
    .rxmode = {
        .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
    },
};

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

static int port_init(uint16_t port) {
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = 1, tx_rings = 1;
    int retval;
    uint16_t q;

    if (!rte_eth_dev_is_valid_port(port))
        return -1;

    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0)
        return retval;

    /* Allocate and set up RX queue */
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, RX_RING_SIZE,
                rte_eth_dev_socket_id(port), NULL, mbuf_pool);
        if (retval < 0)
            return retval;
    }

    /* Start the Ethernet port. */
    retval = rte_eth_dev_start(port);
    if (retval < 0)
        return retval;

    return 0;
}

int main(int argc, char *argv[])
{
    int ret;
    uint16_t port = PORT_ID;
    struct rte_mbuf *rx_mbufs[BURST_SIZE];
    
    /* Initialize EAL */
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

    /* Create the mbuf pool */
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    
    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    /* Initialize port */
    if (port_init(port) != 0)
        rte_exit(EXIT_FAILURE, "Cannot init port %"PRIu16 "\n", port);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("DPDK Server initialized successfully. Waiting for data...\n");

    uint64_t prev_total = 0;
    uint64_t start_time = rte_get_timer_cycles();
    const uint64_t timer_hz = rte_get_timer_hz();

    while (!force_quit) {
        // Receive burst of packets
        const uint16_t nb_rx = rte_eth_rx_burst(port, 0, rx_mbufs, BURST_SIZE);
        
        if (nb_rx > 0) {
            // Process received packets
            for (uint16_t i = 0; i < nb_rx; i++) {
                total_bytes_received += rx_mbufs[i]->pkt_len;
                rte_pktmbuf_free(rx_mbufs[i]);
            }
            total_packets_received += nb_rx;

            // Calculate throughput every second
            uint64_t current_time = rte_get_timer_cycles();
            double time_diff = (double)(current_time - start_time) / timer_hz;
            
            if (time_diff >= 1.0) {
                double throughput = (total_bytes_received - prev_total) / (1024.0 * 1024.0); // MB
                printf("\rReceived: %lu bytes (%.2f MB) | Packets: %lu | Throughput: %.2f MB/s", 
                       total_bytes_received, 
                       (float)total_bytes_received / (1024*1024),
                       total_packets_received,
                       throughput);
                fflush(stdout);
                
                prev_total = total_bytes_received;
                start_time = current_time;
            }
        }
    }

    printf("\nShutting down...\n");
    printf("Final statistics:\n");
    printf("Total data received: %lu bytes (%.2f MB)\n", 
           total_bytes_received, (float)total_bytes_received / (1024*1024));
    printf("Total packets received: %lu\n", total_packets_received);
    printf("Average packet size: %.2f bytes\n", 
           total_packets_received > 0 ? (float)total_bytes_received / total_packets_received : 0);

    /* Clean up */
    rte_eth_dev_stop(port);
    rte_eth_dev_close(port);
    rte_eal_cleanup();

    return 0;
} 