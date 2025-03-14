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
#define CHUNK_SIZE 2048  // 2KB chunks
#define PORT_ID 0

static volatile bool force_quit;
static struct rte_mempool *mbuf_pool;
static uint64_t total_bytes_sent = 0;

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

    /* Allocate and set up 1 TX queue per Ethernet port. */
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, TX_RING_SIZE,
                rte_eth_dev_socket_id(port), NULL);
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

    printf("DPDK Client initialized successfully. Sending 2KB chunks...\n");

    while (!force_quit) {
        struct rte_mbuf *mb = rte_pktmbuf_alloc(mbuf_pool);
        if (mb == NULL) {
            printf("Failed to allocate mbuf\n");
            continue;
        }

        // Prepare the packet with 2KB of data
        char *packet_data = rte_pktmbuf_mtod(mb, char *);
        memset(packet_data, 0xAA, CHUNK_SIZE);  // Fill with pattern
        mb->data_len = CHUNK_SIZE;
        mb->pkt_len = CHUNK_SIZE;

        // Send the packet
        const uint16_t nb_tx = rte_eth_tx_burst(port, 0, &mb, 1);
        if (nb_tx == 0) {
            rte_pktmbuf_free(mb);
        } else {
            total_bytes_sent += CHUNK_SIZE;
            printf("\rTotal data sent: %lu bytes (%.2f MB)", 
                   total_bytes_sent, (float)total_bytes_sent / (1024*1024));
            fflush(stdout);
        }
        
        usleep(1000);  // Small delay to prevent overwhelming
    }

    printf("\nShutting down...\n");
    printf("Final statistics:\n");
    printf("Total data sent: %lu bytes (%.2f MB)\n", 
           total_bytes_sent, (float)total_bytes_sent / (1024*1024));

    /* Clean up */
    rte_eth_dev_stop(port);
    rte_eth_dev_close(port);
    rte_eal_cleanup();

    return 0;
} 
