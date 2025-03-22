#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <signal.h>
#include <stdbool.h>

#define RX_RING_SIZE 128
#define TX_RING_SIZE 512
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

static volatile bool force_quit;

/* Signal handler for graceful shutdown */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\n\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

/* Initialize a port with RX and TX queues */
static int port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
    struct rte_eth_conf port_conf = {
        .rxmode = {
            .max_lro_pkt_size = RTE_ETHER_MAX_LEN,
        },
        .txmode = {
            .mq_mode = RTE_ETH_MQ_TX_NONE,
        },
    };

    /* Configure the Ethernet device */
    int ret = rte_eth_dev_configure(port, 1, 1, &port_conf);
    if (ret != 0) {
        printf("Error configuring port %u\n", port);
        return ret;
    }

    /* Set up RX queue */
    ret = rte_eth_rx_queue_setup(port, 0, RX_RING_SIZE, 0, NULL, mbuf_pool);
    if (ret < 0) {
        printf("Error setting up RX queue\n");
        return ret;
    }

    /* Set up TX queue */
    ret = rte_eth_tx_queue_setup(port, 0, TX_RING_SIZE, 0, NULL);
    if (ret < 0) {
        printf("Error setting up TX queue\n");
        return ret;
    }

    /* Start the device */
    ret = rte_eth_dev_start(port);
    if (ret < 0) {
        printf("Error starting device\n");
        return ret;
    }

    /* Enable promiscuous mode */
    rte_eth_promiscuous_enable(port);

    return 0;
}

int main(int argc, char *argv[])
{
    struct rte_mempool *mbuf_pool;
    uint16_t port_id = 0;
    int ret;

    /* Initialize EAL */
    ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        printf("Error initializing EAL\n");
        return -1;
    }

    /* Check if there is at least one port available */
    if (rte_eth_dev_count_avail() == 0) {
        printf("No Ethernet ports available\n");
        return -1;
    }

    /* Create a mempool for packet buffers */
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
                                       NUM_MBUFS,
                                       MBUF_CACHE_SIZE,
                                       0,
                                       RTE_MBUF_DEFAULT_BUF_SIZE,
                                       rte_socket_id());
    if (mbuf_pool == NULL) {
        printf("Cannot create mbuf pool\n");
        return -1;
    }

    /* Initialize the port */
    if (port_init(port_id, mbuf_pool) != 0) {
        printf("Cannot initialize port %u\n", port_id);
        return -1;
    }

    /* Set up signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("Starting server on port %u...\n", port_id);

    /* Main packet processing loop */
    while (!force_quit) {
        struct rte_mbuf *pkts[BURST_SIZE];
        uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, pkts, BURST_SIZE);

        if (nb_rx > 0) {
            /* Echo back received packets */
            uint16_t nb_tx = rte_eth_tx_burst(port_id, 0, pkts, nb_rx);
            
            /* Free any unsent packets */
            if (nb_tx < nb_rx) {
                uint16_t buf;
                for (buf = nb_tx; buf < nb_rx; buf++)
                    rte_pktmbuf_free(pkts[buf]);
            }
        }
    }

    /* Cleanup */
    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
    rte_eal_cleanup();

    return 0;
} 