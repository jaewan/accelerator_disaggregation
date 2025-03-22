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

/* Create a test packet */
static struct rte_mbuf* create_packet(struct rte_mempool *mbuf_pool)
{
    struct rte_mbuf *pkt = rte_pktmbuf_alloc(mbuf_pool);
    if (pkt == NULL) {
        printf("Error allocating packet buffer\n");
        return NULL;
    }

    /* Initialize Ethernet header */
    char *data = rte_pktmbuf_append(pkt, 64); /* minimum Ethernet frame size */
    if (data == NULL) {
        rte_pktmbuf_free(pkt);
        return NULL;
    }

    /* Fill with test data */
    memset(data, 0xFF, 6);  /* Destination MAC (broadcast) */
    memset(data + 6, 0x00, 6);  /* Source MAC */
    data[12] = 0x08;  /* Ethertype (IPv4) */
    data[13] = 0x00;
    memset(data + 14, 0xAB, 50);  /* Payload */

    return pkt;
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

    printf("Starting client on port %u...\n", port_id);

    /* Main packet processing loop */
    while (!force_quit) {
        /* Create and send a test packet */
        struct rte_mbuf *pkt = create_packet(mbuf_pool);
        if (pkt != NULL) {
            uint16_t nb_tx = rte_eth_tx_burst(port_id, 0, &pkt, 1);
            if (nb_tx < 1) {
                rte_pktmbuf_free(pkt);
            }
        }

        /* Wait for response */
        struct rte_mbuf *rx_pkts[BURST_SIZE];
        uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, rx_pkts, BURST_SIZE);
        
        if (nb_rx > 0) {
            printf("Received %u packets\n", nb_rx);
            uint16_t buf;
            for (buf = 0; buf < nb_rx; buf++) {
                rte_pktmbuf_free(rx_pkts[buf]);
            }
        }

        /* Add a small delay to avoid flooding */
        usleep(1000);  /* 1ms delay */
    }

    /* Cleanup */
    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
    rte_eal_cleanup();

    return 0;
} 