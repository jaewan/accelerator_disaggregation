#include <unistd.h> 
#include <iostream>
#include <cstring>    
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_memzone.h>
// #include <rte_extmem.h> Compiler bug. Wtf is happening with DPDK?
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <unistd.h> 
#include "common.h"
#include "server-runner.h"

// DPDK configuration
//#define RX_RING_SIZE 128
//#define TX_RING_SIZE 512
//#define NUM_MBUFS 8191
//#define MBUF_CACHE_SIZE 250
//#define BURST_SIZE 32

// CUDA configuration
 // 1 MB of data
//#define MBUF_DATA_SIZE RTE_MBUF_DEFAULT_BUF_SIZE


/*
#define RTE_CHECK(call) \
    do { \
        int ret = call; \
        if (ret != 0) { \
            std::cerr << "DPDK error: " << rte_strerror(-ret) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
*/

int main(int argc, char *argv[]) {

    bool quiet = false; // I forgot what this was.

    std::cout << "Initialization START" << std::endl;
    
    /* ============= TODO ===============
       1. Change the code below to be compile-time configurable depending on the protocol
       2. Come up with a better message-passing interface */
       
    int server_fd;
    struct sockaddr_in server_addr;
    int client_fd;
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);


    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    if (((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        || (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1)
        || (listen(server_fd, 5) == -1)) return 1;
     //char command;

     std::cout << "Initialization END" << std::endl;

    int connection_id = 1;

    while (1) {
        client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd == -1)   return 1;

        std::cout << "Connection BEGIN: " << connection_id << std::endl;
        if (run_server(client_fd)) std::cout << "Somthing wrong" << std::endl;
        std::cout << "Connection END" << connection_id << std::endl;

        close(client_fd);
        connection_id++;
    }

    return 0;
}