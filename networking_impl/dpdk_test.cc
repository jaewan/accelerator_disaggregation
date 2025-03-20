#include <iostream>
#include <rte_eal.h>
#include <rte_mempool.h>
#include <rte_errno.h>  // Add this include for rte_errno
#include <string.h>

int main(int argc, char** argv) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        std::cerr << "Error: EAL initialization failed: " << rte_strerror(rte_errno) << std::endl;
        return -1;
    }

    std::cout << "DPDK EAL initialized successfully." << std::endl;

    // Create a mempool (fixed arguments)
    struct rte_mempool *mp = rte_mempool_create(
        "test_mempool",  // name
        1024,            // n (elements)
        2048,            // element size
        128,             // cache size
        0,               // private data size
        NULL,            // mp_init
        NULL,            // mp_init_arg
        NULL,            // obj_init
        NULL,            // obj_init_arg
        SOCKET_ID_ANY,   // socket id
        0                // flags
    );

    if (mp == NULL) {
        std::cerr << "Error: Mempool creation failed: " << rte_strerror(rte_errno) << std::endl;
        rte_eal_cleanup();
        return -1;
    }

    std::cout << "Mempool created successfully." << std::endl;

    rte_mempool_free(mp);
    rte_eal_cleanup();

    return 0;
}
