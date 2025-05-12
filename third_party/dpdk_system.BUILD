cc_library(
    name = "dpdk_lib",
    srcs = [],
    hdrs = glob(["include/dpdk/**/*.h"]),
    includes = ["include/dpdk"],
    linkopts = [
        "-L/usr/local/lib/x86_64-linux-gnu",
        "-lrte_eal",
        "-lrte_mempool",
        "-lrte_mbuf",
        "-lnuma",
        "-lpthread",
        "-ldl",
        "-lm",
    ],
    visibility = ["//visibility:public"],
)
