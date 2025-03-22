cc_library(
    name = "dpdk",
    srcs = [],
    hdrs = glob(["**/*.h"]),
    strip_include_prefix = ".",
    includes = [
        ".",
        "@dpdk_config//:.",
        "@dpdk_arch//:.",
        "@dpdk_build//:.",
    ],
    copts = [
        "-march=native",
        "-O3",
        "-g",
        "-Wall",
        "-Wextra",
        "-D_GNU_SOURCE",
        "-D_FILE_OFFSET_BITS=64",
    ],
    linkopts = [
        "-Wl,--as-needed",
        "-lrte_node",
        "-lrte_graph",
        "-lrte_pipeline",
        "-lrte_table",
        "-lrte_pdump",
        "-lrte_port",
        "-lrte_fib",
        "-lrte_ipsec",
        "-lrte_vhost",
        "-lrte_stack",
        "-lrte_security",
        "-lrte_sched",
        "-lrte_reorder",
        "-lrte_rib",
        "-lrte_dmadev",
        "-lrte_regexdev",
        "-lrte_rawdev",
        "-lrte_power",
        "-lrte_pcapng",
        "-lrte_member",
        "-lrte_lpm",
        "-lrte_latencystats",
        "-lrte_jobstats",
        "-lrte_ip_frag",
        "-lrte_gso",
        "-lrte_gro",
        "-lrte_gpudev",
        "-lrte_eventdev",
        "-lrte_efd",
        "-lrte_distributor",
        "-lrte_cryptodev",
        "-lrte_compressdev",
        "-lrte_cfgfile",
        "-lrte_bpf",
        "-lrte_bitratestats",
        "-lrte_bbdev",
        "-lrte_acl",
        "-lrte_timer",
        "-lrte_hash",
        "-lrte_metrics",
        "-lrte_cmdline",
        "-lrte_pci",
        "-lrte_ethdev",
        "-lrte_meter",
        "-lrte_net",
        "-lrte_mbuf",
        "-lrte_mempool",
        "-lrte_rcu",
        "-lrte_ring",
        "-lrte_eal",
        "-lrte_telemetry",
        "-lrte_kvargs",
        "-lbsd",
    ],
    visibility = ["//visibility:public"],
) 
