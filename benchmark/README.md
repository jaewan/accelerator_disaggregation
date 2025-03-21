For DPDK: igb_uio.ko never actually compiles correctly for some reason.

Clone this repo: https://github.com/F-Stack/f-stack/tree/master
Then follow the instructions to generate igb_uio.ko.

You can compile
bazel build //:benchmark_server
bazel build //:benchmark_client

And run the client from whereever with 
./bazel-bin/benchmark_client <ip address>