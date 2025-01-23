# AI Accelerator Disaggregation Framework

## Setup

1.  **Run the bootstrap script:**

    ```bash
    ./bootstrap.sh
    ```

    This will install Bazel and download libtorch if they are not already installed on your system.

2.  **Build the project:**

    ```bash
    bazel build //...
    ```

3.  **Run the example:**
    ```bash
    bazel run //:basic_example:wq
    bazel run //:remote_server
    ```
