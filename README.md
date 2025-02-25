# AI Accelerator Disaggregation Framework

## Setup

1.  **Run the bootstrap script:**

    ```bash
    ./bootstrap.sh
    ```

    This will install Bazel and download libtorch if they are not already installed on your system.

	Before building the project, ensure the following system packages are installed:

	**Debian/Ubuntu**:
	  ```bash
	  sudo apt-get update
	  sudo apt-get install python3.10-dev
	  ```

	**Fedora**:
	  ```bash
	  sudo dnf install python3.10-devel
	  ```
	**MacOS**:
	  ```bash
	  brew install python@3.10
	  ```
3.  **Build the project:**

    ```bash
    bazel build //...
    ```

4.  **Run the example:**
    ```bash
    bazel run //:basic_example:wq
    bazel run //:remote_server
    ```
