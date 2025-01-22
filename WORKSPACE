# WORKSPACE
workspace(name = "accelerator_disaggregation") #TODO change this once this project is named

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Setup LibTorch (replace with your desired version and CUDA support)
http_archive(
    name = "libtorch",
    sha256 = "YOUR_LIBTORCH_SHA256",  # Replace with the actual SHA256
    strip_prefix = "libtorch-YOUR_VERSION",
    urls = ["https://download.pytorch.org/libtorch/YOUR_VERSION/libtorch-YOUR_VERSION.zip"],
)

# Setup Bazel (if needed)
# Example for setting up a specific Bazel version:
# http_archive(
#     name = "bazel",
#     sha256 = "YOUR_BAZEL_SHA256",
#     url = "https://github.com/bazelbuild/bazel/releases/download/YOUR_BAZEL_VERSION/bazel-YOUR_BAZEL_VERSION-installer-linux-x86_64.sh",
# )

# Load PyTorch dependency
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Declare LibTorch as an external dependency
local_repository(
    name = "libtorch",
    path = "third_party",
)
# Or if you have a local installation
#local_repository(
#    name = "pytorch",
#    path = "/path/to/your/pytorch/installation", # <--- REPLACE WITH YOUR PYTORCH PATH
#)

# Pybind11 (adjust version if needed)
#http_archive(
#    name = "pybind11_bazel",
#    url = "https://github.com/pybind/pybind11/archive/v2.10.4.zip", # Or a specific version
#    sha256 = "4dd9450d1a8b4d969a73573105642d6d1dff674179f19712a77732ff287103c4", # Verify checksum for security
#    strip_prefix = "pybind11-2.10.4", # Adjust if different version
#)
