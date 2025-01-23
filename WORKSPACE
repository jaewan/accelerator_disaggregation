# WORKSPACE
workspace(name = "accelerator_disaggregation") #TODO change this once this project is named

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


# Setup Bazel (if needed)
# Example for setting up a specific Bazel version:
# http_archive(
#     name = "bazel",
#     sha256 = "YOUR_BAZEL_SHA256",
#     url = "https://github.com/bazelbuild/bazel/releases/download/YOUR_BAZEL_VERSION/bazel-YOUR_BAZEL_VERSION-installer-linux-x86_64.sh",
# )

 # Set the LIBTORCH_URL and related variables.
LIBTORCH_OS= "linux"
LIBTORCH_CUDA_VERSION="cpu" # or "cu118" for CUDA, determine based on system check
if native.os_name() == "darwin":
	LIBTORCH_OS = "macos"

# check if nvidia-smi can be run on this machine, which will implies CUDA
CUDA_CHECK_CMD = ["which", "nvidia-smi"]
if native.execute(CUDA_CHECK_CMD).return_code == 0:
	LIBTORCH_CUDA_VERSION="cu118" # Update this as necessary


LIBTORCH_URL = "https://download.pytorch.org/libtorch/cpu/%s/libtorch-%s-cpu-latest.zip" % (LIBTORCH_OS, LIBTORCH_OS)

if LIBTORCH_CUDA_VERSION != "cpu":
	LIBTORCH_URL = "https://download.pytorch.org/libtorch/%s/%s/libtorch-%s-%s-latest.zip" % (LIBTORCH_CUDA_VERSION, LIBTORCH_OS, LIBTORCH_OS, LIBTORCH_CUDA_VERSION)

# Python rules
http_archive(
    name = "rules_python",
    sha256 = "9acc0944c94f4f48997858bf324551b629dc6bb150f30382ee29f66f7f28e02c",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.9.0/rules_python-0.9.0.tar.gz",
)

# Or if you have a local installation
#local_repository(
#    name = "pytorch",
#    path = "/path/to/your/pytorch/installation", # <--- REPLACE WITH YOUR PYTORCH PATH
#)

# Pybind11 (adjust version if needed)
http_archive(
    name = "pybind11_bazel",
    url = "https://github.com/pybind/pybind11/archive/v2.10.4.zip", # Or a specific version
    sha256 = "4dd9450d1a8b4d969a73573105642d6d1dff674179f19712a77732ff287103c4", # Verify checksum for security
    strip_prefix = "pybind11-2.10.4", # Adjust if different version
)

# Libtorch - you'll need to provide the appropriate URL and sha256 for your platform
http_archive(
	name = "libtorch",
	build_file = "//third_party:libtorch.BUILD",
	urls = [LIBTORCH_URL],
	strip_prefix = "libtorch",
	sha256 = "77023279783949578550a8d4b4c91093530f78dd6c755ff5992149a80c88b6be" # this sha needs to be updated based on the actual URL
)
