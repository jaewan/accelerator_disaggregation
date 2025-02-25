# Define the workspace
workspace(name = "accelerator_disaggregation")

# Load Python and Pybind11 rules
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Add bazel_skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
    ],
)

# Load rules_python
http_archive(
    name = "rules_python",
    sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
    strip_prefix = "rules_python-0.26.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.26.0/rules_python-0.26.0.tar.gz",
)

# Initialize rules_python
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

# Register Python toolchain
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)

# Load pip dependencies if needed
load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "python_deps",
    requirements_lock = "//:requirements.txt",  # Create an empty file if you don't have requirements
)
load("@python_deps//:requirements.bzl", "install_deps")
install_deps()

# Load pybind11_bazel
http_archive(
    name = "pybind11_bazel",
    sha256 = "a185aa68c93b9f62c80fcb3aadc3c83c763854750dc3f38be1dadcb7be223837",  # Updated SHA-256
    strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
    urls = [
        "https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip",
    ],
)

# Load pybind11
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c",
    strip_prefix = "pybind11-2.11.1",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz",
    ],
)

# Configure Python for pybind11
load("@python3_10//:defs.bzl", "interpreter")
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(
    name = "local_config_python",
    python_interpreter_target = interpreter,
)

# Load the libtorch_repository rule
load("//:libtorch_repository.bzl", "libtorch_repository")

# Define the libtorch repository
libtorch_repository(
    name = "libtorch",
    pytorch_version = "2.5.1",  # Specify the desired PyTorch version
    cuda_tag = "cu121",         # Specify the CUDA tag (e.g., "cu121" for CUDA 12.1, "cpu" for CPU-only)
    sha256 = "",  # Optional: Add SHA256 for verification
)
