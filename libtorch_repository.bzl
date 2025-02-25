def _libtorch_repository_rule_impl(repository_ctx):
    """Implementation of the libtorch_repository rule.

    Downloads and extracts Libtorch based on the specified PyTorch version and CUDA tag.
    Generates a BUILD file for the Libtorch library.
    """

    # Get attributes from the repository rule
    pytorch_version = repository_ctx.attr.pytorch_version
    cuda_tag = repository_ctx.attr.cuda_tag
    sha256 = repository_ctx.attr.sha256

    # Validate PyTorch version format (X.Y.Z)
    if not _is_valid_pytorch_version(pytorch_version):
        fail("Invalid PyTorch version format: {}. Must be X.Y.Z (e.g., 2.5.1)".format(pytorch_version))

    # Validate CUDA tag
    valid_cuda_tags = ["cpu", "cu116", "cu117", "cu118", "cu121"]
    if cuda_tag not in valid_cuda_tags:
        fail("Invalid CUDA tag: {}. Must be one of: {}".format(cuda_tag, valid_cuda_tags))

    # Construct Libtorch URL based on PyTorch version and CUDA tag
    if cuda_tag == "cpu":
        libtorch_url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-{0}%2Bcpu.zip".format(pytorch_version)
    else:
        libtorch_url = "https://download.pytorch.org/libtorch/{0}/libtorch-shared-with-deps-{1}%2B{0}.zip".format(cuda_tag, pytorch_version)

    print("Downloading LibTorch from:", libtorch_url)

    # Download and extract Libtorch, stripping the top-level libtorch directory
    download_result = repository_ctx.download_and_extract(
        url = libtorch_url,
        output = ".",
        sha256 = sha256,  # Use provided SHA256 for verification (empty string if not specified)
        stripPrefix = "libtorch",  # Strip the top-level libtorch directory
    )

    # Debug: List extracted files to verify structure
    print("Listing extracted files:")
    repository_ctx.execute(["ls", "-l", "include/torch"])

    # Create BUILD file for Libtorch
    repository_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "libtorch",
    srcs = glob(
        [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll*",
        ],
        # Exclude test and benchmark libraries to reduce size
        exclude = [
            "lib/*test*",
            "lib/*benchmark*",
        ],
    ),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
        "include/**/*.h++",
    ]),
    includes = [
        "include",                    # Base include directory
        "include/torch",              # Explicitly include torch subdirectory
        "include/torch/csrc/api/include",  # Additional include paths
    ],
    visibility = ["//visibility:public"],
    linkstatic = 1,
    # Link options for CUDA support (if applicable)
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],  # macOS
        "//conditions:default": [  # Linux/Windows
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
)
""",
    )

def _is_valid_pytorch_version(version):
    """Validate PyTorch version format (X.Y.Z)."""
    parts = version.split(".")
    if len(parts) != 3:
        return False
    digits = [part.isdigit() for part in parts]
    return all(digits)

libtorch_repository = repository_rule(
    implementation = _libtorch_repository_rule_impl,
    attrs = {
        "pytorch_version": attr.string(
            mandatory = True,
            doc = "PyTorch version (e.g., '2.5.1').",
        ),
        "cuda_tag": attr.string(
            mandatory = True,
            doc = "CUDA tag (e.g., 'cu121', 'cpu').",
        ),
        "sha256": attr.string(
            default = "",
            doc = "SHA256 checksum for the Libtorch download (optional).",
        ),
    },
    local = False,
    doc = """
Downloads and configures Libtorch for the specified PyTorch version and CUDA tag.

Example:
    libtorch_repository(
        name = "libtorch",
        pytorch_version = "2.5.1",
        cuda_tag = "cu121",
        sha256 = "YOUR_SHA256_HERE",
    )
""",
)
