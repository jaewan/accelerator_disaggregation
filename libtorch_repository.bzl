def _libtorch_repository_rule_impl(ctx):
    """Implementation of the libtorch_repository rule.

    Downloads and extracts Libtorch based on the specified PyTorch version and CUDA tag.
    Generates a BUILD file for the Libtorch library.
    """

    # Get attributes from the repository rule
    pytorch_version = ctx.attr.pytorch_version
    cuda_tag = ctx.attr.cuda_tag
    sha256 = ctx.attr.sha256

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

    # Prepare download arguments
    download_args = {
        "url": libtorch_url,
        "output": ".",
        # Don't strip the prefix - keep the full directory structure
    }

    # Only add sha256 if it's not empty
    if sha256:
        download_args["sha256"] = sha256

    # Download and extract Libtorch
    download_result = ctx.download_and_extract(**download_args)

    # Debug: List extracted files to verify structure
    print("Listing extracted files:")
    ctx.execute(["ls", "-l", "libtorch/include/torch"])
    ctx.execute(["find", "libtorch", "-name", "device_guard.h"])

    # Create BUILD file for Libtorch
    ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "libtorch",
    srcs = glob(
        [
            "libtorch/lib/*.so*",
            "libtorch/lib/*.dylib*",
            "libtorch/lib/*.dll*",
        ],
        # Exclude test and benchmark libraries to reduce size
        exclude = [
            "libtorch/lib/*test*",
            "libtorch/lib/*benchmark*",
        ],
    ),
    hdrs = glob([
        "libtorch/include/**/*.h",
        "libtorch/include/**/*.hpp",
        "libtorch/include/**/*.cuh",
        "libtorch/include/**/*.h++",
        "libtorch/include/**/*.inl",
    ]),
    includes = [
        "libtorch/include",
        "libtorch/include/torch/csrc/api/include",
        "libtorch/include/torch/csrc/utils",  # Add missing utils directory
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
        sha256 = "YOUR_SHA256_HERE",  # Optional
    )
    """,
)
