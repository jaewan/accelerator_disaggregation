#include <torch/extension.h>
#include "remote_device.h"
#include "remote_dispatch.h"

void setup_logging() {
	try {
		auto logger = spdlog::rotating_logger_mt("device_logger", "/tmp/remote_cuda_log/device.log",
				10*1024*1024, 3);
		spdlog::set_default_logger(logger);

		// Pattern with source location:
		spdlog::set_pattern("[%H:%M:%S.%e][%t][%s:%#] %v");
		spdlog::flush_on(spdlog::level::info);

		// Or if you want a more detailed pattern:
		// spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [thread %t] [%s:%#:%!] %v");
	} catch (const spdlog::spdlog_ex& ex) {
		std::cerr << "Log initialization failed: " << ex.what() << std::endl;
	}
}

// Create the Python module
PYBIND11_MODULE(remote_cuda_ext, m) {
		setup_logging();
    // Register our device. Can do this from __init__.py if some configs are required
    remote_cuda::register_device();
    remote_cuda::register_dispatch_keys();

    // Expose the device type to Python
    m.attr("REMOTE_CUDA") = py::int_(static_cast<int>(remote_cuda::REMOTE_CUDA_TYPE));

    // -------- Add any additional bindings here --------
		m.def("register_device", &remote_cuda::register_device, 
				"Register remote CUDA device type with PyTorch");
		m.def("register_dispatch_keys", &remote_cuda::register_dispatch_keys,
				"Register dispatcher keys for remote operations");
}
