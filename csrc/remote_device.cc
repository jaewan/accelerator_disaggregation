#include "remote_device.h"
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <spdlog/spdlog.h>

namespace remote_cuda {

// Register our device guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, RemoteCUDAGuardImpl);

class RemoteCUDAPrivateUse1Hooks : public at::PrivateUse1HooksInterface {
	public:
		const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override {
			static at::Generator generator;
			return generator;
		}

		at::Device getDeviceFromPtr(void* data) const override {
			return c10::Device(c10::DeviceType::PrivateUse1, 0);
		}

		bool isPinnedPtr(const void* data) const override {
			return false;
		}

		c10::Allocator* getPinnedMemoryAllocator() const override {
			throw std::runtime_error("Pinned memory allocator not implemented yet");
		}

		bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
			return false;
		}

		void initPrivateUse1() const override {
			// Initialize any state required for your device
		}

		void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
			// Update storage size if needed (not implemented here)
		}
};

void register_device() {
	static RemoteCUDAPrivateUse1Hooks private_use_1_hooks;
	// This implementation directly uses Aten/detail code which is discouraged
	// as it is pytorch internal code and subject to change without notice
	// As of pytoch 2.5.1 there is no public API for PrivateUse1
	at::RegisterPrivateUse1HooksInterface(&private_use_1_hooks);

	SPDLOG_INFO("Remote CUDA device registration completed");
}

} // namespace remote_cuda
