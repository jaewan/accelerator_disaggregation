#include "remote_dispatch.h"

#include "absl/container/flat_hash_set.h"
#include <c10/core/SymIntArrayRef.h> 
#include <c10/core/SymInt.h> 
#include <ATen/native/CPUFallback.h>
#include <c10/core/CPUAllocator.h>
#include "proto/remote_execution.grpc.pb.h"
#include <grpcpp/grpcpp.h>

/*
 * Fallback works for operators that do not have a more specific kernel registered for PrivateUse1 device.
 * However, some core tensor creation ops (like aten::empty.memory_format) require explicit registration 
 * because they are part of PyTorch's core tensor allocation logic.
 * We explicitly implemented the bare minimum kernels
 */

namespace remote_cuda {

//TODO(Jae) complete this
void* remote_allocate(size_t total_bytes){
	return malloc(total_bytes);
}

at::ArrayRef<at::Tensor> extract_tensors(c10::Stack& stack) {
	std::vector<at::Tensor> tensors;
	for (const c10::IValue& value : stack) { // Iterate through the stack.  Consider only extracting the tensors for now.  This logic needs refined.
		if (value.isTensor()) {
			tensors.push_back(value.toTensor());
		}
	}
	return at::ArrayRef<at::Tensor>(tensors); // Return as at::ArrayRef
}

// Set of operations that should NOT be transferred to the remote GPU
// Add more operations as needed, using their full schema name (aten::...)
const absl::flat_hash_set<std::string> kLocalOps = {
	/*
		 "aten::size",      // Device-agnostic: size of a tensor
		 "aten::stride",    // Device-agnostic: stride of a tensor
		 "aten::dim",       // Device-agnostic: dimension of a tensor
		 "aten::is_cuda",   // Device-agnostic: check if tensor is on CUDA
		 "aten::is_cpu",    // Device-agnostic: check if tensor is on CPU
		 "aten::device",    // Device-agnostic: gets device of tensor
		 "aten::numel",     // Device-agnostic: number of elements in the tensor
		 "aten::is_contiguous", //Device-agnostic: check if contiguous

		 "aten::item",      // CPU-specific: gets a number from a 1-element tensor
		 "aten::to",        // CPU-specific: Moving data. We will reimplement a different way
		 "aten::cpu",       // CPU-specific: Move to CPU
		 "aten::numpy_T",   // CPU-specific: Convert Tensor to numpy

		 "aten::print",     // CPU-specific: Printing
		 "aten::println",   // CPU-specific: Printing a new line
		 "aten::set_printoptions", // CPU-specific: set printing options

		 */
		 "aten::empty", // Directly allocate tensor on local cpu currently
		 "aten::normal_", // Need to support Generator if removed
		 "aten::view",
		 "aten::reshape",
		 "aten::as_strided",
	// Add more operations as needed
};

at::Tensor change_tensor_device_to_remote_cuda(const at::Tensor &cpu_result){
	size_t tensor_size = cpu_result.numel() * cpu_result.element_size();
	void* remote_ptr = remote_allocate(tensor_size);
	memcpy(remote_ptr, cpu_result.data_ptr(), tensor_size);

	at::TensorOptions options;
	options = options.dtype(cpu_result.scalar_type());
	options = options.layout(cpu_result.layout());
	options = options.device(c10::Device(REMOTE_CUDA_TYPE, 0)); // Important: Specify your custom device
	std::vector<int64_t> dims_vec(cpu_result.sizes().begin(), cpu_result.sizes().end());

	at::Tensor tensor = at::from_blob(remote_ptr, dims_vec, cpu_result.strides(), options);
	return tensor;
}

void execute_op_test(const c10::OperatorHandle& op, c10::Stack* stack) {
	SPDLOG_INFO("test fallback called : {}", op.schema().name());
	
    // Determine the start index of arguments on the stack
    const auto& schema_args = op.schema().arguments();
    const size_t num_args = schema_args.size();
    const size_t args_begin = stack->size() - num_args;
    auto args = torch::jit::last(stack, num_args);

    // Containers for single-tensor inputs
    std::vector<int> tensor_indices;
    std::vector<at::Tensor> remote_tensors;
    std::vector<at::Tensor> cpu_tensors;

    // Containers for TensorList inputs
    std::vector<int> list_indices;
    std::vector<std::vector<at::Tensor>> remote_lists;
    std::vector<std::vector<at::Tensor>> cpu_lists;

    // Optional Device argument (if present)
    std::optional<c10::Device> tgt_device;

    // Step 1: Convert all REMOTE_CUDA tensors/lists to CPU
    for (size_t idx = 0; idx < args.size(); ++idx) {
        auto& iv = args[idx];
        // Handle Device arguments
        if (iv.isDevice()) {
            tgt_device = iv.toDevice();
            (*stack)[args_begin + idx] = c10::IValue(c10::Device(c10::DeviceType::CPU));
            continue;
        }
        // Handle single Tensor
        if (iv.isTensor()) {
            auto t = iv.toTensor();
            if (t.device().type() == REMOTE_CUDA_TYPE) {
                tensor_indices.push_back(idx);
                remote_tensors.push_back(t);
                cpu_tensors.push_back(t.cpu());
                (*stack)[args_begin + idx] = c10::IValue(cpu_tensors.back());
            }
        }
        // Handle TensorList
        else if (iv.isTensorList()) {
            auto tl = iv.toTensorList().vec();
            bool has_remote = false;
            std::vector<at::Tensor> cpu_list;
            cpu_list.reserve(tl.size());
            for (auto& t : tl) {
                if (t.device().type() == REMOTE_CUDA_TYPE) {
                    has_remote = true;
                    cpu_list.push_back(t.cpu());
                } else {
                    cpu_list.push_back(t);
                }
            }
            if (has_remote) {
                list_indices.push_back(idx);
                remote_lists.push_back(std::move(tl));
                cpu_lists.push_back(cpu_list);
                (*stack)[args_begin + idx] = c10::IValue(c10::List<at::Tensor>(cpu_list));
            }
        }
    }

    // Step 2: Call the CPU implementation
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

    // Step 3a: Sync in-place writes for single-tensor inputs
    for (size_t i = 0; i < tensor_indices.size(); ++i) {
        int idx = tensor_indices[i];
        const auto* alias_info = schema_args[idx].alias_info();
        if (alias_info && alias_info->isWrite()) {
            auto& cpu_t = cpu_tensors[i];
            auto& orig  = remote_tensors[i];
            if (cpu_t.defined() && orig.defined()) {
                at::_copy_from_and_resize(cpu_t, orig);
            }
        }
    }

    // Step 3b: Sync in-place writes for TensorList inputs
    for (size_t i = 0; i < list_indices.size(); ++i) {
        int idx = list_indices[i];
        const auto* alias_info = schema_args[idx].alias_info();
        if (alias_info && alias_info->isWrite()) {
            const auto& cpu_list = cpu_lists[i];
            auto& orig_list = remote_lists[i];
            for (size_t j = 0; j < orig_list.size(); ++j) {
                const auto& cpu_t = cpu_list[j];
                auto& orig_t = orig_list[j];
                if (cpu_t.defined() && orig_t.defined()) {
                    at::_copy_from_and_resize(cpu_t, orig_t);
                }
            }
        }
    }

    // Step 4: Convert outputs back to REMOTE_CUDA device
    const auto& schema_rets = op.schema().returns();
    const size_t num_rets = schema_rets.size();
    const size_t rets_begin = stack->size() - num_rets;
    if (!tgt_device.has_value() && !remote_tensors.empty()) {
        tgt_device = remote_tensors[0].device();
    }
    for (size_t i = 0; i < num_rets; ++i) {
        auto& ov = (*stack)[rets_begin + i];
        if (!tgt_device.has_value()) continue;
        // Single Tensor output
        if (ov.isTensor() && ov.toTensor().defined()) {
            auto out_cpu = ov.toTensor();
            ov = c10::IValue(change_tensor_device_to_remote_cuda(out_cpu));
        }
        // TensorList output
        else if (ov.isTensorList()) {
            auto tl = ov.toTensorList().vec();
            std::vector<at::Tensor> new_list;
            new_list.reserve(tl.size());
            for (auto& t : tl) {
                new_list.push_back(change_tensor_device_to_remote_cuda(t));
            }
            ov = c10::IValue(c10::List<at::Tensor>(new_list));
        }
    }
}

// Helper to serialize tensor to TensorProto
remote_execution::TensorProto tensorToProto(const at::Tensor& tensor) {
    remote_execution::TensorProto proto;
    
    // proto.set_dtype(tensor.dtype().name());
	proto.set_dtype(std::string(tensor.dtype().name()));
    for (auto s : tensor.sizes()) {
        proto.add_shape(s);
    }

    auto tensor_contig = tensor.contiguous();
    proto.set_data(tensor_contig.data_ptr(), tensor_contig.nbytes());

    return proto;
}

remote_execution::ScalarProto scalarToProto(const c10::IValue& iv) {
    remote_execution::ScalarProto p;
    if (iv.isInt())        p.set_i(iv.toInt());
    else if (iv.isDouble()) p.set_f(iv.toDouble());
    else if (iv.isBool())   p.set_b(iv.toBool());
    else if (iv.isString()) p.set_s(iv.toStringRef());
    else throw std::runtime_error("Unsupported scalar type");
    return p;
}

c10::IValue scalarFromProto(const remote_execution::ScalarProto& p) {
    switch (p.value_case()) {
      case remote_execution::ScalarProto::kI: return (int64_t)p.i();
      case remote_execution::ScalarProto::kF: return (double)p.f();
      case remote_execution::ScalarProto::kB: return (bool)p.b();
      case remote_execution::ScalarProto::kS: return p.s();
      default: throw std::runtime_error("Unknown scalar");
    }
}

remote_execution::ListProto listToProto(const c10::IValue& iv) {
    if (!iv.isList()) {
        throw std::runtime_error("Expected List, got non-list IValue");
    }
    auto elems = iv.toListRef();
    remote_execution::ListProto p;
    for (const auto& elem : elems) {
        remote_execution::Arg* a = p.add_values();
        if (elem.isTensor())     *a->mutable_tensor() = tensorToProto(elem.toTensor());
        else if (elem.isScalar()) *a->mutable_scalar() = scalarToProto(elem);
        else if (elem.isList())   *a->mutable_list()   = listToProto(elem);
    }
    return p;
}

// Helper to deserialize TensorProto to at::Tensor
at::Tensor protoToTensor(const remote_execution::TensorProto& proto) {
    std::vector<int64_t> shape(proto.shape().begin(), proto.shape().end());
    auto options = at::TensorOptions().dtype(at::ScalarType::Float);  // Add other types as needed
    
    at::Tensor tensor = at::empty(shape, options);
    std::memcpy(tensor.data_ptr(), proto.data().data(), proto.data().size());

    return tensor;
}

c10::IValue listFromProto(const remote_execution::ListProto& lp) {
    c10::impl::GenericList gl(c10::AnyType::get());
    for (const auto& arg : lp.values()) {
        if (arg.has_tensor())  gl.push_back(protoToTensor(arg.tensor()));
        else if (arg.has_scalar()) gl.push_back(scalarFromProto(arg.scalar()));
        else if (arg.has_list())   gl.push_back(listFromProto(arg.list()));
    }
    return gl;
}

// Function to execute an operation on the remote server
c10::IValue execute_op_remotely(
    const std::string& op_name,
    const std::string& overload_name,
    c10::Stack* stack) 
{
    SPDLOG_INFO("[DEBUG] Executing remote op {}/{}",
                op_name, overload_name);

    remote_execution::OpRequest request;
    request.set_op_name(op_name);
    request.set_overload_name(overload_name);
    
    // Helper function to detect if a List contains only Tensors
    auto isTensorList = [](const c10::IValue& iv) -> bool {
        if (!iv.isList()) return false;
        
        auto list_ref = iv.toListRef();
        if (list_ref.empty()) return false;
        
        for (const auto& item : list_ref) {
            if (!item.isTensor()) return false;
        }
        return true;
    };
    
    for (size_t i = 0; i < stack->size(); ++i) {
        const auto& iv = (*stack)[i];
        
        if (iv.isTensor()) {
            SPDLOG_DEBUG("[CLIENT] Arg {} type=Tensor", i);
            auto* arg = request.add_arguments();
            *arg->mutable_tensor() = tensorToProto(iv.toTensor());
        }
        else if (iv.isTensorList()) {
            SPDLOG_DEBUG("[CLIENT] Arg {} type=TensorList with {} tensors", i, iv.toTensorList().size());
            
            auto* arg = request.add_arguments();
            auto tensor_list = iv.toTensorList().vec();
            for (const auto& t : tensor_list) {
                *arg->mutable_tensor_list()->add_tensors() = tensorToProto(t);
            }
            SPDLOG_DEBUG("[CLIENT] Added {} tensors to TensorList", tensor_list.size());
        }
        else if (isTensorList(iv)) {
            // General handling for List[Tensor] - convert to TensorList for proper serialization
            SPDLOG_DEBUG("[CLIENT] Arg {} type=List containing only Tensors, converting to TensorList", i);
            
            auto* arg = request.add_arguments();
            auto list_ref = iv.toListRef();
            for (const auto& item : list_ref) {
                *arg->mutable_tensor_list()->add_tensors() = tensorToProto(item.toTensor());
            }
            SPDLOG_DEBUG("[CLIENT] Converted List to TensorList with {} tensors", list_ref.size());
        }
        else if (iv.isScalar()) {
            SPDLOG_DEBUG("[CLIENT] Arg {} type=Scalar", i);
            auto* arg = request.add_arguments();
            *arg->mutable_scalar() = scalarToProto(iv);
        }
        else if (iv.isList()) {
            SPDLOG_DEBUG("[CLIENT] Arg {} type=List", i);
            auto* arg = request.add_arguments();
            *arg->mutable_list() = listToProto(iv);
        }
        else if (iv.isIntList()) {
            SPDLOG_DEBUG("[CLIENT] Arg {} type=IntList", i);
            auto dims = iv.toIntList();  // c10::IntArrayRef
            auto* arg = request.add_arguments();
            for (auto d : dims) {
                auto* entry = arg->mutable_list()->add_values();
                entry->mutable_scalar()->set_i(d);
            }
        }
        else {
            SPDLOG_DEBUG("[CLIENT] Arg {} type=Unknown: {}", i, iv.tagKind());
        }
    }

    SPDLOG_DEBUG("[CLIENT] Created request with {} arguments", request.arguments_size());

    auto channel = grpc::CreateChannel(
        "localhost:50051", grpc::InsecureChannelCredentials());
    auto stub = remote_execution::RemoteExecutionService::NewStub(channel);
    grpc::ClientContext ctx;
    remote_execution::OpResponse resp;
    
    SPDLOG_DEBUG("[CLIENT] Sending RPC request");
    auto status = stub->ExecuteOp(&ctx, request, &resp);
    if (!status.ok()) {
        SPDLOG_ERROR("[CLIENT] RPC failed: {} ({})", status.error_message(), status.error_code());
        throw std::runtime_error("RPC failed: " + status.error_message());
    }

    const auto& result_arg = resp.result();
    if (result_arg.has_tensor()) {
        SPDLOG_INFO("[CLIENT] Received result type=Tensor");
        return protoToTensor(result_arg.tensor());
    }
    else if (result_arg.has_tensor_list()) {
        SPDLOG_INFO("[CLIENT] Received result type=TensorList");
        std::vector<at::Tensor> tensors;
        for (const auto& tp : result_arg.tensor_list().tensors()) {
            tensors.push_back(protoToTensor(tp));
        }
        return c10::IValue(c10::List<at::Tensor>(std::move(tensors)));
    }
    else if (result_arg.has_scalar()) {
        SPDLOG_INFO("[CLIENT] Received result type=Scalar");
        return scalarFromProto(result_arg.scalar());
    }
    else if (result_arg.has_list()) {
        SPDLOG_INFO("[CLIENT] Received result type=List");
        return listFromProto(result_arg.list());
    }
    else {
        SPDLOG_ERROR("[CLIENT] RPC returned empty Arg");
        throw std::runtime_error("Empty Arg in response");
    }
}

// Function to execute operation locally
void execute_op_locally(const c10::OperatorHandle& op, c10::Stack* stack) {
	SPDLOG_INFO("[DEBUG] Executing operation locally {}",op.schema().name());

	// The correct way to call op locally. Figure out how to do this properly
	//auto kernel = c10::Dispatcher::singleton().findSchema(op.schema());
	//kernel.call(stack);
	// Redispatch to CPU backend
	//op.redispatchBoxed(c10::DispatchKeySet(at::DispatchKey::CPU), stack);

	// Slower but more stable and suggested version
	// at::native::cpu_fallback(op, stack);
	execute_op_test(op, stack);
}

// Define a boxed fallback function outside the registerFallback call
void remote_cuda_fallback(const c10::OperatorHandle& op, c10::Stack* stack) {
    const auto& schema = op.schema();
    const std::string& op_name = schema.name();            // e.g. "aten::add"
    std::string overload_name = schema.overload_name();   // e.g. "out" or "Tensor"

    SPDLOG_INFO("[DEBUG] remote_cuda_fallback called: {}.{}", op_name, overload_name);

    if (kLocalOps.count(op_name)) {
        execute_op_locally(op, stack);
        return;
    }

    // Store out variant
    bool is_out = (overload_name == "out");
    
    // Map the overloaded operation correctly
    // For many operations with 'out' variant, the correct schema uses no overload name or just "Tensor"
    std::string remote_overload = overload_name;
    if (is_out) {
        // Try to find the appropriate non-out schema
        // Use empty string first (most common for base implementations)
        remote_overload = "";
        
        // Check if the non-out schema exists
        try {
            c10::Dispatcher::singleton().findSchema({op_name, remote_overload});
            SPDLOG_DEBUG("[CLIENT] Found matching schema without overload: {}", op_name);
        } catch (const c10::Error& e) {
            // Try with "Tensor" overload as fallback
            try {
                remote_overload = "Tensor";
                c10::Dispatcher::singleton().findSchema({op_name, remote_overload});
                SPDLOG_DEBUG("[CLIENT] Found matching schema with Tensor overload: {}.{}", op_name, remote_overload);
            } catch (const c10::Error& e2) {
                // If both fail, keep original overload and let server handle it
                remote_overload = overload_name;
                SPDLOG_DEBUG("[CLIENT] Could not find matching schema, keeping original: {}.{}", op_name, remote_overload);
            }
        }
    }

    c10::IValue out_placeholder;
    if (is_out) {
        out_placeholder = stack->back();
        stack->pop_back();
    }

    c10::IValue result = execute_op_remotely(op_name, remote_overload, stack);

    stack->clear();

    if (is_out && result.isTensor()) {
        at::_copy_from_and_resize(
            result.toTensor(),
            out_placeholder.toTensor()
        );
        stack->push_back(out_placeholder);
    } 
    else if (result.isTensor()) {
        at::Tensor cpu_res = result.toTensor();
        at::Tensor remote_res = change_tensor_device_to_remote_cuda(cpu_res);
        stack->push_back(remote_res);
    }
    // TensorList handling
    else if (result.isTensorList()) {
        auto tensor_list = result.toTensorList().vec();
        std::vector<at::Tensor> remote_tensors;
        remote_tensors.reserve(tensor_list.size());
        
        for (const auto& cpu_tensor : tensor_list) {
            remote_tensors.push_back(change_tensor_device_to_remote_cuda(cpu_tensor));
        }
        
        stack->push_back(c10::IValue(c10::List<at::Tensor>(std::move(remote_tensors))));
    }
    // Scalar, List, etc.
    else {
        stack->push_back(result);
    }
}

void register_dispatch_keys() {
	// Even though this is an empty function, calling this is critical
	SPDLOG_INFO("Register dispatch keys called");
}

//----------- Bare Minimum Operations -----------
at::Tensor handle_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, 
		c10::optional<c10::Layout> layout_opt, c10::optional<c10::Device> device_opt, 
		c10::optional<bool> pin_memory_opt) {
	SPDLOG_INFO("[DEBUG] empty_strided called");
	// Ensure the device is of type REMOTE_CUDA_TYPE
	TORCH_CHECK(device_opt.has_value() && device_opt->type() == REMOTE_CUDA_TYPE, 
			"empty_strided: Expected device of type REMOTE_CUDA_TYPE");

	// 1. Determine the data type
	at::ScalarType scalar_type = dtype_opt.value_or(at::kFloat);

	// Validate layout
	TORCH_CHECK(layout_opt.value_or(c10::kStrided) == c10::kStrided, 
			"empty_strided: Only supports strided layout");
	//TODO For now, we don't handle pinned memory
	TORCH_CHECK(!pin_memory_opt.has_value() || !pin_memory_opt.value(), 
			"empty_strided: Pinned memory is not supported on remote_cuda");

	// 2. Calculate the total size in bytes
	int64_t num_elements = 1;
	for (int64_t dim : size) {
		num_elements *= dim;
	}
	size_t element_size = at::elementSize(scalar_type);
	size_t total_bytes = num_elements * element_size;

	// 3. Allocate memory on the remote device
	//    (This is where you'd communicate with your remote server)
	void* remote_ptr = remote_allocate(total_bytes);

	// 4. Construct a TensorOptions object
	at::TensorOptions options;
	options = options.dtype(scalar_type);
	options = options.layout(layout_opt.value_or(at::kStrided));
	options = options.device(device_opt.value_or(c10::Device(REMOTE_CUDA_TYPE, 0))); // Important: Specify your custom device

	// 5. Create a tensor from the remote memory
	at::Tensor tensor = at::from_blob(remote_ptr, size, stride, options);
	return tensor;
}

//TODO(Yue) make this to return proxy tensor instead of actual tensor and see if it works
at::Tensor test_handle_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, 
		c10::optional<c10::Layout> layout_opt, c10::optional<c10::Device> device_opt, 
		c10::optional<bool> pin_memory_opt) {
	SPDLOG_INFO("[DEBUG] empty_strided called");
	// Ensure the device is of type REMOTE_CUDA_TYPE
	TORCH_CHECK(device_opt.has_value() && device_opt->type() == REMOTE_CUDA_TYPE, 
			"empty_strided: Expected device of type REMOTE_CUDA_TYPE");

	// 1. Determine the data type
	at::ScalarType scalar_type = dtype_opt.value_or(at::kFloat);

	// Validate layout
	TORCH_CHECK(layout_opt.value_or(c10::kStrided) == c10::kStrided, 
			"empty_strided: Only supports strided layout");
	//TODO For now, we don't handle pinned memory
	TORCH_CHECK(!pin_memory_opt.has_value() || !pin_memory_opt.value(), 
			"empty_strided: Pinned memory is not supported on remote_cuda");

	// 2. Calculate the total size in bytes
	int64_t num_elements = 1;
	for (int64_t dim : size) {
		num_elements *= dim;
	}
	size_t element_size = at::elementSize(scalar_type);
	size_t total_bytes = num_elements * element_size;

	// 3. Allocate memory on the remote device
	//    (This is where you'd communicate with your remote server)
	void* remote_ptr = remote_allocate(total_bytes);

	// 4. Construct a TensorOptions object
	at::TensorOptions options;
	options = options.dtype(scalar_type);
	options = options.layout(layout_opt.value_or(at::kStrided));
	options = options.device(device_opt.value_or(c10::Device(REMOTE_CUDA_TYPE, 0))); // Important: Specify your custom device

	// 5. Create a tensor from the remote memory
	at::Tensor tensor = at::from_blob(remote_ptr, size, stride, options);
	return tensor;
}

at::Tensor handle_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
	SPDLOG_INFO("[DEBUG] [Manual Kernel] copy_from called");
	// // Ensure the destination tensor is on your custom device
	// TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
	// 		"_copy_from: Destination tensor must be on the REMOTE_CUDA device");

	// // Ensure the source tensor is not on your custom device
	// TORCH_CHECK(self.device().type() != c10::DeviceType::PrivateUse1,
	// 		"_copy_from: Source tensor must not be on the REMOTE_CUDA device");

	// 1. Serialize the source tensor's data
	const void* src_data = self.data_ptr();
	size_t src_num_bytes = self.nbytes();

	// For demonstration, log the copy operation
	SPDLOG_INFO("[DEBUG] [Manual Kernel] Copying {} bytes from {} to {}", 
				src_num_bytes, self.device().str(), dst.device().str());

	// 2. Allocate memory on the remote device (if not already allocated)
	void* dest_data = dst.data_ptr();
	TORCH_CHECK(dest_data, "_copy_from: Destination tensor's data pointer is null");

	// 3. Simulate copying the data to the remote device
	memcpy(dest_data, src_data, src_num_bytes);

	// 4. Return the destination tensor
	return dst;
}

at::Tensor handle_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
	SPDLOG_INFO("[DEBUG] [Manual Kernel] copy_from_and_resize called");
	return handle_copy_from(self, dst, false);
}


at::Tensor& handle_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
	SPDLOG_INFO("[DEBUG] [Manual Kernel] copy_ called");
	/*
		 TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
		 "copy_: Destination tensor must be on the REMOTE_CUDA device {}",self.device().type());

		 TORCH_CHECK(src.device().type() != c10::DeviceType::PrivateUse1,
		 "copy_: Source tensor must not be on the REMOTE_CUDA device");
		 */

	// 1. Serialize the source tensor's data
	const void* src_data = src.data_ptr();
	size_t src_num_bytes = src.nbytes();

	// 2. Copy the data to the destination tensor
	void* dest_data = self.data_ptr();
	TORCH_CHECK(dest_data, "copy_: Destination tensor's data pointer is null");

	memcpy(dest_data, src_data, src_num_bytes);

	// 3. Return the modified destination tensor
	return self;
}

at::Tensor handle_to(const at::Tensor& self, c10::Device device, at::ScalarType dtype, bool non_blocking, bool copy) {
	SPDLOG_INFO("[DEBUG] [Manual Kernel] to called");
	if (device.type() == c10::DeviceType::PrivateUse1) {
		// Create a new tensor on the REMOTE_CUDA device
		at::Tensor result = at::empty_strided(self.sizes(), self.strides(), self.options().device(device).dtype(dtype));
		return handle_copy_from(self, result, non_blocking);
	} else {
		TORCH_CHECK(false, "handle_to: Only supports moving to REMOTE_CUDA for now");
	}
}

at::Tensor const& handle_resize_(at::Tensor const& self,
		c10::ArrayRef<c10::SymInt> size,
		c10::optional<c10::MemoryFormat> memory_format) {
	SPDLOG_INFO("[DEBUG] [Manual Kernel] resize called");
	// Get a mutable reference to work with
	at::Tensor& mutable_self = const_cast<at::Tensor&>(self);

	// Ensure the tensor is on the REMOTE_CUDA device
	TORCH_CHECK(mutable_self.device().type() == c10::DeviceType::PrivateUse1,
			"resize_: Tensor must be on the REMOTE_CUDA device");

	// Convert c10::ArrayRef<c10::SymInt> to std::vector<int64_t>
	std::vector<int64_t> size_vec;
	size_vec.reserve(size.size());
	for (const auto& symint : size) {
		size_vec.push_back(symint.expect_int());
	}

	// Handle the memory format
	if (memory_format.has_value()) {
		TORCH_CHECK(
				memory_format.value() == c10::MemoryFormat::Contiguous,
				"resize_: Only contiguous memory format is supported for remote_cuda"
				);
	}

	// Perform the resize operation
	mutable_self.resize_(size_vec);

	// Return the const reference as required by the signature
	return self;
}

} // namespace remote_cuda


//TORCH_LIBRARY_IMPL(aten, c10::DispatchKey::PrivateUse1, m) {
/*
	 TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
	 m.impl("empty_strided", remote_cuda::handle_empty_strided);
	 m.impl("_copy_from", remote_cuda::handle_copy_from);
	 m.impl("to", remote_cuda::handle_to);
	 m.impl("resize_", remote_cuda::handle_resize_);
	 m.impl("copy_", remote_cuda::handle_copy_);
	 }
	 */
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
	m.impl("empty_strided", remote_cuda::handle_empty_strided);
	m.impl("_copy_from", remote_cuda::handle_copy_from);
	m.impl("_copy_from_and_resize", remote_cuda::handle_copy_from_and_resize);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
	m.fallback(torch::CppFunction::makeFromBoxedFunction<&remote_cuda::remote_cuda_fallback>());
	// m.fallback(torch::CppFunction::makeFromBoxedFunction<&remote_cuda::execute_op_test>());
}
