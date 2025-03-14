#include "rpc_client.h"
#include "remote.grpc.pb.h"
#include <thread>
#include <chrono>
#include <sstream>
#include <iostream>
#include <atomic>

namespace rpc_client {

// Global state
namespace {
    std::mutex g_mutex;
    std::shared_ptr<grpc::Channel> g_channel;
    std::unique_ptr<remote::RemoteExecutor::Stub> g_stub;
    ClientConfig g_config;
    std::atomic<bool> g_is_connected{false};
    std::atomic<int> g_current_device{0};

    // Helper function to create channel with appropriate options
    std::shared_ptr<grpc::Channel> create_channel(const ClientConfig& config) {
        grpc::ChannelArguments args;

        if (config.use_compression) {
            args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
        }

        args.SetMaxSendMessageSize(config.max_send_message_size);
        args.SetMaxReceiveMessageSize(config.max_receive_message_size);

        if (config.enable_keepalive) {
            args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,
                      static_cast<int>(config.keepalive_time.count() * 1000));
            args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
                      static_cast<int>(config.keepalive_timeout.count() * 1000));
            args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        }

        return grpc::CreateCustomChannel(
            config.server_address,
            grpc::InsecureChannelCredentials(),  // In production, use secure credentials
            args
        );
    }

    // Helper for tensor type conversion
    remote::TensorInfo::DataType torch_to_proto_dtype(c10::ScalarType dtype) {
        switch (dtype) {
            case c10::ScalarType::Float: return remote::TensorInfo::FLOAT;
            case c10::ScalarType::Double: return remote::TensorInfo::DOUBLE;
            case c10::ScalarType::Int: return remote::TensorInfo::INT;
            case c10::ScalarType::Long: return remote::TensorInfo::LONG;
            case c10::ScalarType::Bool: return remote::TensorInfo::BOOL;
            case c10::ScalarType::Byte: return remote::TensorInfo::BYTE;
            case c10::ScalarType::Short: return remote::TensorInfo::SHORT;
            case c10::ScalarType::Half: return remote::TensorInfo::HALF;
            case c10::ScalarType::ComplexFloat: return remote::TensorInfo::COMPLEX_FLOAT;
            case c10::ScalarType::ComplexDouble: return remote::TensorInfo::COMPLEX_DOUBLE;
            case c10::ScalarType::BFloat16: return remote::TensorInfo::BFLOAT16;
            default:
                throw std::runtime_error("Unsupported tensor data type");
        }
    }

    c10::ScalarType proto_to_torch_dtype(remote::TensorInfo::DataType dtype) {
        switch (dtype) {
            case remote::TensorInfo::FLOAT: return c10::ScalarType::Float;
            case remote::TensorInfo::DOUBLE: return c10::ScalarType::Double;
            case remote::TensorInfo::INT: return c10::ScalarType::Int;
            case remote::TensorInfo::LONG: return c10::ScalarType::Long;
            case remote::TensorInfo::BOOL: return c10::ScalarType::Bool;
            case remote::TensorInfo::BYTE: return c10::ScalarType::Byte;
            case remote::TensorInfo::SHORT: return c10::ScalarType::Short;
            case remote::TensorInfo::HALF: return c10::ScalarType::Half;
            case remote::TensorInfo::COMPLEX_FLOAT: return c10::ScalarType::ComplexFloat;
            case remote::TensorInfo::COMPLEX_DOUBLE: return c10::ScalarType::ComplexDouble;
            case remote::TensorInfo::BFLOAT16: return c10::ScalarType::BFloat16;
            default:
                throw std::runtime_error("Unsupported proto data type");
        }
    }

    // Convert gRPC status to our Error type
    Error grpc_status_to_error(const grpc::Status& status) {
        if (status.ok()) {
            return Error::ok();
        }

        switch (status.error_code()) {
            case grpc::StatusCode::UNAVAILABLE:
                return Error::connection_failed(status.error_message());
            case grpc::StatusCode::INVALID_ARGUMENT:
                return Error::invalid_argument(status.error_message());
            case grpc::StatusCode::DEADLINE_EXCEEDED:
                return Error::timeout(status.error_message());
            default:
                return Error::operation_failed(status.error_message());
        }
    }

    // Convert protobuf Status to our Error type
    Error proto_status_to_error(const remote::Status& status) {
        if (status.code() == remote::Status::OK) {
            return Error::ok();
        }

        std::string message = status.message();
        switch (status.code()) {
            case remote::Status::INVALID_ARGUMENT:
                return Error::invalid_argument(message);
            case remote::Status::DEADLINE_EXCEEDED:
                return Error::timeout(message);
            case remote::Status::UNAVAILABLE:
                return Error::connection_failed(message);
            default:
                return Error::operation_failed(message);
        }
    }
}

namespace internal {

std::shared_ptr<grpc::Channel> get_channel() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_channel;
}

std::unique_ptr<remote::RemoteExecutor::Stub>& get_stub() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_stub;
}

grpc::ClientContext create_context(std::chrono::milliseconds timeout) {
    grpc::ClientContext context;
    if (timeout.count() > 0) {
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::now() + timeout;
        context.set_deadline(deadline);
    }
    return context;
}

Error status_to_error(const grpc::Status& status) {
    return grpc_status_to_error(status);
}

} // namespace internal

// Connection management
Error initialize(const ClientConfig& config) {
    std::lock_guard<std::mutex> lock(g_mutex);

    // Store config for reconnection
    g_config = config;

    // Create channel with appropriate options
    g_channel = create_channel(config);
    g_stub = remote::RemoteExecutor::NewStub(g_channel);

    // Try to connect with timeout
    auto timeout = std::chrono::system_clock::now() + config.connection_timeout;
    bool connected = false;

    while (std::chrono::system_clock::now() < timeout) {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));

        remote::PingRequest request;
        remote::PingResponse response;

        grpc::Status status = g_stub->Ping(&context, request, &response);

        if (status.ok() && response.status().code() == remote::Status::OK) {
            connected = true;
            g_is_connected.store(true);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!connected) {
        g_channel.reset();
        g_stub.reset();
        return Error::connection_failed("Failed to connect to remote executor within timeout");
    }

    return Error::ok();
}

bool is_connected() {
    return g_is_connected.load();
}

Error reconnect() {
    if (!g_config.enable_reconnect) {
        return Error::connection_failed("Reconnection is disabled in configuration");
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    // Try to reconnect with exponential backoff
    Error last_error = Error::unknown("No reconnection attempts made");

    for (int attempt = 0; attempt < g_config.max_reconnect_attempts; ++attempt) {
        // Create a new channel and stub
        g_channel = create_channel(g_config);
        g_stub = remote::RemoteExecutor::NewStub(g_channel);

        // Try to ping
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() +
                            std::chrono::milliseconds(g_config.connection_timeout));

        remote::PingRequest request;
        remote::PingResponse response;

        grpc::Status status = g_stub->Ping(&context, request, &response);

        if (status.ok() && response.status().code() == remote::Status::OK) {
            g_is_connected.store(true);
            return Error::ok();
        }

        last_error = grpc_status_to_error(status);

        // Backoff before next attempt
        auto backoff = g_config.reconnect_backoff_ms * (1 << attempt);
        std::this_thread::sleep_for(backoff);
    }

    // Clean up failed connection
    g_channel.reset();
    g_stub.reset();
    g_is_connected.store(false);

    return last_error;
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);

    // If connected, try to gracefully shutdown the server connection
    if (g_is_connected.load() && g_stub) {
        grpc::ClientContext context;
        remote::ShutdownRequest request;
        remote::ShutdownResponse response;

        g_stub->Shutdown(&context, request, &response);
    }

    g_stub.reset();
    g_channel.reset();
    g_is_connected.store(false);
}

// Device management
int get_device_count(Error* error) {
    if (!is_connected()) {
        if (error) *error = Error::connection_failed("Not connected to remote executor");
        return 0;
    }

    grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
    remote::DeviceCountRequest request;
    remote::DeviceCountResponse response;

    grpc::Status status = g_stub->GetDeviceCount(&context, request, &response);

    if (!status.ok()) {
        Error err = grpc_status_to_error(status);
        if (error) *error = err;

        // Try to reconnect on connection failures if enabled
        if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
            g_is_connected.store(false);
            reconnect();
        }

        return 0;
    }

    if (response.status().code() != remote::Status::OK) {
        if (error) *error = proto_status_to_error(response.status());
        return 0;
    }

    if (error) *error = Error::ok();
    return response.count();
}

Error set_device(int device_index) {
    if (!is_connected()) {
        return Error::connection_failed("Not connected to remote executor");
    }

    grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
    remote::SetDeviceRequest request;
    remote::SetDeviceResponse response;

    request.set_device_index(device_index);

    grpc::Status status = g_stub->SetDevice(&context, request, &response);

    if (!status.ok()) {
        Error err = grpc_status_to_error(status);

        // Try to reconnect on connection failures if enabled
        if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
            g_is_connected.store(false);
            reconnect();
        }

        return err;
    }

    if (response.status().code() != remote::Status::OK) {
        return proto_status_to_error(response.status());
    }

    g_current_device.store(device_index);
    return Error::ok();
}

int get_current_device(Error* error) {
    if (!is_connected()) {
        if (error) *error = Error::connection_failed("Not connected to remote executor");
        return 0;
    }

    grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
    remote::GetCurrentDeviceRequest request;
    remote::GetCurrentDeviceResponse response;

    grpc::Status status = g_stub->GetCurrentDevice(&context, request, &response);

    if (!status.ok()) {
        Error err = grpc_status_to_error(status);
        if (error) *error = err;

        // Try to reconnect on connection failures if enabled
        if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
            g_is_connected.store(false);
            reconnect();
        }

        return g_current_device.load();
    }

    if (response.status().code() != remote::Status::OK) {
        if (error) *error = proto_status_to_error(response.status());
        return g_current_device.load();
    }

    g_current_device.store(response.device_index());
    if (error) *error = Error::ok();
    return response.device_index();
}

// Memory management
void* alloc(size_t nbytes, Error* error) {
    if (!is_connected()) {
        if (error) *error = Error::connection_failed("Not connected to remote executor");
        return nullptr;
    }

    grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
    remote::AllocateRequest request;
    remote::AllocateResponse response;

    request.set_size(nbytes);
    request.set_device_index(g_current_device.load());

    grpc::Status status = g_stub->Allocate(&context, request, &response);

    if (!status.ok()) {
        Error err = grpc_status_to_error(status);
        if (error) *error = err;

        // Try to reconnect on connection failures if enabled
        if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
            g_is_connected.store(false);
            reconnect();
        }

        return nullptr;
    }

    if (response.status().code() != remote::Status::OK) {
        if (error) *error = proto_status_to_error(response.status());
        return nullptr;
    }

    if (error) *error = Error::ok();
    return reinterpret_cast<void*>(response.memory_handle());
}

Error free(void* ptr) {
    if (!is_connected()) {
        return Error::connection_failed("Not connected to remote executor");
    }

    grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
    remote::FreeRequest request;
    remote::FreeResponse response;

    request.set_memory_handle(reinterpret_cast<uint64_t>(ptr));

    grpc::Status status = g_stub->Free(&context, request, &response);

    if (!status.ok()) {
        Error err = grpc_status_to_error(status);

        // Try to reconnect on connection failures if enabled
        if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
            g_is_connected.store(false);
            reconnect();
        }

        return err;
    }

    return proto_status_to_error(response.status());
}

// Data transfer
Error upload_tensor_data(void* remote_ptr, const void* cpu_ptr, size_t nbytes) {
    if (!is_connected()) {
        return Error::connection_failed("Not connected to remote executor");
    }

    constexpr size_t max_chunk_size = 4 * 1024 * 1024;  // 4MB chunks to avoid gRPC size limits
    uint64_t offset = 0;

    while (offset < nbytes) {
        size_t chunk_size = std::min(max_chunk_size, nbytes - offset);

        grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
        remote::UploadTensorDataRequest request;
        remote::UploadTensorDataResponse response;

        request.set_memory_handle(reinterpret_cast<uint64_t>(remote_ptr));
        request.set_data(static_cast<const char*>(cpu_ptr) + offset, chunk_size);
        request.set_size(chunk_size);
        request.set_offset(offset);

        grpc::Status status = g_stub->UploadTensorData(&context, request, &response);

        if (!status.ok()) {
            Error err = grpc_status_to_error(status);

            // Try to reconnect on connection failures if enabled
            if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
                g_is_connected.store(false);
                reconnect();
            }

            return err;
        }

        if (response.status().code() != remote::Status::OK) {
            return proto_status_to_error(response.status());
        }

        offset += chunk_size;
    }

    return Error::ok();
}

Error download_tensor_data(const void* remote_ptr, void* cpu_ptr, size_t nbytes) {
    if (!is_connected()) {
        return Error::connection_failed("Not connected to remote executor");
    }

    constexpr size_t max_chunk_size = 4 * 1024 * 1024;  // 4MB chunks to avoid gRPC size limits
    uint64_t offset = 0;

    while (offset < nbytes) {
        size_t chunk_size = std::min(max_chunk_size, nbytes - offset);

        grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
        remote::DownloadTensorDataRequest request;
        remote::DownloadTensorDataResponse response;

        request.set_memory_handle(reinterpret_cast<uint64_t>(remote_ptr));
        request.set_size(chunk_size);
        request.set_offset(offset);

        grpc::Status status = g_stub->DownloadTensorData(&context, request, &response);

        if (!status.ok()) {
            Error err = grpc_status_to_error(status);

            // Try to reconnect on connection failures if enabled
            if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
                g_is_connected.store(false);
                reconnect();
            }

            return err;
        }

        if (response.status().code() != remote::Status::OK) {
            return proto_status_to_error(response.status());
        }

        // Copy data to CPU buffer
        std::memcpy(static_cast<char*>(cpu_ptr) + offset, response.data().data(), response.data().size());
        offset += response.data().size();
    }

    return Error::ok();
}

// Helper to fill in TensorInfo from a Tensor
void fill_tensor_info(remote::TensorInfo* info, const at::Tensor& tensor, uint64_t memory_handle) {
    info->set_memory_handle(memory_handle);
    info->set_data_type(torch_to_proto_dtype(tensor.scalar_type()));
    info->set_requires_grad(tensor.requires_grad());

    for (auto& size : tensor.sizes()) {
        info->add_sizes(size);
    }

    for (auto& stride : tensor.strides()) {
        info->add_strides(stride);
    }
}

// Helper to serialize IValue to ArgumentValue
void serialize_ivalue(const c10::IValue& ivalue, remote::ArgumentValue* arg_value) {
    if (ivalue.isInt()) {
        arg_value->set_int_value(ivalue.toInt());
    } else if (ivalue.isDouble()) {
        arg_value->set_double_value(ivalue.toDouble());
    } else if (ivalue.isBool()) {
        arg_value->set_bool_value(ivalue.toBool());
    } else if (ivalue.isString()) {
        arg_value->set_string_value(ivalue.toStringRef());
    } else if (ivalue.isTensor()) {
        at::Tensor tensor = ivalue.toTensor();
        uint64_t memory_handle = reinterpret_cast<uint64_t>(tensor.data_ptr());

        auto* tensor_info = arg_value->mutable_tensor_value();
        fill_tensor_info(tensor_info, tensor, memory_handle);
    } else if (ivalue.isList()) {
        auto list = ivalue.toList();
        auto* list_value = arg_value->mutable_list_value();

        for (size_t i = 0; i < list.size(); ++i) {
            auto* element = list_value->add_elements();
            serialize_ivalue(list.get(i), element);
        }
    } else if (ivalue.isGenericDict()) {
        auto dict = ivalue.toGenericDict();
        auto* dict_value = arg_value->mutable_dict_value();

        for (auto& pair : dict) {
            auto* item = dict_value->add_items();
            serialize_ivalue(pair.key(), item->mutable_key());
            serialize_ivalue(pair.value(), item->mutable_value());
        }
    }
    // Add more types as needed
}

// Operation execution
at::Tensor execute_op(const char* op_name, const char* overload_name,
                     at::ArrayRef<at::Tensor> tensors, c10::Stack& stack,
                     Error* error) {
    if (!is_connected()) {
        if (error) *error = Error::connection_failed("Not connected to remote executor");
        return at::Tensor();
    }

    grpc::ClientContext context = internal::create_context(g_config.operation_timeout);
    remote::ExecuteOperationRequest request;
    remote::ExecuteOperationResponse response;

    request.set_op_name(op_name);
    if (overload_name) {
        request.set_overload_name(overload_name);
    }

    // Add input tensors
    for (const auto& tensor : tensors) {
        auto* tensor_info = request.add_input_tensors();
        fill_tensor_info(tensor_info, tensor, reinterpret_cast<uint64_t>(tensor.data_ptr()));
    }

    // Add arguments from stack
    for (const auto& arg : stack) {
        auto* arg_value = request.add_arguments();
        serialize_ivalue(arg, arg_value);
    }

    grpc::Status status = g_stub->ExecuteOperation(&context, request, &response);

    if (!status.ok()) {
        Error err = grpc_status_to_error(status);
        if (error) *error = err;

        // Try to reconnect on connection failures if enabled
        if (err.code == ErrorCode::CONNECTION_FAILED && g_config.enable_reconnect) {
            g_is_connected.store(false);
            reconnect();
        }

        return at::Tensor();
    }

    if (response.status().code() != remote::Status::OK) {
        if (error) *error = proto_status_to_error(response.status());
        return at::Tensor();
    }

    // Convert result tensor to a local tensor handle
    const remote::TensorInfo& result_info = response.result_tensor();

    // Extract tensor metadata
    std::vector<int64_t> sizes(result_info.sizes().begin(), result_info.sizes().end());
    std::vector<int64_t> strides(result_info.strides().begin(), result_info.strides().end());
    c10::ScalarType dtype = proto_to_torch_dtype(result_info.data_type());

    // Create options with the appropriate device and dtype
    auto options = at::TensorOptions()
        .dtype(dtype)
        .device(c10::Device(c10::DeviceType::PrivateUse1, g_current_device.load()));

    // Create a tensor that wraps the remote memory
    void* data_ptr = reinterpret_cast<void*>(result_info.memory_handle());

    at::Tensor result_tensor = at::from_blob(
        data_ptr,
        sizes,
        strides,
        [data_ptr](void*) {
            // Free the remote memory when the tensor is destroyed
            rpc_client::free(data_ptr);
        },
        options
    );

    // Set requires_grad if needed
    if (result_info.requires_grad()) {
        result_tensor.set_requires_grad(true);
    }

    if (error) *error = Error::ok();
    return result_tensor;
}

// Helper functions for common operations
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b, Error* error) {
    c10::Stack stack;
    stack.push_back(a);
    stack.push_back(b);
    return execute_op("aten::add", "", {a, b}, stack, error);
}

at::Tensor multiply_tensors(const at::Tensor& a, const at::Tensor& b, Error* error) {
    c10::Stack stack;
    stack.push_back(a);
    stack.push_back(b);
    return execute_op("aten::mul", "", {a, b}, stack, error);
}

at::Tensor matmul_tensors(const at::Tensor& a, const at::Tensor& b, Error* error) {
    c10::Stack stack;
    stack.push_back(a);
    stack.push_back(b);
    return execute_op("aten::matmul", "", {a, b}, stack, error);
}

// Error utilities
const char* error_code_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::OK: return "OK";
        case ErrorCode::CONNECTION_FAILED: return "CONNECTION_FAILED";
        case ErrorCode::OPERATION_FAILED: return "OPERATION_FAILED";
        case ErrorCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ErrorCode::TIMEOUT: return "TIMEOUT";
        case ErrorCode::UNKNOWN: return "UNKNOWN";
        default: return "UNRECOGNIZED_ERROR";
    }
}

} // namespace rpc_client
