#pragma once

#include <torch/extension.h>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <chrono>
#include <atomic>
#include <mutex>

// Forward declarations for the generated gRPC code
namespace remote {
class RemoteExecutor;
}

namespace rpc_client {

enum class ErrorCode {
    OK = 0,
    CONNECTION_FAILED,
    OPERATION_FAILED,
    INVALID_ARGUMENT,
    TIMEOUT,
    UNKNOWN
};

struct Error {
    ErrorCode code;
    std::string message;
    
    static Error ok() { return {ErrorCode::OK, ""}; }
    static Error connection_failed(const std::string& msg) { return {ErrorCode::CONNECTION_FAILED, msg}; }
    static Error operation_failed(const std::string& msg) { return {ErrorCode::OPERATION_FAILED, msg}; }
    static Error invalid_argument(const std::string& msg) { return {ErrorCode::INVALID_ARGUMENT, msg}; }
    static Error timeout(const std::string& msg) { return {ErrorCode::TIMEOUT, msg}; }
    static Error unknown(const std::string& msg) { return {ErrorCode::UNKNOWN, msg}; }
    
    bool is_ok() const { return code == ErrorCode::OK; }
    explicit operator bool() const { return !is_ok(); }
};

// Configuration for the RPC client
struct ClientConfig {
    // Connection settings
    std::string server_address = "localhost:50051";
    std::chrono::milliseconds connection_timeout = std::chrono::milliseconds(5000);
    std::chrono::milliseconds operation_timeout = std::chrono::milliseconds(30000);
    
    // Reconnection settings
    bool enable_reconnect = true;
    int max_reconnect_attempts = 5;
    std::chrono::milliseconds reconnect_backoff_ms = std::chrono::milliseconds(1000);
    
    // Performance settings
    int max_send_message_size = 50 * 1024 * 1024;  // 50 MB
    int max_receive_message_size = 50 * 1024 * 1024;  // 50 MB
    bool use_compression = true;
    
    // Advanced settings
    bool enable_keepalive = true;
    std::chrono::seconds keepalive_time = std::chrono::seconds(60);
    std::chrono::seconds keepalive_timeout = std::chrono::seconds(20);
};

// Initialize with custom configuration
Error initialize(const ClientConfig& config = ClientConfig());

// Connection management
bool is_connected();
Error reconnect();
void shutdown();

// Device management
int get_device_count(Error* error = nullptr);
Error set_device(int device_index);
int get_current_device(Error* error = nullptr);

// Memory management
void* alloc(size_t nbytes, Error* error = nullptr);
Error free(void* ptr);

// Data transfer
Error upload_tensor_data(void* remote_ptr, const void* cpu_ptr, size_t nbytes);
Error download_tensor_data(const void* remote_ptr, void* cpu_ptr, size_t nbytes);

// Operation execution
at::Tensor execute_op(const char* op_name, const char* overload_name, 
                     at::ArrayRef<at::Tensor> tensors, c10::Stack& stack, 
                     Error* error = nullptr);

// Helper functions for common operations
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b, Error* error = nullptr);
at::Tensor multiply_tensors(const at::Tensor& a, const at::Tensor& b, Error* error = nullptr);
at::Tensor matmul_tensors(const at::Tensor& a, const at::Tensor& b, Error* error = nullptr);

// Error handling
const char* error_code_to_string(ErrorCode code);

// Internal use only - don't use these directly
namespace internal {
    std::shared_ptr<grpc::Channel> get_channel();
    std::unique_ptr<remote::RemoteExecutor::Stub>& get_stub();
    grpc::ClientContext create_context(std::chrono::milliseconds timeout);
    Error status_to_error(const grpc::Status& status);
}

} // namespace rpc_client
