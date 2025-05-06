#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "proto/remote_execution.grpc.pb.h"
#include "proto/remote_execution.pb.h"

#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Optional.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using remote_execution::OpRequest;
using remote_execution::OpResponse;
using remote_execution::RemoteExecutionService;
using remote_execution::TensorProto;

// Deserialize
at::Tensor protoToTensor(const TensorProto& proto) {
    std::vector<int64_t> shape(proto.shape().begin(), proto.shape().end());
    size_t expected_bytes = 1;
    for (auto s : shape) expected_bytes *= s;
    expected_bytes *= sizeof(float);  // assuming float

    if (proto.data().size() != expected_bytes) {
        std::cerr << "[SERVER][WARN] Data size mismatch: "
                  << proto.data().size() << " vs expected " << expected_bytes
                  << std::endl;
    }
    at::Tensor tensor = at::empty(shape, at::TensorOptions().dtype(at::kFloat));
    std::memcpy(tensor.data_ptr(), proto.data().data(),
                std::min(proto.data().size(), expected_bytes));
    return tensor;
}

// Serialize
TensorProto tensorToProto(const at::Tensor& tensor) {
    TensorProto proto;
    auto dtype_name = std::string(tensor.dtype().name());
    proto.set_dtype(dtype_name);
    std::cout << "[SERVER] Serializing tensor (dtype=" << dtype_name
              << ", sizes=";
    for (auto s : tensor.sizes()) std::cout << s << ",";
    std::cout << ")" << std::endl;

    for (auto s : tensor.sizes()) {
        proto.add_shape(s);
    }
    auto contig = tensor.contiguous();
    proto.set_data(contig.data_ptr(), contig.nbytes());
    return proto;
}

class RemoteExecutionServiceImpl final : public RemoteExecutionService::Service {
    Status ExecuteOp(ServerContext* context,
                     const OpRequest* request,
                     OpResponse* response) override {
        std::cout << "[SERVER] ==> ExecuteOp start" << std::endl;
        std::cout << "[SERVER] Received op_name=" << request->op_name()
                  << ", overload=" << request->overload_name()
                  << ", #args=" << request->arguments_size()
                  << std::endl;

        // Build stack
        torch::jit::Stack stack;
        for (int i = 0; i < request->arguments_size(); ++i) {
            auto& arg_proto = request->arguments(i);
            at::Tensor t = protoToTensor(arg_proto);
            std::cout << "[SERVER]  - Arg " << i << " tensor sizes="
                      << t.sizes() << std::endl;
            stack.push_back(t);
        }

        // Lookup operator
        const char* name_c    = request->op_name().c_str();
        const char* overload_c= request->overload_name().c_str();
        std::cout << "[SERVER] Looking up schema " << name_c
                  << "/" << overload_c << std::endl;
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow(name_c, overload_c);

        // Dispatch
        try {
            std::cout << "[SERVER] Dispatching on CPU" << std::endl;
            op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU),
                               &stack);
        } catch (const std::exception& e) {
            std::cerr << "[SERVER][ERROR] Dispatch failed: "
                      << e.what() << std::endl;
            return Status(grpc::StatusCode::UNKNOWN,
                          std::string("Dispatch error: ") + e.what());
        }

        // Get result
        auto result_ivalue = stack.back();
        if (!result_ivalue.isTensor()) {
            std::cerr << "[SERVER][ERROR] Result is not a tensor"
                      << std::endl;
            return Status(grpc::StatusCode::UNKNOWN,
                          "Result not tensor");
        }
        at::Tensor result = result_ivalue.toTensor();
        std::cout << "[SERVER] Execution complete, result sizes="
                  << result.sizes() << std::endl;

        *response->mutable_result() = tensorToProto(result);
        std::cout << "[SERVER] <== ExecuteOp done" << std::endl;
        return Status::OK;
    }
};

void RunServer() {
    std::string addr("0.0.0.0:50051");
    RemoteExecutionServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[SERVER] Listening on " << addr << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    std::cout << "[SERVER] Starting up..." << std::endl;
    RunServer();
    return 0;
}
