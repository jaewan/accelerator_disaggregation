#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "remote_execution.grpc.pb.h"

#include <ATen/ATen.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/OperatorName.h>
#include <c10/core/Dispatcher.h>
#include <c10/util/Optional.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using remote_execution::OpRequest;
using remote_execution::OpResponse;
using remote_execution::RemoteExecutionService;
using remote_execution::TensorProto;

// Helper to deserialize TensorProto to at::Tensor
at::Tensor protoToTensor(const TensorProto& proto) {
    std::vector<int64_t> shape(proto.shape().begin(), proto.shape().end());
    at::ScalarType dtype = at::kFloat; // TODO: handle more dtypes
    
    at::Tensor tensor = at::empty(shape, at::TensorOptions().dtype(dtype));
    std::memcpy(tensor.data_ptr(), proto.data().data(), proto.data().size());

    return tensor;
}

// Helper to serialize at::Tensor to TensorProto
TensorProto tensorToProto(const at::Tensor& tensor) {
    TensorProto proto;
    proto.set_dtype(tensor.dtype().name());
    for (auto s : tensor.sizes()) {
        proto.add_shape(s);
    }

    auto tensor_contig = tensor.contiguous();
    proto.set_data(tensor_contig.data_ptr(), tensor_contig.nbytes());

    return proto;
}

class RemoteExecutionServiceImpl final : public RemoteExecutionService::Service {
    Status ExecuteOp(ServerContext* context, const OpRequest* request, OpResponse* response) override {
        std::cout << "[SERVER] Received op: " << request->op_name() << std::endl;

        // Build stack
        c10::Stack stack;

        // Parse arguments (only Tensor for now)
        for (const auto& tensor_proto : request->arguments()) {
            at::Tensor tensor = protoToTensor(tensor_proto);
            stack.push_back(tensor);
        }

        // Lookup operator
        c10::OperatorName op_name(request->op_name(), request->overload_name());
        auto op = c10::Dispatcher::singleton().findSchemaOrThrow(op_name);

        // Dispatch op
        try {
            op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), &stack);
        } catch (const std::exception& e) {
            return Status(grpc::StatusCode::UNKNOWN, std::string("Op execution failed: ") + e.what());
        }

        // Get result
        auto result_ivalue = stack.back();
        stack.pop_back();

        if (!result_ivalue.isTensor()) {
            return Status(grpc::StatusCode::UNKNOWN, "Non-tensor results are not supported yet");
        }

        at::Tensor result_tensor = result_ivalue.toTensor();
        *response->mutable_result() = tensorToProto(result_tensor);

        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    RemoteExecutionServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[SERVER] Listening on " << server_address << std::endl;

    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}

