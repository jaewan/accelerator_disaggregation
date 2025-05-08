#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

#include <grpcpp/grpcpp.h>
#include "proto/remote_execution.grpc.pb.h"
#include "proto/remote_execution.pb.h"

#include <torch/script.h>                // for torch::jit::Stack
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Optional.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using remote_execution::TensorProto;
using remote_execution::ScalarProto;
using remote_execution::ListProto;
using remote_execution::Arg;
using remote_execution::OpRequest;
using remote_execution::OpResponse;
using remote_execution::RemoteExecutionService;

// --- (De)serialization Helpers ---

// Tensor ↔ TensorProto
at::Tensor protoToTensor(const TensorProto& p) {
    std::vector<int64_t> shape(p.shape().begin(), p.shape().end());
    size_t expected_bytes = 1;
    for (auto d : shape) expected_bytes *= d;
    expected_bytes *= sizeof(float);  // assume float
    if (p.data().size() != expected_bytes) {
        std::cerr << "[SERVER][WARN] Tensor bytes mismatch: "
                  << p.data().size() << " vs " << expected_bytes
                  << std::endl;
    }
    auto t = at::empty(shape, at::kFloat);
    std::memcpy(t.data_ptr(), p.data().data(),
                std::min(p.data().size(), expected_bytes));
    return t;
}

TensorProto tensorToProto(const at::Tensor& t) {
    TensorProto p;
    // Wrap name() string_view into std::string for Protobuf :contentReference[oaicite:6]{index=6}
    p.set_dtype(std::string(t.dtype().name()));
    std::cout << "[SERVER] Serializing tensor dtype=" << t.dtype().name()
              << " sizes=";
    for (auto d : t.sizes()) std::cout << d << ",";
    std::cout << std::endl;
    for (auto d : t.sizes()) p.add_shape(d);
    auto ct = t.contiguous();
    p.set_data(ct.data_ptr(), ct.nbytes());
    return p;
}

// Scalar ↔ ScalarProto
c10::IValue scalarFromProto(const ScalarProto& p) {
    switch (p.value_case()) {
      case ScalarProto::kI: return (int64_t)p.i();
      case ScalarProto::kF: return (double)p.f();
      case ScalarProto::kB: return (bool)p.b();
      case ScalarProto::kS: return p.s();
      default: throw std::runtime_error("Unknown scalar");
    }
}

ScalarProto scalarToProto(const c10::IValue& iv) {
    ScalarProto p;
    if (iv.isInt())        p.set_i(iv.toInt());
    else if (iv.isDouble()) p.set_f(iv.toDouble());
    else if (iv.isBool())   p.set_b(iv.toBool());
    else if (iv.isString()) p.set_s(iv.toStringRef());
    else throw std::runtime_error("Unsupported scalar type");
    return p;
}

// List ↔ ListProto using isList() & toListRef() :contentReference[oaicite:7]{index=7}
c10::IValue listFromProto(const ListProto& lp) {
    c10::impl::GenericList gl(c10::AnyType::get());
    for (const auto& arg : lp.values()) {
        if (arg.has_tensor())  gl.push_back(protoToTensor(arg.tensor()));
        else if (arg.has_scalar()) gl.push_back(scalarFromProto(arg.scalar()));
        else if (arg.has_list())   gl.push_back(listFromProto(arg.list()));
    }
    return gl;
}

ListProto listToProto(const c10::IValue& iv) {
    if (!iv.isList()) {
        throw std::runtime_error("Expected List, got non-list IValue");
    }
    auto elems = iv.toListRef();  // ArrayRef<IValue> iteration :contentReference[oaicite:8]{index=8}
    ListProto p;
    for (const auto& elem : elems) {
        Arg* a = p.add_values();
        if (elem.isTensor())     *a->mutable_tensor() = tensorToProto(elem.toTensor());
        else if (elem.isScalar()) *a->mutable_scalar() = scalarToProto(elem);
        else if (elem.isList())   *a->mutable_list()   = listToProto(elem);
    }
    return p;
}

// Convert a tensor list into a TensorList proto
void tensorListToProto(const std::vector<at::Tensor>& tensors, remote_execution::TensorListProto* proto) {
    for (const auto& tensor : tensors) {
        *proto->add_tensors() = tensorToProto(tensor);
    }
}

// Convert TensorListProto to a vector of tensors
std::vector<at::Tensor> protoToTensorList(const remote_execution::TensorListProto& proto) {
    std::vector<at::Tensor> tensors;
    tensors.reserve(proto.tensors_size());
    for (const auto& tp : proto.tensors()) {
        tensors.push_back(protoToTensor(tp));
    }
    return tensors;
}

// Any Arg ↔ IValue
c10::IValue argToIValue(const Arg& a) {
    if (a.has_tensor()) return protoToTensor(a.tensor());
    if (a.has_tensor_list()) {
        std::vector<at::Tensor> vec = protoToTensorList(a.tensor_list());
        return c10::IValue(c10::List<at::Tensor>(std::move(vec)));
    }
    if (a.has_scalar()) return scalarFromProto(a.scalar());
    if (a.has_list()) {
        const auto& lp = a.list();
        bool all_ints = true;
        for (const auto& entry : lp.values()) {
            if (!entry.has_scalar() || entry.scalar().value_case() != ScalarProto::kI) {
                all_ints = false;
                break;
            }
        }
        if (all_ints) {
            std::vector<int64_t> dims;
            dims.reserve(lp.values_size());
            for (const auto& entry : lp.values()) {
                dims.push_back(entry.scalar().i());
            }
            return c10::IValue(dims);
        }
        return listFromProto(lp);
    }

    throw std::runtime_error("Empty or Unsupportted Arg");
}

Arg iValueToArg(const c10::IValue& iv) {
    Arg a;
    if (iv.isTensor()) {
        *a.mutable_tensor() = tensorToProto(iv.toTensor());
    }
    else if (iv.isTensorList()) {
        auto tensors = iv.toTensorList().vec();
        tensorListToProto(tensors, a.mutable_tensor_list());
    }
    else if (iv.isScalar()) {
        *a.mutable_scalar() = scalarToProto(iv);
    }
    else if (iv.isList()) {
        *a.mutable_list() = listToProto(iv);
    }
    else {
        std::string type_name = iv.tagKind();
        throw std::runtime_error("Unsupported return type: " + type_name);
    }
    return a;
}

// Helper function to find the correct operator schema
c10::OperatorHandle findOperatorSchema(const std::string& op_name, const std::string& overload_name) {
    // Try with the provided overload first
    try {
        return c10::Dispatcher::singleton().findSchemaOrThrow(op_name.c_str(), overload_name.c_str());
    } catch (const c10::Error& e) {
        // If that fails, try known alternate overloads
        
        // 1. Try with empty overload (many ops use this as default)
        if (!overload_name.empty()) {
            try {
                return c10::Dispatcher::singleton().findSchemaOrThrow(op_name.c_str(), "");
            } catch (const c10::Error&) {
                // Continue to next attempt
            }
        }
        
        // 2. Try with "Tensor" overload (common for arithmetic ops)
        if (overload_name != "Tensor") {
            try {
                return c10::Dispatcher::singleton().findSchemaOrThrow(op_name.c_str(), "Tensor");
            } catch (const c10::Error&) {
                // Continue to next attempt
            }
        }
        
        // 3. Special case handling for known ops
        // For add/sub/mul/div without overload, try "Tensor"
        if (overload_name.empty() && 
            (op_name == "aten::add" || op_name == "aten::sub" || 
             op_name == "aten::mul" || op_name == "aten::div")) {
            try {
                return c10::Dispatcher::singleton().findSchemaOrThrow(op_name.c_str(), "Tensor");
            } catch (const c10::Error&) {
                // Continue to next attempt
            }
        }
        
        // 4. Try with "int" overload (common for some ops like softmax)
        if (overload_name != "int") {
            try {
                return c10::Dispatcher::singleton().findSchemaOrThrow(op_name.c_str(), "int");
            } catch (const c10::Error&) {
                // Continue to next attempt
            }
        }
        
        // Rethrow the original error if all attempts fail
        throw;
    }
}

// --- Service Implementation ---

class RemoteExecutionServiceImpl final 
    : public RemoteExecutionService::Service {
  Status ExecuteOp(ServerContext* /*ctx*/, const OpRequest* req, OpResponse* resp) override {
    std::cout << "[SERVER] ==> ExecuteOp start: "
              << req->op_name() << "/" << req->overload_name()
              << " args=" << req->arguments_size() << std::endl;

    try {
        torch::jit::Stack stack;
        for (int i = 0; i < req->arguments_size(); ++i) {
            try {
                auto iv = argToIValue(req->arguments(i));
                std::cout << "[SERVER]  - Arg " << i 
                        << (iv.isTensor()    ? " Tensor" :
                           iv.isScalar()    ? " Scalar" :
                           iv.isList()      ? " List" :
                           iv.isTensorList() ? " TensorList" : " Other")
                        << std::endl;
                
                if (iv.isTensor()) {
                    auto t = iv.toTensor();
                    std::cout << "[SERVER]    Tensor shape=[";
                    for (auto d : t.sizes()) {
                        std::cout << d << ",";
                    }
                    std::cout << "] dtype=" << t.dtype().name() << std::endl;
                }
                else if (iv.isTensorList()) {
                    auto tensors = iv.toTensorList().vec();
                    std::cout << "[SERVER]    TensorList with " << tensors.size() << " tensors" << std::endl;
                    for (size_t j = 0; j < tensors.size(); j++) {
                        std::cout << "[SERVER]      Tensor[" << j << "] shape=[";
                        for (auto d : tensors[j].sizes()) {
                            std::cout << d << ",";
                        }
                        std::cout << "] dtype=" << tensors[j].dtype().name() << std::endl;
                    }
                }
                
                stack.push_back(iv);
            } catch (const std::exception& e) {
                std::cerr << "[SERVER][ERROR] Failed to process argument " << i 
                        << ": " << e.what() << std::endl;
                return Status(grpc::StatusCode::INVALID_ARGUMENT, 
                             "Invalid argument at index " + std::to_string(i) + ": " + e.what());
            }
        }

        const std::string& op_name = req->op_name();
        const std::string& overload_name = req->overload_name();
        std::cout << "[SERVER] Dispatching " << op_name << "/" << overload_name << std::endl;
        
        // Try to find the schema with our helper function
        try {
            auto op = findOperatorSchema(op_name, overload_name);
            std::cout << "[SERVER] Found schema: " << op.schema() << std::endl;
            std::cout << "[SERVER] Argument count: " << stack.size() << std::endl;
            
            try {
                at::native::cpu_fallback(op, &stack);
                std::cout << "[SERVER] Operation completed successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[SERVER][ERROR] Dispatch failed: " << e.what() << std::endl;
                return Status(grpc::StatusCode::INTERNAL, "Dispatch error: " + std::string(e.what()));
            }
        } catch (const c10::Error& e) {
            std::cerr << "[SERVER][ERROR] Schema not found for op " << op_name << "/" << overload_name 
                      << ": " << e.what() << std::endl;
            return Status(grpc::StatusCode::NOT_FOUND, 
                        "Schema not found: " + std::string(e.what()));
        }

        auto out_iv = stack.back();
        std::cout << "[SERVER] Result type: "
                << (out_iv.isTensor() ? "Tensor" :
                    out_iv.isScalar() ? "Scalar" :
                    out_iv.isList()   ? "List" : 
                    out_iv.isTensorList() ? "TensorList" : "Other")
                << std::endl;
                
        if (out_iv.isTensor()) {
            auto t = out_iv.toTensor();
            std::cout << "[SERVER]    Tensor shape=[";
            for (auto d : t.sizes()) {
                std::cout << d << ",";
            }
            std::cout << "] dtype=" << t.dtype().name() << std::endl;
        }
        else if (out_iv.isTensorList()) {
            auto tensors = out_iv.toTensorList().vec();
            std::cout << "[SERVER]    TensorList with " << tensors.size() << " tensors" << std::endl;
        }
        
        try {
            *resp->mutable_result() = iValueToArg(out_iv);
            std::cout << "[SERVER] Response serialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[SERVER][ERROR] Failed to serialize result: " << e.what() << std::endl;
            return Status(grpc::StatusCode::INTERNAL, "Serialization error: " + std::string(e.what()));
        }

        std::cout << "[SERVER] <== ExecuteOp done" << std::endl;
        return Status::OK;
    } catch (const std::exception& e) {
        std::cerr << "[SERVER][ERROR] Unexpected error in ExecuteOp: " << e.what() << std::endl;
        return Status(grpc::StatusCode::INTERNAL, "Unexpected error: " + std::string(e.what()));
    } catch (...) {
        std::cerr << "[SERVER][ERROR] Unknown error in ExecuteOp" << std::endl;
        return Status(grpc::StatusCode::INTERNAL, "Unknown error");
    }
  }
};

void RunServer() {
  std::string addr("0.0.0.0:50051");
  RemoteExecutionServiceImpl svc;
  ServerBuilder b;
  b.AddListeningPort(addr, grpc::InsecureServerCredentials());
  b.RegisterService(&svc);
  auto server = b.BuildAndStart();
  std::cout << "[SERVER] Listening on " << addr << std::endl;
  server->Wait();
}

int main() {
  std::cout << "[SERVER] Starting up..." << std::endl;
  RunServer();
  return 0;
}
