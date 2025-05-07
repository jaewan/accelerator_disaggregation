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

// Any Arg ↔ IValue
c10::IValue argToIValue(const Arg& a) {
    if (a.has_tensor()) return protoToTensor(a.tensor());
    if (a.has_scalar()) return scalarFromProto(a.scalar());
    if (a.has_list())   return listFromProto(a.list());
    throw std::runtime_error("Empty Arg");
}

Arg iValueToArg(const c10::IValue& iv) {
    Arg a;
    if (iv.isTensor())        *a.mutable_tensor() = tensorToProto(iv.toTensor());
    else if (iv.isScalar())   *a.mutable_scalar() = scalarToProto(iv);
    else if (iv.isList())     *a.mutable_list()   = listToProto(iv);
    else throw std::runtime_error("Unsupported return type");
    return a;
}

// --- Service Implementation ---

class RemoteExecutionServiceImpl final 
    : public RemoteExecutionService::Service {
  Status ExecuteOp(ServerContext* /*ctx*/, const OpRequest* req, OpResponse* resp) override {
    std::cout << "[SERVER] ==> ExecuteOp start: "
              << req->op_name() << "/" << req->overload_name()
              << " args=" << req->arguments_size() << std::endl;

    torch::jit::Stack stack;
    for (int i = 0; i < req->arguments_size(); ++i) {
      auto iv = argToIValue(req->arguments(i));
      std::cout << "[SERVER]  - Arg " << i 
                << (iv.isTensor()    ? " Tensor" :
                   iv.isScalar()    ? " Scalar" :
                   iv.isList()      ? " List" : " Other")
                << std::endl;
      stack.push_back(iv);
    }

    const char* name = req->op_name().c_str();
    const char* over = req->overload_name().c_str();
    std::cout << "[SERVER] Dispatching " << name << "/" << over << std::endl;
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, over);
    try {
        at::native::cpu_fallback(op, &stack);
    //   op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), &stack);
    } catch (const std::exception& e) {
      std::cerr << "[SERVER][ERROR] Dispatch failed: " << e.what() << std::endl;
      return Status(grpc::StatusCode::UNKNOWN, "Dispatch error: " + std::string(e.what()));
    }

    auto out_iv = stack.back();
    std::cout << "[SERVER] Result type: "
              << (out_iv.isTensor() ? "Tensor" :
                  out_iv.isScalar() ? "Scalar" :
                  out_iv.isList()   ? "List" : "Other")
              << std::endl;
    *resp->mutable_result() = iValueToArg(out_iv);

    std::cout << "[SERVER] <== ExecuteOp done" << std::endl;
    return Status::OK;
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
