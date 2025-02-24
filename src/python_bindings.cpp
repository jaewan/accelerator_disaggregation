#include <pybind11/pybind11.h>
#include "include/remote_gpu_extension.h"

namespace py = pybind11;

PYBIND11_MODULE(remote_gpu_extension, m) {
    // Add your bindings here
    // For example:
    // m.def("some_function", &your_namespace::some_function);
    // py::class_<your_namespace::SomeClass>(m, "SomeClass")
    //     .def(py::init<>())
    //     .def("some_method", &your_namespace::SomeClass::some_method);
}
