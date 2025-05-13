#include <unistd.h>
#include <vector>


void kernel_add_wrapper(std::vector<void*> parents, float* C, size_t n);

void kernel_relu_wrapper(std::vector<void*> parents, float* C, size_t n);

void kernel_matmul_wrapper(std::vector<void*> parents, float* C, size_t n);