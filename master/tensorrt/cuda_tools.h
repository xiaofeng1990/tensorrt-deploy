#ifndef TENSORRT_CUDA_TOOLS_H_
#define TENSORRT_CUDA_TOOLS_H_

#include <cuda_runtime.h>

#define checkRuntime(call) cudatools::check_runtime(call, #call, __LINE__, __FILE__)

#define checkKernel(...)                                                                                               \
    __VA_ARGS__;                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                                                \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                                \
        }                                                                                                              \
    } while (0);

namespace cudatools {
bool check_runtime(cudaError_t e, const char *call, int line, const char *file);
}
#endif // TENSORRT_CUDA_TOOLS_H_
