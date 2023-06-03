#include "cuda_tools.h"
#include "common/logging.h"

namespace cudatools {

bool check_runtime(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        XF_LOGT(ERROR, "cuda_tools",
                "CUDA Runtime error %s # %s, code = %s [%d] in file %s:%d",
                call, cudaGetErrorString(e), cudaGetErrorName(e), e, file,
                line);
        return false;
    }
    return true;
}
} // namespace cudatools
