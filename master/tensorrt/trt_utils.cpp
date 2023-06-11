#include "trt_utils.h"

#include <assert.h>

#include <stdarg.h>
namespace trt {

std::string join_dims(const std::vector<int> dims)
{
    std::stringstream output;
    char buf[64];
    const char *fmts[] = {"%d", " x %d"};
    for (int i = 0; i < dims.size(); ++i)
    {
        snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
        output << buf;
    }
    return output.str();
}

std::string dims_str(const nvinfer1::Dims dims) { return join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims)); }
bool exists(const std::string filepath) { return access(filepath.c_str(), R_OK) == 0; }
std::string format(const char *fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    char buffer[10000];
    vsprintf(buffer, fmt, vl);
    return buffer;
}

} // namespace trt