#ifndef MASTER_TENSORRT_TRT_UTILS_H_
#define MASTER_TENSORRT_TRT_UTILS_H_
#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
namespace trt {
constexpr long double operator"" _GiB(long double val) { return val * (1 << 30); }
constexpr long double operator"" _MiB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KiB(long double val) { return val * (1 << 10); }

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(unsigned long long val) { return val * (1 << 30); }
constexpr long long int operator"" _MiB(unsigned long long val) { return val * (1 << 20); }
constexpr long long int operator"" _KiB(unsigned long long val) { return val * (1 << 10); }

struct InferDeleter {
    template <typename T>
    void operator()(T *obj) const
    {
        delete obj;
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

template <typename T>
std::shared_ptr<T> make_shared_ptr(T *ptr)
{
    if (!ptr)
    {
        throw std::runtime_error(std::string("Failed to create shared ptr"));
    }
    return std::shared_ptr<T>(ptr, [](T *ptr) { ptr->destroy(); });
}

template <typename T>
std::unique_ptr<T> make_unique_ptr(T *ptr)
{
    if (!ptr)
    {
        throw std::runtime_error(std::string("Failed to create unique ptr"));
    }
    return std::unique_ptr<T>(ptr, [](T *ptr) { ptr->destroy(); });
}

bool exists(const std::string filepath);
std::string join_dims(const std::vector<int> dims);
std::string dims_str(const nvinfer1::Dims dims);
std::string format(const char *fmt, ...);
} // namespace trt
#endif