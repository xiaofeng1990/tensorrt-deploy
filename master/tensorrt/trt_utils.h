#ifndef MASTER_TENSORRT_TRT_UTILS_H_
#define MASTER_TENSORRT_TRT_UTILS_H_
#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <numeric>
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

typedef struct _InferConfig {
    float conf_threshold;
    float iou_threshold;
    int max_batch_size;
    std::string model_path;
} InferConfig;

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

// template <typename T>
// std::unique_ptr<T> make_unique_ptr(T *ptr)
// {
//     if (!ptr)
//     {
//         throw std::runtime_error(std::string("Failed to create unique ptr"));
//     }
//     return std::unique_ptr<T>(ptr, [](T *ptr) { ptr->destroy(); });
// }

template <typename A, typename B>
inline A div_up(A x, B n)
{
    return (x + n - 1) / n;
}

inline int64_t volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline uint32_t get_element_size(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    return 0;
}

bool exists(const std::string filepath);
std::string join_dims(const std::vector<int> dims);
std::string dims_str(const nvinfer1::Dims dims);
std::string format(const char *fmt, ...);

// XX(num, name, string)
#define TRT_STATUS_MAP(XX)                                                                                             \
    XX(0, SUCCESS, 检测成功)                                                                                           \
    XX(1, MESSAGE_FOAMAT_ERROR, 消息解析失败)                                                                          \
    XX(2, NO_IAMGE, 图片不存在)                                                                                        \
    XX(3, IAMGE_FORMAT_ERROR, 图片格式错误)                                                                            \
    XX(4, INFERENE_ERROR, 推理失败)                                                                                    \
    XX(5, IMAGE_DECODE_ERROR, 图片转码失败)                                                                            \
    XX(6, IMAGE_TOO_BIG, 图片尺寸太大(最大4000x4000))                                                                  \
    XX(7, NO_FIND_OBJ, 图片中没有检测目标)

// XT_STATUS_##name
enum trt_status {
#define XX(num, name, string) TRT_STATUS_##name = num,
    TRT_STATUS_MAP(XX)
#undef XX
        TRT_CUSTOM_STATUS
};

static const char *trt_status_str(enum trt_status status)
{
    switch (status)
    {
#define XX(num, name, string)                                                                                          \
    case TRT_STATUS_##name:                                                                                            \
        return #string;
        TRT_STATUS_MAP(XX)
#undef XX
    default:
        return "<unknown>";
    }
}

} // namespace trt
#endif