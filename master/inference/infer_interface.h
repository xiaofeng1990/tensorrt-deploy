#ifndef __MASTER_INFERENCE_INFER_INTERFACE_H_
#define __MASTER_INFERENCE_INFER_INTERFACE_H_
//尽量不暴露外部不需要的部分
#include <future>
#include <string>
#include <vector>
struct Box {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label)
    {
    }
};

class InferInterface {
  public:
    virtual std::shared_future<std::vector<Box>> Commits(const std::string &image_file) = 0;
};

#endif //__MASTER_INFERENCE_INFER_INTERFACE_H_