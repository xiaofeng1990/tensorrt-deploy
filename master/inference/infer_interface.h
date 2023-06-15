#ifndef __MASTER_INFERENCE_INFER_INTERFACE_H_
#define __MASTER_INFERENCE_INFER_INTERFACE_H_
//尽量不暴露外部不需要的部分
#include <future>
class InferInterface {

  public:
    virtual std::shared_future<std::string> Commits(std::string image) = 0;
};

#endif //__MASTER_INFERENCE_INFER_INTERFACE_H_