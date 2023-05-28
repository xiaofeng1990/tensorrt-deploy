#ifndef __MASTER_INFERENCE_INFER_INTERFACE_H_
#define __MASTER_INFERENCE_INFER_INTERFACE_H_
//尽量不暴露外部不需要的部分
class InferInterface {

  public:
    virtual void Forward() = 0;
};

#endif //__MASTER_INFERENCE_INFER_INTERFACE_H_