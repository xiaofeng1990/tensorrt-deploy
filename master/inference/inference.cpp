#include "inference.h"
#include "common/logging.h"
#include "infer_interface.h"
#include "inference.h"
namespace xf {

class InferImpl : public InferInterface {

  public:
    bool LoadModel(const std::string &file);
    virtual void Forward() override;
    void Destroy();

  private:
    std::string context_;

    static constexpr const char *TAG = "scheduler";
};

bool InferImpl::LoadModel(const std::string &file)
{
    context_ = file;
    return true;
}
void InferImpl::Forward()
{
    XF_LOGT(INFO, TAG, "using %s inference\n", context_.c_str());
}

void InferImpl::Destroy() { context_.clear(); }

// create instance and load model
std::shared_ptr<InferInterface> create_inference(const std::string &file)
{
    std::shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->LoadModel(file))
        instance.reset();
    return instance;
}

} // namespace xf
