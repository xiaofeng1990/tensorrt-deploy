#include "inference.h"
#include "common/config_helper.h"
#include "common/logging.h"
#include "infer_interface.h"
#include "inference.h"

#include <future>
#include <mutex>
#include <queue>
#include <thread>
namespace xf {

class InferImpl : public InferInterface {

  public:
    virtual ~InferImpl();
    bool LoadModel(const std::string &file);
    virtual std::shared_future<std::string> Forward(std::string image) override;
    void Destroy();

    //推理消费
    void Worker(std::string file, std::promise<bool> &pro);

  private:
    //消费者
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::queue<Job> qjobs_;
    std::mutex lock_job_;
    std::condition_variable condition_;
    int max_batch_size_;
    static constexpr const char *TAG = "scheduler";
};
InferImpl::~InferImpl()
{
    running_ = false;
    condition_.notify_one();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

bool InferImpl::LoadModel(const std::string &file)
{
    //资源哪里分配，哪里释放，哪里使用
    std::promise<bool> pro;
    running_ = true;
    // todo 优化为线程池
    thread_ = std::thread(&InferImpl::Worker, this, file, std::ref(pro));

    return pro.get_future().get();
}

void InferImpl::Worker(std::string file, std::promise<bool> &pro)
{
    //模型加载
    std::string context = file;
    if (context.empty())
    {
        XF_LOGT(ERROR, TAG, "load model %s failed\n", file.c_str());
        pro.set_value(false);
        return;
    }
    else
    {
        XF_LOGT(INFO, TAG, "load model %s succeed\n", file.c_str());
        xffw::XfConfig *config = xffw::ConfigHelper::Ins();
        bool ret = config->Get("inference", "max_batch_size", max_batch_size_);
        if (!ret)
        {
            max_batch_size_ = DEFAULT_MAX_BATCH_SIZE;
        }
        pro.set_value(true);
    }
    //模型使用

    std::vector<Job> jobs;
    int batch_id = 0;
    while (running_)
    {
        std::unique_lock<std::mutex> lock(lock_job_);
        condition_.wait(lock, [&]() {
            // true return
            // false wait
            return !qjobs_.empty() || !running_;
        });
        if (!running_)
        {
            XF_LOGT(INFO, TAG, "stop inference thread\n");
            break;
        }

        // pop data into queue
        while (jobs.size() < max_batch_size_ && !qjobs_.empty())
        {
            jobs.push_back(qjobs_.front());
            qjobs_.pop();
        }
        // batch inference
        for (int i = 0; i < jobs.size(); i++)
        {
            auto &job = jobs[i];
            char result[100];
            sprintf(result, "%s: batch->%d[%d]", job.input.c_str(), batch_id,
                    jobs.size());
            job.pro->set_value(result);
        }
        batch_id++;
        jobs.clear();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        // if queue is empty, surrendering control of cpu
        // std::this_thread::yield();
    }
    XF_LOGT(INFO, TAG, "thread down\n");
    // delete model
    XF_LOGT(INFO, TAG, "release model %s\n", context.c_str());
    context.clear();
}

std::shared_future<std::string> InferImpl::Forward(std::string image)
{
    // push data into queue
    Job job;
    job.pro.reset(new std::promise<std::string>());
    job.input = image;
    std::lock_guard<std::mutex> lock(lock_job_);
    qjobs_.push(job);
    // detecton push
    // face push
    // feature push
    // detection face feature 并发执行
    condition_.notify_one();
    return job.pro->get_future();
    // XF_LOGT(INFO, TAG, "using %s inference\n", context_.c_str());
}

void InferImpl::Destroy() {}

// create instance and load model
std::shared_ptr<InferInterface> create_inference(const std::string &file)
{
    std::shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->LoadModel(file))
        instance.reset();
    return instance;
}

} // namespace xf
