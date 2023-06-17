#include "yolo_infer.h"
#include "common/common.h"
#include "common/config_helper.h"
#include "common/logging.h"
#include "infer_interface.h"
#include "tensorrt/builder.h"
#include "tensorrt/infer.h"
#include <future>
#include <mutex>
#include <queue>
#include <thread>
namespace xf {

class YoloImpl : public InferInterface {

  public:
    virtual ~YoloImpl();
    bool LoadModel(const std::string &onnx_file);
    virtual std::shared_future<std::string> Commits(std::string image) override;
    void Destroy();
    void Preprocess();
    void Postprocess();
    //推理消费
    void Worker(std::string onnx_file, std::promise<bool> &pro);

  private:
    //消费者
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::queue<Job> qjobs_;
    std::mutex lock_job_;
    std::condition_variable condition_;
    int max_batch_size_;
    trt::Infer infer_;
    static constexpr const char *TAG = "scheduler";
};
YoloImpl::~YoloImpl()
{
    running_ = false;
    condition_.notify_one();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

bool YoloImpl::LoadModel(const std::string &engine_file)
{
    //资源哪里分配，哪里释放，哪里使用
    std::promise<bool> pro;
    running_ = true;
    // todo 优化为线程池
    thread_ = std::thread(&YoloImpl::Worker, this, engine_file, std::ref(pro));

    return pro.get_future().get();
}

void YoloImpl::Worker(std::string engine_file, std::promise<bool> &pro)
{
    //模型加载
    bool ret = infer_.LoadModel(engine_file);

    if (!ret)
    {
        XF_LOGT(ERROR, TAG, "load model %s failed\n", engine_file.c_str());
        pro.set_value(false);
        return;
    }
    else
    {
        XF_LOGT(INFO, TAG, "load model %s succeed\n", engine_file.c_str());
        max_batch_size_ = infer_.GetMaxBatchSize();
        XF_LOGT(INFO, TAG, "max batch size %d\n", max_batch_size_);
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
        // set input host data
        std::vector<cv::Mat> images;
        for (int i = 0; i < jobs.size(); i++)
        {
            // load image
            images.push_back(jobs[i].image);
            // todo 前处理
        }

        infer_.Inference(images);

        // 后处理
        for (int i = 0; i < jobs.size(); i++)
        {
            auto &job = jobs[i];
            char result[100];
            // sprintf(result, "%s: batch->%d[%d]", job.input.c_str(), batch_id, jobs.size());
            //后处理
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
    XF_LOGT(INFO, TAG, "release model %s\n", engine_file.c_str());
    // context.clear();
}

std::shared_future<std::string> YoloImpl::Commits(std::string image)
{
    // push data into queue
    Job job;
    job.pro.reset(new std::promise<std::string>());
    // job.input = image;
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

void YoloImpl::Destroy() {}

// create instance and load model
std::shared_ptr<InferInterface> create_inference(trt::InferConfig infer_config)
{
    //通过onnx file build engine文件
    //生成engine文件名称
    auto folder = xffw::GetDirectory(infer_config.model_path);
    auto filename = xffw::GetFileNameFromPath(infer_config.model_path);
    std::string engine_name = folder + "/" + filename + ".engine";
    //生成engine文件
    trt::builder_engine(trt::FP16, infer_config.max_batch_size, infer_config.model_path, engine_name);

    std::shared_ptr<YoloImpl> instance(new YoloImpl());
    if (!instance->LoadModel(engine_name))
        instance.reset();
    return instance;
}

} // namespace xf
