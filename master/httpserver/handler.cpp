#include "handler.h"

#include <chrono> // import std::chrono
#include <thread> // import std::thread

#include "EventLoop.h" // import setTimeout, setInterval
#include "common/logging.h"
#include "hbase.h"
#include "hfile.h"
#include "hstring.h"
#include "htime.h"
#include "HttpMessage.h"
#include "base64.h"
#include <opencv2/opencv.hpp>
#include"uri_cb.h"
#include "json.hpp"

int Handler::preprocessor(HttpRequest *req, HttpResponse *resp)
{
    // printf("%s:%d\n", req->client_addr.ip.c_str(), req->client_addr.port);
    // printf("%s\n", req->Dump(true, true).c_str());

#if REDIRECT_HTTP_TO_HTTPS
    // 301
    if (req->scheme == "http")
    {
        std::string location = hv::asprintf("https://%s:%d%s", req->host.c_str(), 8443, req->path.c_str());
        return resp->Redirect(location, HTTP_STATUS_MOVED_PERMANENTLY);
    }
#endif

    // Unified verification request Content-Type?
    // if (req->content_type != APPLICATION_JSON) {
    //     return response_status(resp, HTTP_STATUS_BAD_REQUEST);
    // }

    // Deserialize request body to json, form, etc.
    req->ParseBody();

    // Unified setting response Content-Type?
    resp->content_type = APPLICATION_JSON;

    return HTTP_STATUS_NEXT;
}

int Handler::postprocessor(HttpRequest *req, HttpResponse *resp)
{
    // printf("%s\n", resp->Dump(true, true).c_str());
    return resp->status_code;
}

int Handler::errorHandler(const HttpContextPtr &ctx)
{
    int error_code = ctx->response->status_code;
    return response_status(ctx, error_code);
}

int Handler::json(HttpRequest *req, HttpResponse *resp)
{
    if (req->content_type != APPLICATION_JSON)
    {
        return response_status(resp, HTTP_STATUS_BAD_REQUEST);
    }
    resp->content_type = APPLICATION_JSON;
    resp->json = req->GetJson();
    resp->json["int"] = 123;
    resp->json["float"] = 3.14;
    resp->json["string"] = "hello";
    return 200;
}

int Handler::Detector(HttpRequest *req, HttpResponse *resp)
{
    XF_LOGT(DEBUG, TAG, "%s\n", __FUNCTION__);
    if (req->content_type != APPLICATION_JSON)
    {
        return response_status(resp, HTTP_STATUS_BAD_REQUEST);
    }
    resp->content_type = APPLICATION_JSON;
    try
    {
        // 解析json
        XF_LOGT(DEBUG, TAG, "parse json\n");
        std::string image_data_base64 = req->json["imageData"];
        auto find = image_data_base64.find(",");
      if (find != std::string::npos) {
        image_data_base64 = image_data_base64.substr(find + 1);
      }
      std::string raw_data = hv::Base64Decode(image_data_base64.c_str());
        std::vector<unsigned char> cvdata(raw_data.c_str(), raw_data.c_str() + raw_data.size());
      cv::Mat img;
      img = cv::imdecode(cvdata, cv::IMREAD_COLOR);
      cv::imwrite("./test_cv.jpg", img);
    }
    catch (const std::exception &e)
    {
        std::cout << "receive data message: " << e.what() << std::endl;
        return 200;
    }

    if(UriCallback::Ins()->received_data_cb_)
    {
         auto result = UriCallback::Ins()->received_data_cb_("./test_cv.jpg");
         auto boxes = result.get();
    cv::Mat image = cv::imread("./test_cv.jpg");
    for (auto &box : boxes)
    {
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0),2); 
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8,
                    cv::Scalar(0, 0, 255), 2, 16);
    }
    std::string save_image_file = "image-draw.jpg";
    cv::imwrite(save_image_file, image);
    //  resp->json = hv::Json::parse();
    // hv::Json jroot;
    // resp->DumpBody();
    }
    return 200;
}

int Handler::recvLargeFile(const HttpContextPtr &ctx, http_parser_state state, const char *data, size_t size)
{
    // printf("recvLargeFile state=%d\n", (int)state);
    int status_code = HTTP_STATUS_UNFINISHED;
    HFile *file = (HFile *)ctx->userdata;
    switch (state)
    {
    case HP_HEADERS_COMPLETE:
    {
        if (ctx->is(MULTIPART_FORM_DATA))
        {
            // NOTE: You can use multipart_parser if you want to use multipart/form-data.
            ctx->close();
            return HTTP_STATUS_BAD_REQUEST;
        }
        std::string save_path = "html/uploads/";
        std::string filename = ctx->param("filename", "unnamed.txt");
        std::string filepath = save_path + filename;
        file = new HFile;
        if (file->open(filepath.c_str(), "wb") != 0)
        {
            ctx->close();
            return HTTP_STATUS_INTERNAL_SERVER_ERROR;
        }
        ctx->userdata = file;
    }
    break;
    case HP_BODY:
    {
        if (file && data && size)
        {
            if (file->write(data, size) != size)
            {
                ctx->close();
                return HTTP_STATUS_INTERNAL_SERVER_ERROR;
            }
        }
    }
    break;
    case HP_MESSAGE_COMPLETE:
    {
        status_code = HTTP_STATUS_OK;
        ctx->setContentType(APPLICATION_JSON);
        response_status(ctx, status_code);
        if (file)
        {
            delete file;
            ctx->userdata = NULL;
        }
    }
    break;
    case HP_ERROR:
    {
        if (file)
        {
            file->remove();
            delete file;
            ctx->userdata = NULL;
        }
    }
    break;
    default:
        break;
    }
    return status_code;
}

int Handler::sendLargeFile(const HttpContextPtr &ctx)
{
    std::thread([ctx]() {
        ctx->writer->Begin();
        std::string filepath = ctx->service->document_root + ctx->request->Path();
        HFile file;
        if (file.open(filepath.c_str(), "rb") != 0)
        {
            ctx->writer->WriteStatus(HTTP_STATUS_NOT_FOUND);
            ctx->writer->WriteHeader("Content-Type", "text/html");
            ctx->writer->WriteBody("<center><h1>404 Not Found</h1></center>");
            ctx->writer->End();
            return;
        }
        http_content_type content_type = CONTENT_TYPE_NONE;
        const char *suffix = hv_suffixname(filepath.c_str());
        if (suffix)
        {
            content_type = http_content_type_enum_by_suffix(suffix);
        }
        if (content_type == CONTENT_TYPE_NONE || content_type == CONTENT_TYPE_UNDEFINED)
        {
            content_type = APPLICATION_OCTET_STREAM;
        }
        size_t filesize = file.size();
        ctx->writer->WriteHeader("Content-Type", http_content_type_str(content_type));
#if USE_TRANSFER_ENCODING_CHUNKED
        ctx->writer->WriteHeader("Transfer-Encoding", "chunked");
#else
        ctx->writer->WriteHeader("Content-Length", filesize);
#endif
        ctx->writer->EndHeaders();

        char *buf = NULL;
        int len = 40960; // 40K
        SAFE_ALLOC(buf, len);
        size_t total_readbytes = 0;
        int last_progress = 0;
        int sleep_ms_per_send = 0;
        if (ctx->service->limit_rate <= 0)
        {
            // unlimited
        }
        else
        {
            sleep_ms_per_send = len * 1000 / 1024 / ctx->service->limit_rate;
        }
        if (sleep_ms_per_send == 0)
            sleep_ms_per_send = 1;
        int sleep_ms = sleep_ms_per_send;
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time;
        while (total_readbytes < filesize)
        {
            if (!ctx->writer->isConnected())
            {
                break;
            }
            if (!ctx->writer->isWriteComplete())
            {
                hv_delay(1);
                continue;
            }
            size_t readbytes = file.read(buf, len);
            if (readbytes <= 0)
            {
                // read file error!
                ctx->writer->close();
                break;
            }
            int nwrite = ctx->writer->WriteBody(buf, readbytes);
            if (nwrite < 0)
            {
                // disconnected!
                break;
            }
            total_readbytes += readbytes;
            int cur_progress = total_readbytes * 100 / filesize;
            if (cur_progress > last_progress)
            {
                // printf("<< %s progress: %ld/%ld = %d%%\n",
                //     ctx->request->path.c_str(), (long)total_readbytes, (long)filesize, (int)cur_progress);
                last_progress = cur_progress;
            }
            end_time += std::chrono::milliseconds(sleep_ms);
            std::this_thread::sleep_until(end_time);
        }
        ctx->writer->End();
        SAFE_FREE(buf);
        // auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        // printf("<< %s taked %ds\n", ctx->request->path.c_str(), (int)elapsed_time.count());
    })
        .detach();
    return HTTP_STATUS_UNFINISHED;
}
