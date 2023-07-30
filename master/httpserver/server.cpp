#include "server.h"
#include "common/logging.h"

namespace xf {
HttpServer::HttpServer() { XF_LOGT(DEBUG, kTag, "%s\n", __FUNCTION__); }

bool HttpServer::Start(const std::string &ip, int32_t port)
{
    XF_LOGT(DEBUG, kTag, "%s\n", __FUNCTION__);
    ip_ = ip;
    port_ = port;
    return Init();
}
bool HttpServer::Stop() {}

bool HttpServer::Init()
{
    XF_LOGT(DEBUG, kTag, "%s start api server host:%s, port:%d", __FUNCTION__, ip_.c_str(), port_);
    http_server_.worker_processes = 0;
    http_server_.worker_threads = 4;
    http_server_.port = port_;
    Router::Register(http_service_);
    http_server_.service = &http_service_;
    auto ret = http_server_run(&http_server_, 0);
    if (ret == 0)
    {
        XF_LOGT(DEBUG, kTag, "%s http_server_run ok.\n", __FUNCTION__);
        return true;
    }
    else
    {
        XF_LOGT(ERROR, kTag, "%s http_server_run failed.\n", __FUNCTION__);
        return false;
    }
}
} // namespace xf
