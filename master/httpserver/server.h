#ifndef HTTPSERVER_SERVER_H_
#define HTTPSERVER_SERVER_H_
#include "HttpServer.h"
#include "HttpService.h"
#include "hv.h"
#include "router.h"
#include <string>
namespace xf {
class HttpServer {
  public:
    HttpServer();
    ~HttpServer() = default;
    bool Start(const std::string &ip, int32_t port);
    bool Stop();

  private:
    bool Init();

  private:
    std::string ip_;
    int port_;
    http_server_t http_server_;
    HttpService http_service_;
    static constexpr const char *kTag = "http_server";
};
} // namespace xf
#endif // HTTPSERVER_SERVER_H_
