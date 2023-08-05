#ifndef MASTER_HTTPSERVER_HTTP_URI_CB_H_
#define MASTER_HTTPSERVER_HTTP_URI_CB_H_

#include <functional>
#include <map>
#include <string>
#include "inference/infer_interface.h"
#include <future>
#include <memory>

typedef std::function<std::shared_future<std::vector<Box>>(const std::string &)> received_data_cb_t;

class UriCallback {
  public:
    static UriCallback *Ins()
    {
        static UriCallback uri_cb;
        return &uri_cb;
    }
    received_data_cb_t received_data_cb_;

  private:
    UriCallback() = default;
    UriCallback(const UriCallback &) = delete;
    UriCallback &operator=(const UriCallback &) = delete;
};

#endif // MASTER_HTTPSERVER_HTTP_URI_CB_H_
