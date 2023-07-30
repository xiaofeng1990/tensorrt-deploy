#include "router.h"

#include "handler.h"
#include "hasync.h" // import hv::async
#include "hthread.h"
#include "requests.h" // import requests::async

void Router::Register(hv::HttpService &router)
{
    // preprocessor => middleware -> handlers => postprocessor
    router.preprocessor = Handler::preprocessor;
    router.postprocessor = Handler::postprocessor;
    // router.errorHandler = Handler::errorHandler;
    // router.largeFileHandler = Handler::sendLargeFile;
    // curl -v http://ip:port/ping
    router.GET("/api/v1/ping", [](HttpRequest *req, HttpResponse *resp) { return resp->String("pong"); });
    // Content-Type: application/json
    // curl -v http://ip:port/json -H "Content-Type:application/json" -d '{"user":"admin","pswd":"123456"}'
    router.POST("/api/v1/json", Handler::json);

    router.POST("/api/v1/detector", Handler::Detector);
}
