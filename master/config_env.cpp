#include "config_env.h"
#include "common/config_helper.h"
#include "common/logging.h"
#include <sys/stat.h>
#include <unistd.h>

namespace xf {

ConfigEnv::ConfigEnv() { XfSetMinLevel(XF_DEBUG); }
ConfigEnv::~ConfigEnv() {}
void ConfigEnv::Init()
{
    // set log
    XfSetFilePrefix("master_init");
    XfSetLogPath("./log/");
    XfSaveToFile(1);
    // get config file path and set config
    InitConfigPath();
    InitConfig();
}
void ConfigEnv::Uninit() {}

void ConfigEnv::InitConfig()
{

    // config log
    xffw::XfConfig *config = xffw::ConfigHelper::Ins();

    int max_num;
    config->Get("debug", "log_file_max_num", max_num);
    XfSetMaxFileNum(max_num);

    int file_size;
    config->Get("debug", "log_file_size", file_size);
    if (XfSetFileSize(file_size) != 0)
    {
        XF_LOGT(ERROR, kTag, "Set file size(%d) fail, use default value",
                file_size);
    }
    std::string prefix;
    config->Get("debug", "log_file_prefix", prefix);
    XfSetFilePrefix(prefix.c_str());

    int severity;
    config->Get("debug", "log_level", severity);
    XfSetMinLevel((XfSeverity)severity);
    XF_LOGT(DEBUG, kTag, "log level: %d", severity);

    int to_file;
    config->Get("debug", "save_log_to_file", to_file);

    std::string log_path;
    config->Get("debug", "log_file_path", log_path);
    int ret = XfSetLogPath(log_path.c_str());
    if (ret)
    {
        XF_LOGT(DEBUG, kTag, "save log to file: %d, path: %s", to_file,
                log_path.c_str());
    }
    else
    {
        XF_LOGT(ERROR, kTag,
                "save log to file: %d, log path(%s) do not exists, "
                "Save to default path",
                to_file, log_path.c_str());

        int force_to_file;
        config->Get("debug", "log_force_to_file", force_to_file);
        if (force_to_file)
        {
            XF_LOGT(ERROR, kTag, "log file path abnormal, pls check. fatal");
            XF_ASSERT(false);
        }
    }
    int to_stderr;
    config->Get("debug", "log_to_stderr", to_stderr);
    if (!to_stderr)
    {
        XF_LOGT(INFO, kTag, "log do not output to terminal.");
    }
    XfSetLogToStderr((bool)to_stderr);
}
void ConfigEnv::InitConfigPath()
{
    // Get config file path list
    std::string config_path;
    std::vector<std::string> paths = GetPaths();
    // 验证配置文件
    auto itr = paths.begin();
    for (; itr != paths.end(); itr++)
    {
        const std::string path = *itr;
        if (access(path.c_str(), F_OK | R_OK | W_OK) != 0)
        {
            XF_LOGT(ERROR, kTag,
                    "config file(%s) do not exist or no permission",
                    path.c_str());
            continue;
        }
        if (std::string(path).find_last_of(".ini") == std::string::npos)
        {
            continue;
        }
        config_path = path;
        XF_LOGT(INFO, kTag, "config file path: %s", config_path.c_str());
        break;
    }
    XF_ASSERT(!config_path.empty() && "do not find config file.");
    bool ret = xffw::ConfigHelper::Ins()->Init(config_path);
    XF_ASSERT(ret && "config system init fail");
    std::string magic;
    xffw::ConfigHelper::Ins()->Get("master", "magic", magic);
    XF_ASSERT(magic.compare(CONFIG_MAGIC) == 0 && "unknow config file");

    std::string version;
    xffw::ConfigHelper::Ins()->Get("master", "version", version);
    XF_LOGT(INFO, kTag, "config version: %s", version.c_str());
}
std::vector<std::string> ConfigEnv::GetPaths()
{
    // todo 从环境变量获取
    std::vector<std::string> paths;
    //从本地获取
    paths.emplace_back("./master/config/master_config.ini");
    paths.emplace_back("./config/master_config.ini");
    return std::move(paths);
}
// std::string ConfigEnv::GetNewConfigPath() {}
} // namespace xf