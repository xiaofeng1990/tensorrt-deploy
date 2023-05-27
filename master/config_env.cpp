#include "config_env.h"
#include "config_helper.h"
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
    // set config
    InitConfigPath();
    InitConfig();
}
void ConfigEnv::Uninit() {}

void ConfigEnv::InitConfig() {}
void ConfigEnv::InitConfigPath()
{
    // Get config file path list

    // 验证配置文件
}
std::vector<std::string> ConfigEnv::GetPaths() {}
std::string ConfigEnv::GetNewConfigPath() {}
} // namespace xf