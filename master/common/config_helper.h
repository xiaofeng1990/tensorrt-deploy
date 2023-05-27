#ifndef XFCONFIG_CONFIG_HELPER_H_
#define XFCONFIG_CONFIG_HELPER_H_
#include "config.h"
#include <string>

namespace xffw {
class ConfigHelper {
  public:
    static XfConfig *Ins()
    {
        static XfConfig instance;
        return &instance;
    }
    ~ConfigHelper() {}

  private:
    ConfigHelper() = default;
    ConfigHelper(const ConfigHelper &) = delete;
    ConfigHelper &operator=(const ConfigHelper &) = delete;
    static constexpr const char *TAG = "config_helper";
};
} // namespace xffw
#endif // XFCONFIG_CONFIG_HELPER_H_
