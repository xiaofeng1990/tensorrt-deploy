#ifndef MASTER_CONFIG_ENV_H_
#define MASTER_CONFIG_ENV_H_
#include <string>
#include <vector>
namespace xf {
class ConfigEnv final {
  public:
    ConfigEnv();
    ~ConfigEnv();
    // init env value
    void Init();
    void Uninit();

  private:
    void InitConfig();
    void InitConfigPath();
    std::vector<std::string> GetPaths();
    std::string GetNewConfigPath();
    static constexpr const char *CONFIG_MAGIC = "master_230318";
    static constexpr const char *kTag = "configenv";
};
} // namespace xf

#endif // MASTER_CONFIG_ENV_H_