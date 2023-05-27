
#ifndef XFFW_CONFIG_H_
#define XFFW_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <glib.h>
#include <glib/gprintf.h>
#ifdef __cplusplus
};
#endif

#include "logging.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace std {
inline std::string to_string(std::string str) { return str; }
} // namespace std

namespace xffw {
class XfConfig {
  public:
    XfConfig() = default;
    ~XfConfig();
    bool Open(std::string config_file_path);
    bool SetConfigFilePath(std::string config_file_path);
    bool GetKeys(std::string group, std::vector<std::string> &keys) const;
    bool GetGroups(std::vector<std::string> &groups) const;
    // set a single value
    template <typename T>
    bool Set(const std::string group, const std::string key, T values);
    // set a list
    template <typename T>
    bool Set(const std::string group, const std::string key,
             std::vector<T> values);
    template <typename T>
    bool Get(const std::string group, const std::string key, T &value);
    template <typename T>
    bool Get(const std::string group, const std::string key,
             std::vector<T> &values);
    bool HasKey(std::string group, std::string key) const;
    bool HasGroup(std::string group) const;
    bool RemoveGroup(std::string group);
    bool RemoveKey(std::string group, std::string key);

  private:
    std::vector<std::string> _Split(std::string str, std::string part);
    bool SaveToFile();

  private:
    GKeyFile *config_key_file_ = nullptr;
    std::string config_file_path_;
    bool save_file_ = false;
    static constexpr const char *TAG = "Config";
};

template <typename T>
bool XfConfig::Get(const std::string group, const std::string key, T &value)
{
    gchar *gvalue;
    GError *error = nullptr;
    gvalue = g_key_file_get_value(config_key_file_, group.c_str(), key.c_str(),
                                  &error);
    if (error)
    {
        XF_LOGT(ERROR, TAG, "get group: %s keys: %s value error %s\n",
                group.c_str(), key.c_str(), error->message)
        g_error_free(error);
        error = NULL;
        return false;
    }
    std::istringstream is(gvalue);
    is >> value;
    g_free(gvalue);
    return true;
}
/**
 * @brief get values list from config file
 *
 * @tparam T int bool, string, double
 * @param group
 * @param key
 * @param values
 * @return true
 * @return false
 */
template <typename T>
bool XfConfig::Get(const std::string group, const std::string key,
                   std::vector<T> &values)
{
    gchar *gvalue;
    GError *error = nullptr;
    gvalue = g_key_file_get_value(config_key_file_, group.c_str(), key.c_str(),
                                  &error);
    if (error)
    {
        XF_LOGT(ERROR, TAG, "get group: %s keys: %s value error %s\n",
                group.c_str(), key.c_str(), error->message)
        g_error_free(error);
        error = NULL;
        return false;
    }

    std::vector<std::string> strs = _Split(gvalue, ";");
    g_free(gvalue);
    T value;
    std::istringstream is;
    for (auto &str : strs)
    {
        is.clear();
        is.str(str);
        is >> value;
        values.push_back(value);
    }

    return true;
}

template <typename T>
bool XfConfig::Set(const std::string group, const std::string key, T value)
{
    std::ostringstream os;
    os << value;
    g_key_file_set_value(config_key_file_, group.c_str(), key.c_str(),
                         os.str().c_str());
    // if (!save_file_)
    //     save_file_ = true;
    SaveToFile();

    return true;
}

template <typename T>
bool XfConfig::Set(const std::string group, const std::string key,
                   std::vector<T> values)
{
    std::string gvalues;
    for (auto &value : values)
        gvalues += std::to_string(value) + ";";

    g_key_file_set_value(config_key_file_, group.c_str(), key.c_str(),
                         gvalues.c_str());
    // if (!save_file_)
    //     save_file_ = true;
    SaveToFile();
    return true;
}
} // namespace xffw

#endif //__INI_CONFIG_H_
