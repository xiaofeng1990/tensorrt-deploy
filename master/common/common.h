#ifndef XFLOG_COMMON_H_
#define XFLOG_COMMON_H_

#include <string>
#include <vector>
namespace xffw {

/**
 * @brief Get the Executable Path object
 *
 * @param processdir [out] executable file path
 * @param processname [out] executable filename
 * @return size_t   error if -1 otherwise filename length
 */
size_t GetExecutablePath(std::string &processdir, std::string &processname);

/**
 * @brief get file list for specify dir
 *
 * @param dir_name directory
 * @return std::vector<std::string> file list
 */
std::vector<std::string> FindFilesForDir(const char *dir_name);

} // namespace xffw

#endif // XFLOG_COMMON_H_