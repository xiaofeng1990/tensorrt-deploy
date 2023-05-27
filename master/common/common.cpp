#include "common.h"
#include <dirent.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

namespace xffw {

size_t GetExecutablePath(std::string &processdir, std::string &processname)
{
    char tmp_process_dir[1024] = {0};
    if (readlink("/proc/self/exe", tmp_process_dir, sizeof(tmp_process_dir)) <=
        0)
        return -1;
    processdir.append(tmp_process_dir);
    char *path_end = strrchr(tmp_process_dir, '/');
    if (path_end == NULL)
        return -1;
    ++path_end;
    processname.append(path_end);
    return processname.size();
}

std::vector<std::string> FindFilesForDir(const char *dir_name)
{
    std::vector<std::string> v;
    DIR *dirp;
    struct dirent *dp;
    dirp = opendir(dir_name);
    while ((dp = readdir(dirp)) != NULL)
    {
        if (0 == strcmp(dp->d_name, ".") || 0 == strcmp(dp->d_name, ".."))
        {
            continue;
        }
        v.push_back(std::string(dp->d_name));
    }
    (void)closedir(dirp);
    return v;
}
} // namespace xffw