#include "common.h"
#include "md5.h"
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
namespace xffw {

size_t GetExecutablePath(std::string &processdir, std::string &processname)
{
    char tmp_process_dir[1024] = {0};
    if (readlink("/proc/self/exe", tmp_process_dir, sizeof(tmp_process_dir)) <= 0)
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
bool IsDir(std::string path)
{
    struct stat buf;
    if (lstat(path.c_str(), &buf) < 0)
    {
        return false;
    }
    int ret = __S_IFDIR & buf.st_mode;
    if (ret)
    {
        return true;
    }
    return false;
}

std::string file_md5(const std::string &file)
{
    std::ifstream in(file.c_str(), std::ios::binary);
    if (!in)
    {
        return {};
    }

    xf::MD5 md5;
    std::streamsize length;
    char buffer[1024];
    while (!in.eof())
    {
        in.read(buffer, 1024);
        length = in.gcount();
        if (length > 0)
        {
            md5.update(buffer, length);
        }
    }
    in.close();
    return md5.toString();
}

// string path = "./my_directory/my_file.txt";
// ret: "./my_directory"
std::string GetDirectory(std::string path)
{
    if (IsDir(path))
    {
        return path;
    }

    std::string directory;
    const size_t last_slash_idx = path.rfind('/');
    if (std::string::npos != last_slash_idx)
    {
        directory = path.substr(0, last_slash_idx);
    }
    return directory;
}

// string path = "./my_directory/my_file.txt";
// ret: "my_file.txt"
std::string GetFileNameFromPath(std::string path)
{
    if (IsDir(path))
    {
        return "isdir";
    }

    std::string filename;
    const size_t last_slash_idx = path.rfind('/');
    if (std::string::npos != last_slash_idx)
    {
        filename = path.substr(last_slash_idx + 1);
    }
    else
    {
        // only filename
        filename = path;
    }
    return filename;
}

} // namespace xffw