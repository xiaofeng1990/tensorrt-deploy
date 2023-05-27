#include "logging.h"
#include "common.h"

#include <algorithm>
#include <errno.h>
#include <iostream>
#include <mutex>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <vector>

static constexpr const int kDefault1024 = 1024;
static constexpr const int kDefault256 = 256;

static XfSeverity g_min_log_level = XF_DEBUG;
static bool g_is_save_to_file = true;

static bool g_log_to_stderr = true;
static bool g_use_process_name_for_file = true;

static const size_t kMaxLogMessageLen = 2048;
static const size_t K1M = 1 * 1024 * 1024;     // 1M
static const size_t kMaxLogFileSize = 1 * K1M; // 1M
static constexpr const int kFilePathSize = 256;
static char g_save_path[kFilePathSize] = "./log";
static bool g_save_path_updated = false;

static constexpr const int kFilePrefixSize = 256;
static char g_file_prefix[kFilePrefixSize] = {0};
static bool g_file_prefix_updated = false;

static FILE *g_file = NULL;
static std::string g_log_file_name;
static std::string g_full_path;
static unsigned int g_log_file_max_num = 10;
static size_t g_log_file_size = kMaxLogFileSize;
static std::vector<std::string> g_log_file_vec;
static std::mutex g_write_mutex_;
static size_t g_log_file_idx = 0;

const char *LogArray[] = {
    "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "SILENT",
};

static std::string generate_full_path(std::string path, std::string file_name)
{
    std::string full_path;
    auto pos = path.find_last_of("/");
    if (pos != std::string::npos && (pos + 1) == path.size())
    {
        full_path = std::string(g_save_path) + file_name;
    }
    else
    {
        full_path = std::string(g_save_path) + "/" + file_name;
    }
    return full_path;
}
static bool check_and_delete_log_file()
{

    if (g_log_file_vec.size() >= g_log_file_max_num)
    {
        // delete old file
        auto it = g_log_file_vec.begin();
        if (it == g_log_file_vec.end())
        {
            fprintf(stderr, "No log file\n");
            return false;
        }
        std::string file_name((*it));
        std::string path =
            generate_full_path(std::string(g_save_path), file_name);
        g_log_file_vec.erase(it);
        if (access(path.c_str(), F_OK) != 0)
        {
            fprintf(stderr,
                    "ERROR: log file(%s) do not exists, can not deleted\n",
                    (*it).c_str());
            return false;
        }
        if (remove(path.c_str()) != 0)
        {
            fprintf(stderr, "ERROR: delete file(%s) fail\n", (*it).c_str());
            return false;
        }
    }
    return true;
}
static size_t now_micros()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<size_t>((tv.tv_sec) * 1000000 + tv.tv_usec);
}

static int MkDir(const char *path)
{
    // check the path
    if (access(path, F_OK) != 0)
    {
        if (mkdir(path, 0777) == -1)
        {
            fprintf(stderr, "DEBUG: create path [%s] fail [%s]\n", path,
                    strerror(errno));
            return -1;
        }
        else
        {
            fprintf(stderr, "DEBUG: log path [%s] do not exists, create it\n",
                    path);
        }
        if (chmod(path, 0777) == -1)
        {
            fprintf(stderr,
                    "DEBUG: modify permission fail [%s] for path [%s]\n", path,
                    strerror(errno));
            return -1;
        }
    }
    return 0;
}
static std::string log_file_prefix()
{
    static std::string file_prefix;
    if (!file_prefix.empty() && !g_file_prefix_updated)
    {
        return file_prefix;
    }
    char file_name[kDefault256] = {0};
    if (g_use_process_name_for_file)
    {
        std::string exec_path;
        std::string exec_name;

        if (xffw::GetExecutablePath(exec_path, exec_name) == -1)
        {
            fprintf(stderr, "ERROR: get process name fail\n");
        }
        snprintf(file_name, sizeof(file_name), "%slog_%s", g_file_prefix,
                 exec_name.c_str());
    }
    else
    {
        snprintf(file_name, sizeof(file_name), "%slog_%d", g_file_prefix,
                 getpid());
    }
    file_prefix = std::move(file_name);
    return file_prefix;
}

// file format:
//   log_[pid/processname]_timestamp_id.log
// eg:
//  log_522_202008291211_1.log
// or:
//  log_[processname]_202008291211_1.log
static std::string create_log_file(size_t micos)
{
    struct tm local_time;
    time_t seconds = static_cast<time_t>(micos / 1000000);
    localtime_r(&seconds, &local_time);
    char buffer[30] = {0};
    strftime(buffer, sizeof(buffer), "%Y%m%d%H%M", &local_time);
    char file_name[kDefault256] = {0};
    const std::string prefix = log_file_prefix();
    snprintf(file_name, sizeof(file_name), "%s_%s_%lu.txt", prefix.c_str(),
             buffer, g_log_file_idx);
    g_log_file_idx++;
    g_log_file_vec.emplace_back(file_name);
    return file_name;
}

static void write_to_file(const char *data, size_t len, size_t micros)
{
    if (access(g_save_path, F_OK | W_OK) != 0)
    {
        if (MkDir(g_save_path) == -1)
        {
            fprintf(stderr,
                    "ERROR: log path [%s] do not exist or is not writable\n",
                    g_save_path);
            return;
        }
    }
    if (g_log_file_max_num <= 0)
    {
        return;
    }

    // if (g_use_process_name_for_file &&
    //     (g_file_prefix_updated || g_log_file_vec.size() == 0))
    if (g_file_prefix_updated || (g_log_file_vec.size() == 0))
    {
        std::cout << "***************" << g_log_file_vec.size() << std::endl;
        // todo log_file_prerfix
        const auto prefix = log_file_prefix();
        auto files = xffw::FindFilesForDir(g_save_path);
        for (auto &file : files)
        {
            auto pos = file.find(prefix);
            if (0 != pos)
                continue;
            g_log_file_vec.emplace_back(file);
        }
        if (g_log_file_vec.size() != 0)
        {
            std::sort(g_log_file_vec.begin(), g_log_file_vec.end());

            const auto it = g_log_file_vec.back();
            // prese id of log filename
            // format(@see create_log_file()):
            // log_[pid/processname]_timestamp_id.log
            auto pos1 = it.find_last_of("_");
            const auto pos2 = it.find_last_of(".");
            if (std::string::npos != pos1 && std::string::npos != pos2 &&
                pos2 > pos1)
            {
                pos1 = pos1 + 1;
                const auto substr = it.substr(pos1, pos2 - pos1);
                const auto digit = stoi(substr);
                printf("pos1: %ld, pos2: %ld, %s == %d <- %s\n", pos1, pos2,
                       substr.c_str(), digit, it.c_str());
                // modify value
                g_log_file_idx = digit + 1;
            }
        }
    }
    // Get log file name, empty if firstly
    if (g_log_file_name.empty() || g_full_path.empty() ||
        g_file_prefix_updated || g_save_path_updated)
    {
        g_log_file_name = create_log_file(micros);
        g_full_path =
            generate_full_path(std::string(g_save_path), g_log_file_name);
        fprintf(stderr, "log file: %s\n", g_full_path.c_str());
        g_file_prefix_updated = false;
        g_save_path_updated = false;
    }
    // check file size
    {
        struct stat buf;
        int result = stat(g_full_path.c_str(), &buf);
        if (result != 0)
        {
            fprintf(stderr, "ERROR: file(%s) stat fail\n", g_full_path.c_str());
        }
        else
        {
            check_and_delete_log_file();
            if ((size_t)buf.st_size > g_log_file_size)
            {
                // Release old log file handle
                fclose(g_file);
                g_file = NULL;
                // g_log_file_idx = 0;

                // New log file
                g_log_file_name = create_log_file(micros);
                g_full_path = generate_full_path(std::string(g_save_path),
                                                 g_log_file_name);
                fprintf(stderr, "New log file: %s, old size: %lu\n",
                        g_full_path.c_str(), buf.st_size);
            }
        }
    }
    if (access(g_full_path.c_str(), F_OK) != 0 && NULL != g_file)
    {
        fclose(g_file);
        g_file = NULL;
    }

    if (NULL == g_file)
    {
        g_file = fopen(g_full_path.c_str(), "a");
        if (NULL == g_file)
        {
            // printf("Could not create log file\n");
            fprintf(stderr, "Could not create log file(%s)\n",
                    g_full_path.c_str());
            return;
        }
        const int fd = fileno(g_file);
        if (-1 == fchmod(fd, 0666))
        {
            fprintf(stderr, "modify permission fail(%s)\n",
                    g_full_path.c_str());
        }
    }

    std::string data_str(data);
    fwrite(data, 1, data_str.size(), g_file);
    fflush(g_file);
}

static void internal_logging(int prio, const char *tag, const char *buf)
{
    size_t micros = now_micros();
    time_t seconds = static_cast<time_t>(micros / 1000000);
    int remainder = static_cast<int>(micros % 1000000);
    char time_format[30] = {0};
    struct tm local_time;
    localtime_r(&seconds, &local_time);
    strftime(time_format, sizeof(time_format), "%m-%d %H:%M:%S", &local_time);
    std::string buf_str(buf);
    size_t n = buf_str.find_last_not_of("\n");
    if (std::string::npos != n)
    {
        buf_str.erase(n + 1, buf_str.size() - n);
    }
    char log_data[kMaxLogMessageLen] = {0};
    snprintf(log_data, sizeof(log_data), "[%s] %s.%06d %d %ld /%s] %s\n",
             LogArray[prio], time_format, remainder, getpid(), gettid(), tag,
             buf_str.c_str());

    if (g_log_to_stderr)
    {
        // std::string log_data_str(log_data);
        fwrite(log_data, 1, strlen(log_data), stderr);
    }
    if (g_is_save_to_file)
    {
        std::unique_lock<std::mutex> lock(g_write_mutex_);
        // todo  write_to_file
        write_to_file(log_data, strlen(log_data), micros);
    }
    if (XF_FATAL == prio)
    {
        std::string msg("fatal, abort. >>>>>>>>\n");
        fwrite(msg.c_str(), 1, msg.size(), stderr);
        write_to_file(msg.c_str(), msg.size(), micros);
        abort();
    }
}

void xt_logging(int prio, const char *tag, const char *fmt, ...)
{
    va_list ap;
    char buf[1024];
    va_start(ap, fmt);
    vsnprintf(buf, 1024, fmt, ap);
    va_end(ap);
    internal_logging(prio, tag, buf);
}

void XfSetMinLevel(XfSeverity severity) { g_min_log_level = severity; }

XfSeverity XfGetMinLevel() { return g_min_log_level; }

void XfSaveToFile(bool save) { g_is_save_to_file = save; }

void XfEnableProcessName() { g_use_process_name_for_file = true; }

int XfSetLogPath(const char *path)
{
    if (path == NULL)
    {
        return -1;
    }
    if (MkDir(path) == -1)
    {
        return -1;
    }
    if (access(path, W_OK) != 0)
    {
        fprintf(stderr, "ERROR: file path [%s] is not writable\n", path);
        return -1;
    }
    if (strcmp(g_save_path, path) != 0)
    {
        strcpy(g_save_path, path);
        g_save_path_updated = true;
    }
    return 0;
}

void XfSetMaxFileNum(unsigned int num) { g_log_file_max_num = num; }

int XfSetFileSize(unsigned int size)
{
    if (size < 1 || size > 50)
    {
        fprintf(stderr, "DEBUG: File size [%d] is out of range([1 ~ 50]) M \n",
                size);
        return -1;
    }

    g_log_file_size = size * K1M;
    return 0;
}

void XfSetFilePrefix(const char *prefix)
{
    if (NULL == prefix)
    {
        return;
    }

    if (0 != strcmp(g_file_prefix, prefix))
    {
        strcpy(g_file_prefix, prefix);
        g_file_prefix_updated = true;
    }
}

void XfSetLogToStderr(bool log_to_stderr) { g_log_to_stderr = log_to_stderr; }
