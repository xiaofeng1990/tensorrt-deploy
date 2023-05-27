#ifndef XFFW_LOGGING_H_
#define XFFW_LOGGING_H_

#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>

#ifdef __cplusplus
extern "C" {
#endif

#define gettid() syscall(SYS_gettid)

#define _toString(str) #str
#define toString(str) _toString(str)
#define _conStr(a, b) (a##b)
#define conStr(a, b) _conStr(a, b)

typedef enum _XfSeverity {
    XF_DEBUG,
    XF_INFO,
    XF_WARN,
    XF_ERROR,
    XF_FATAL,
    XF_SILENT
} XfSeverity;

void xt_logging(int prio, const char *tag, const char *fmt, ...);

/**
 * @brief set min log level
 *
 * @param severity log level
 */
void XfSetMinLevel(XfSeverity severity);

/**
 * @brief get min log level
 *
 * @return XfSeverity
 */
XfSeverity XfGetMinLevel();

/**
 * @brief whether save log to file
 *
 * @param save save log to file if 1, do not save if 0
 */
void XfSaveToFile(int save);

void XfEnableProcessName();

/**
 * @brief set log path (default is "./")
 *
 * @param path save path
 * @return int 1 if success, otherwise fail
 */
int XfSetLogPath(const char *path);

void XfSetMaxFileNum(unsigned int num);

/**
 * @brief Create new file when file size greater than speficy value
 *
 * @param size file size, unit: M, range is [1 ~ 50]
 * @return int
 */
int XfSetFileSize(unsigned int size);

void XfSetFilePrefix(const char *prefix);

void XfSetLogToStderr(bool log_to_stderr);

#define IS_ON(severity) (severity >= XfGetMinLevel())

#define XF_LOGT(severity, tag, ...)                                            \
    !(IS_ON(XF_##severity))                                                    \
        ? (void)0                                                              \
        : xt_logging((int)(XF_##severity), tag, ##__VA_ARGS__);

#define XF_LOG(severity, ...)                                                  \
    !(IS_ON(XF_##severity))                                                    \
        ? (void)0                                                              \
        : xt_logging((int)(XF_##severity), "", ##__VA_ARGS__);

#define XF_ASSERT(condition)                                                   \
    if (!(condition))                                                          \
    {                                                                          \
        XF_LOG(FATAL, "%s:%d, assert(%s), fatal", __FILE__, __LINE__,          \
               #condition);                                                    \
        abort();                                                               \
    }

#ifdef __cplusplus
}
#endif
#endif // XF_LOGGING_H_