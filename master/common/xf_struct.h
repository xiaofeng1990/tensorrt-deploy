#ifndef MASTER_COMMON_XF_STATUS_H_
#define MASTER_COMMON_XF_STATUS_H_

// XX(num, name, string)
#define XT_STATUS_MAP(XX)                                                                                              \
    XX(0, SUCCESS, 检测成功)                                                                                           \
    XX(1, MESSAGE_FOAMAT_ERROR, 消息解析失败)                                                                          \
    XX(2, NO_IAMGE, 图片不存在)                                                                                        \
    XX(3, IAMGE_FORMAT_ERROR, 图片格式错误)                                                                            \
    XX(4, INFERENE_ERROR, 推理失败)                                                                                    \
    XX(5, IMAGE_DECODE_ERROR, 图片转码失败)                                                                            \
    XX(6, IMAGE_TOO_BIG, 图片尺寸太大(最大4000x4000))                                                                  \
    XX(7, NO_FIND_OBJ, 图片中没有检测目标)

// XT_STATUS_##name
enum xt_status {
#define XX(num, name, string) XT_STATUS_##name = num,
    XT_STATUS_MAP(XX)
#undef XX
        XT_CUSTOM_STATUS
};

static const char *xt_status_str(enum xt_status status)
{
    switch (status)
    {
#define XX(num, name, string)                                                                                          \
    case XT_STATUS_##name:                                                                                             \
        return #string;
        XT_STATUS_MAP(XX)
#undef XX
    default:
        return "<unknown>";
    }
}

#endif // MASTER_COMMON_XF_STATUS_H_