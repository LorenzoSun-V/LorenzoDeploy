#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

//模型结果存放数据结构
struct DetBox {
    float x, y, h, w;//目标框左上角坐标x,y和框的长宽h,w
    float confidence;//预测精度
    int classID;//类别ID
    DetBox() {
        x = 0.0;
        y = 0.0;
        h = 0.0;
        w = 0.0;
        confidence = 0.0;
        classID = -1;
    }
    bool operator==(const DetBox& other) const {
        return x == other.x && y == other.y && h == other.h && w == other.w &&
               confidence == other.confidence && classID == other.classID;
    }
};     

typedef enum {
    /// 通用错误码定义:范围0x0000-0x00FF
    ///< 正确 
    ENUM_OK=0x0000,
    ///< 参数错误
    ERR_INVALID_PARAM=0x00001,
    ///< 内存不足
    ERR_NO_FREE_MEMORY=0x0002,
    ///< 输入句柄参数无效
    ERR_INPUT_INSTANCE_INVALID=0x0003,
    ///< 输入图像为空
    ERR_INPUT_IMAGE_EMPTY=0x0004,
    ///< 获得图像为空
    ERR_GET_IMAGE_EMPTY=0x0005,
    ///< 没有检测到目标
    ERR_DETECT_OBJECT_EMPTY=0x0006,
    ///< 预测失败   
    ERR_DETECT_PREDICT_FAIL=0x0007,	
    ///< 模型反序列化失败   
    ERR_MODEL_DESERIALIZE_FAIL=0x0008,
    ///< 输入模型路径不存在
    ERR_MODEL_INPUTPATH_NOT_EXIST=0x0009,

    ///推拉流相关，错误码定义:范围0x0100-0x01FF  
    ///< 输入检查IP地址输入范围有误
    ERR_INPUT_IP_INVALID=0x0100,
    ///<  创建管道失败
    ERR_MAKE_PIPE_FAILED=0x0101,
    ///<  创建管道失败
    ERR_MAKE_RTSP_SERVER_FAILED=0x0102,
    ///< 无法打开视频流
    ERR_CANNOT_OPEN_VIDEOSTREAM=0x0103,
    ///< 打开视频推流功能失败
    ERR_OPEN_PUSH_VIDEOSTREAM_FAILED=0x0104,
    ///< 视频推流失败
    ERR_PUSH_RTMPSTREAM_FAILED=0x0105,
    ///< 初始化视频保存功能失败
    ERR_INIT_SAVE_VIDEO_FAILED=0x0106,
    ///< 保存视频文件失败
    ERR_SAVE_VIDEO_FILE_FAILED=0x0107,
    ///< 解码初始化失败
    ERR_FFMEDIA_INIT_FAILED=0x0108,
    ///< 创建编码器失败
    ERR_CREATE_ENCODER_FAILED=0x0109,
    ///< 编码器加载参数打开失败
    ERR_ENCODER_OPEN_FAILED=0x010A,
    ///< 编码当前帧失败
    ERR_ENCODER_CURRENT_FRAME_FAILED=0x010B,
    ///< 打开写管道描述名失败
    ERR_OPEN_PIPE_NAME_FAILED=0x010C,
    ///< 拉流时间戳未更新
    ERR_RTSP_TIMESTAMP_NOT_UPDATE=0x0010D,
    
    /// 工业摄像头专用错误码定义:范围0x0200-0x02FF
    ///< 枚举设备失败，检查环境变量
    ERR_MVS_ENUM_IP_ADRESS=0x0200,
    ///< 输入IP在线设备列表未找到
    ERR_MVS_CANNOT_ONLINE_IP=0x0201,
    ///< 不支持设备类型
    ERR_MVS_UNSUPPORT_DEVICE_TYPE=0x0202,
    ///< 创建工业摄像头句柄失败
    ERR_MVS_CREATE_HANDLE_FAILED=0x0203,
    ///< 打开工业摄像头句柄失败
    ERR_MVS_OPEN_DEVICE_FAILED=0x0204,
    ///< 设置手动触发模式失败
    ERR_MVS_SET_TRIGGER_FAILED=0x0205,
    ///< 获取数据包大小失败
    ERR_MVS_GET_PAYLOADSIZE_FAILED=0x0206,
    ///< 输入修复分辨率binning参数错误
    ERR_MVS_SET_BINNING_NUMBER=0x0207,
    ///< 设置曝光参数错误
    ERR_MVS_SET_EXPOSURE_FAILED=0x0208,
    ///< 获取曝光参数错误
    ERR_MVS_GET_EXPOSURE_FAILED=0x0209,
    ///< 设置摄像头曝光时间失败
    ERR_MVS_SET_EXPOSURE_TIME_FAILED=0x020A,  
    ///< 设置自动增益失败
    ERR_MVS_SET_GAINAUTO_FAILED=0x020B,
    ///< 配置设置帧率失败
    ERR_MVS_SET_FRAMERATE_FAILED=0x020C,
    ///< 开始取流失败
    ERR_MVS_START_GET_STREAM_FAILED=0x020D,
    ///< 无效的摄像头bing参数
    ERR_MVS_INVALID_BINNING_PARAM=0x020E,
    ///< 设置bing参数失败
    ERR_MVS_SET_BINNING_FAILED=0x020F,
     ///< 获取视频帧超时
    ERR_MVS_GET_FRAME_TIMEOUT=0x0210,   
    ///< 转换MAT帧失败
    ERR_MVS_CONVERT_MAT_FAILED=0x0211,  
    ///< 设备链接断开
    ERR_MVS_DEVICE_DISCONNECT=0x0212,  
    ///< 停止取流失败
    ERR_MVS_STOP_GRABBING_FAILED=0x0213,  
    ///< 关闭设备失败
    ERR_MVS_CLOSE_DEVICE_FAILED=0x0214,  
    ///< 销毁句柄失败
    ERR_MVS_DESTORY_HANDLE_FAILED=0x0215,  
    ///< 缓存数据未准备好
    ERR_MVS_CACHE_FRAMES_NOT_READY=0x0216,
    ///< 设置RGB8像素格式失败
    ERR_MVS_SET_PIXRGB8FORMAT_FAILED=0x0217,
    ///< 获取RGB8像素格式失败
    ERR_MVS_GET_PIXRGB8FORMAT_FAILED=0x0218,

    /// 模型推理错误码定义:范围0x0300-0x03FF
    ///< 加载模型失败
    ERR_ROCKCHIP_LOAD_MODEL_FAILED=0x0300,  
    ///< rknn初始化失败
    ERR_ROCKCHIP_RKNN_INIT_FAILED=0x0301,  
    ///< 询问版本失败
    ERR_ROCKCHIP_QUERY_VERSION_FAILED=0x0302,  
    ///< 询问模型输入输出头失败
    ERR_ROCKCHIP_QUERY_IN_OUT_HEAD_FAILED=0x0303,  
    ///< 询问模型输入属性
    ERR_ROCKCHIP_QUERY_INPUT_ATTR_FAILED=0x0304,  
    ///< 模型不是uint8格式
    ERR_ROCKCHIP_NOT_UINT8_TYPE=0x0305,  
    ///< 运行返回为空
    ERR_ROCKCHIP_RUN_FAILED=0x0306,  
    ///< 运行结束获得结果返回为空
    ERR_ROCKCHIP_OUTPUT_GET_FAILED=0x0307,  
    ///< 加载模型失败
    ERR_NVIDIA_LOAD_MODEL_FAILED=0x0308,  
    /// 网络通信错误码定义:范围0x0400-0x04FF
    ///< 端口输入错误
    ERR_NETWORK_INPUT_PORT_ERROR=0x0400,  

     ///< 创建套结字错误
    ERR_NETWORK_CREATE_SOCKET_ERROR=0x0401,        
     ///< 端口绑定失败
    ERR_NETWORK_BINNING_PORT_FAILED=0x0402,    
     ///< 端口监听失败
    ERR_NETWORK_LISTEN_SOCKET_FAILED=0x0403,  
     ///< 接收客户端链接失败
    ERR_NETWORK_ACCEPT_CLIENT_FAILED=0x0404,    
     ///< 与客户端链接断开
    ERR_NETWORK_DISCONNECT_CLIENT=0x0405,   
     ///< 数据发送出错
    ERR_NETWORK_SENDDATA_ERROR=0x0406,     
     ///< 接收客户端数据失败
    ERR_NETWORK_RECVDATA_ERROR=0x0407, 
    ///< 输入服务器IP地址出错
    ERR_NETWORK_INPUT_SERVERIP_ERROR=0x0408, 
    ///< 设置服务器IP地址出错
    ERR_NETWORK_SETTING_SERVERIP_ERROR=0x0409, 
    ///< 设置非堵塞方式出错
    ERR_NETWORK_SETTING_NON_BLOCKING_ERROR=0x040A, 

    /// 串口通信错误码定义:范围0x0500-0x05FF
    ///< 输入串口名称为空
    ERR_SERICOM_INPUT_DEVICE_NAME_EMPTY=0x0500,  
    ///< 打开串口失败
    ERR_SERICOM_OPEN_DEVICE_NAME_FAILED=0x0501,   
    ///< 控制串口FCNTL功能失败
    ERR_SERICOM_FCNTL_FUNCTION_FAILED=0x0502,   
    ///< 串口为非终端设备
    ERR_SERICOM_UNTERMINAL_DEVICE=0x0503,   
    ///< 配置串口参数出错
    ERR_SERICOM_SET_CONFIG_ERROR=0x0504,   
    ///< 接收数据出错
    ERR_SERICOM_RECV_DATA_ERROR=0x0505,  
    ///< 发送数据出错
    ERR_SERICOM_SEND_DATA_ERROR=0x0506,  
    ///< 检查串口套接字出错
    ERR_SERICOM_FD_CHECK_ERROR=0x0507,  

    /// 读写配置文件错误码定义:范围0x0600-0x06FF
    ///< 名称为空
    ERR_PROFILE_INPUT_FILENAME_EMPTY=0x0600,  
    ///< 输入查询关键字为空
    ERR_PROFILE_SEARCH_KEYNAME_EMPTY=0x0601,  
    ///< 读配置文件错误
    ERR_PROFILE_READ_CONFIG_ERROR=0x0602,  
    ///< 未查询到输入的关键字
    ERR_PROFILE_NOT_FOUND_SEARCH_KEYNAME=0x0603,  
    ///< 添加关键字
    ERR_PROFILE_CONFIG_ADD_OPTION_ERROR=0x0604,   
    ///< 写入文件失败
    ERR_PROFILE_CONFIG_WRITE_ERROR=0x0605,     
    
    /// 红外相机错误码定义:范围0x0700-0x07FF
    ///< 初始化巨哥红外相机服务失败
    ERR_HWJG_CAMERA_SERVER_INIT_FAILED=0x0700,  
    ///< 初始化巨哥红外相机设备失败
    ERR_HWJG_CAMERA_DEVICE_INIT_FAILED=0x0701,  
    ///< 链接巨哥相机失败
    ERR_HWJG_CAMERA_CONTACT_FAILED=0x0702,  
    ///< 获取相机参数失败
    ERR_HWJG_GET_CAMERA_PARAM_FAILED=0x0703,  
    ///< 获取检测区域温度信息失败
    ERR_HWJG_GET_RECT_TEMPERATURE_FAILED=0x0704,
    ///< 红外相机启动失败
    ERR_HWJG_CAMERA_START_FAILED=0x0705,
    ///< 数据未准备好,等待时间戳更新，请继续尝试
    ERR_HWJG_WAITING_TIMESTAMP_UPDATE=0x0706,
    ///<  温度数据未准备好
    ERR_HWJG_TEMPERATURE_DATA_NOT_FAILED=0x0707,
    ///<  图像数据未准备好
    ERR_HWJG_IMAGE_DATA_NOT_FAILED=0x0708,
    ///<  巨歌摄像头断开
    ERR_HWJG_DEVICE_DISLINKED=0x0709,
    ///<  设备登入不成功
    ERR_HWIR_LOGIN_FAILED=0x070A,
    ///<  无效的区域检测点数量
    ERR_HWIR_INVALID_DETECT_POINT=0x070B,
    ///< 缓存温度数据未准备好
    ERR_HWIR_CACHE_TEMP_NOT_READY=0x070C,
}ENUM_ERROR_CODE;
