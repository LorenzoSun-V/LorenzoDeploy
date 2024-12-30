/*
 * @Description: 集成视觉相关小功能模块
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-27 16:30:28
 */
#pragma once

#include "common.h"
#include "opencv2/opencv.hpp"

extern "C"
{
    struct Rotates
    {
        cv::Point2f center; // 旋转矩阵中心点X,Y
        cv::Size2f size; // 旋转矩阵宽高W,H
        cv::RotatedRect rotatedRect; // 传入(center, size, angle)

        float radian = 0.0; // 传入的弧度
        float angle = 0.0; // 计算出角度
        int width, height; // 图像宽高

        std::vector<cv::Point2f> polygon; // 多边形的四个顶点
        std::vector<cv::Point> pointsInPolygon; // 存储监测区域内旋转矩形的所有点

        std::vector<cv::Point> points; // 旋转矩阵所有坐标点
    };

    /*
     * @brief 图片处理接口
     *
	 * @param  pImagePath           输入图片地址 
	 * @param  frame                返回图片frame
     *
     * @return  bool                返回true成功
     */    
    bool ReadFrameFromPath(
        const char* pImagePath, 
        cv::Mat& frame
    );
                    
    /*
     * @brief 标记识别框，返回带框图像
     *	   
     * @param   frame                   输入检测图片/返回结果帧 
     * @param   detBoxs                 输入检测框结果       
     * 
     * @return  ENUM_ERROR_CODE         返回错误码
     */
	ENUM_ERROR_CODE DrawRectDetectResultForImage(
	    cv::Mat &frame,
        std::vector<DetBox> detBoxs
    );   

    /*
     * @brief 标记识别框，绘制旋转矩形框
     *	   
     * @param   frame                   输入检测图片/返回结果帧 
     * @param   detBoxs                 输入检测框结果       
     * 
     * @return  ENUM_ERROR_CODE         返回错误码
     */
    void DrawRotatedRectForImage(
        cv::Mat &image, 
        const std::vector<DetBox> detBoxs
    );

    /*
     * @brief 标记识别框，绘制实例分割结果
     *	   
     * @param   frame                   输入检测图片/返回结果帧 
     * @param   detBoxs                 输入实例分割 检测框和mask结果       
     * 
     * @return  ENUM_ERROR_CODE         返回错误码
     */
    ENUM_ERROR_CODE DrawInstanceSegmentResultForImage(
        cv::Mat &frame, 
        std::vector<SegBox> detBoxs,
        std::vector<cv::Mat> masks
    );

     /*
    * @brief 输出当前时间戳
    * @return  double         毫秒级
    */
    double GetCurrentTimeStampMS();    

    /*
     * @brief 检查IP地址是否有效
     *
	 * @param   ipaddr              输入字符串IP地址
     *
     * @return  bool                返回true成功
     */   
    int CheckValidIPAddress(
        const std::string& ipaddr
    ); 
    
    /*
     * @brief 提取IP地址存储到unsigned char数组中
     *
	 * @param   ipaddr              输入字符串IP地址
     * @param   iparray             输出字符数字
     *
     * @return  bool                返回true成功
     */ 
    int ExtractIPAddress(
        const std::string& ipaddr, 
        unsigned char ip[4]
    );

    /*
     * @brief 输入十六进制字符数组，
     * @brief 输出转换后的十六进制数据
     * @brief 输出十六进制的长度是输入nsize的一半
     * 
	 * @param   ipaddr              输入字符串IP地址
     * @param   iparray             输出字符数字
     *
     * @return  bool                返回true成功
     */ 
    void ConvertStringToHexArray(
        char *pDest, 
        const char *pSrc, 
        int nSize
    ); 

    /*
     * @brief 输入模型检测的图像和检测区域的坐标点，检测区域的长宽，
     * @brief 将检测区域的坐标转为与输入原始图像相同尺寸的坐标
     * 
	 * @param   image_width         输入检测图像的宽
     * @param   image_height        输入检测图像的高
     * @param   area_points         输入检测区域绘制的坐标点
     * @param   area_width          输入检测区域的图像宽
     * @param   area_height         输入检测区域的图像高
     * @param   converted_points    映射到图像相同尺寸的绘制检测区域的坐标
     * 
     * @return  bool                返回true成功
     */ 
    bool NormalizedPointsToImageSize(
        int image_width, 
        int image_height, 
        std::vector<cv::Point> area_points, 
        int area_width, 
        int area_height, 
        std::vector<cv::Point> &converted_points
    );

    /*
     * @brief 计算目标框与检测区域的百分比，判断是否大于阈值
     * 
	 * @param   detbox              输入识别物体目标的矩形坐标
     * @param   converted_points    输入映射后检测区域的坐标,需要输入多边形的每个点坐标
     * @param   threshold           输入过滤交集的百分比，输入（0~100）
     * 
     * @return  bool                返回true成功
     */ 
    bool CalculateAreaRatio(
        const cv::Rect detbox, 
        const std::vector<cv::Point> converted_points, 
        int threshold
    );

    /*
     * @brief  输入图像文件夹获得图像路径列表
     * 
	 * @param   InputFolder         输入图像文件夹路径
     * 
     * @return  vector              返回图像列表
     */ 
    std::vector<std::string> getImagePaths(
        const std::string& InputFolder
    );

    /*
     * @brief  替换图像后缀为txt后缀
     * 
	 * @param   path         输入图像路径
     * 
     * @return  string       返回替换后缀的路径
     */ 
    std::string replaceImageExtensionWithTxt(const std::string& path);

    /*
     * @brief  在图像名称后加上后缀
     * 
	 * @param   path         输入图像路径
     * @param   suffix_name  输入添加的后缀名称
     * 
     * @return  string       返回加上后缀后的路径
     */ 
    std::string replaceImageOutPath(const std::string& path, std::string suffix_name);

    /**
     * 获取旋转矩形内的所有像素点和交集部分的面积
     * @param Rotating 旋转矩形结构体
     * @return 交集部分占比包含旋转矩形在多边形内所有像素点的集合
     */
    float getPointsInRotatedRectorArea(Rotates& Rotating);

    /**
     * 绘制多边形和旋转矩形，并保存图像
     * @param Rotating 旋转矩形结构体
     * @param filename 保存图片的文件名
     */
    void drawAndSaveTemperatureMap(
        const Rotates& Rotating,
        const std::string& filename
    );

    /**
     * 获取旋转矩形在多边形内区域的最大温度
     * @param points 旋转矩形在多边形内的所有像素点
     * @param temp 存储温度的二维数组
     * @return 旋转矩形内的最大温度
     */
    float getMaxTemperature(
        const std::vector<cv::Point>& points,
        const float temp[384][288]
    );

}
