/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 09:56:53
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-27 11:13:21
 * @Description:  YOLOv8OBB模型前处理、推理、后处理代码
 */
#ifndef YOLOV8OBBMODEL_H
#define YOLOV8OBBMODEL_H

#include "cuda.h"
#include "NvInfer.h"
#include "preprocess.h"
#include "postprocess.h"

using namespace nvinfer1;

// TensorRT日志记录器
// 继承自nvinfer1::ILogger，用于处理TensorRT的日志输出
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// TensorRT相关参数结构体
typedef struct {
    Logger gLogger;         // TensorRT日志记录器
    IRuntime* runtime;      // TensorRT运行时
    ICudaEngine* engine;    // TensorRT引擎
    IExecutionContext* context; // 执行上下文
    cudaStream_t stream;    // CUDA流
}trt_param_t;

// YOLOv8 OBB模型类
// 负责管理YOLOv8 OBB模型的加载、推理和资源管理
class YOLOV8OBBModel {
public:
    // 构造函数
    YOLOV8OBBModel();
    
    // 析构函数
    ~YOLOV8OBBModel();
    
    // 加载TensorRT引擎模型
    // @param engine_name: 引擎文件路径
    // @return: 成功返回true，失败返回false
    bool loadModel(const std::string engine_name);
    
    // 单张图像推理
    // @param frame: 输入图像
    // @param result: 输出检测结果
    // @return: 成功返回true，失败返回false
    bool inference(cv::Mat frame, std::vector<DetBox>& result);
    
    // 批量图像推理
    // @param batch_images: 输入图像列表
    // @param batch_result: 输出批量检测结果
    // @return: 成功返回true，失败返回false
    bool batch_inference(std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>>& batch_result);
private:
    // 模型参数
    int m_kOutputSize;  // 输出尺寸
    int m_kInputSize;   // 输入尺寸

    model_param_t m_model;  // 模型参数
    trt_param_t m_trt;      // TensorRT相关参数

    // 反序列化TensorRT引擎
    // @param engine_name: 引擎文件路径
    // @return: 成功返回true，失败返回false
    bool deserializeEngine(const std::string engine_name);

    // 执行推理
    // @param img_batch: 输入图像批次
    // @return: 成功返回true，失败返回false
    bool doInference(std::vector<cv::Mat> img_batch);
    
    int kMaxInputImageSize;  // 最大输入图像尺寸

    // 设备端内存指针
    float* inputSrcDevice;   // 输入数据设备指针
    float* outputSrcDevice;  // 输出数据设备指针
    
    std::vector<float> output_data;  // 输出数据
};

#endif // YOLOV10MODEL_H
