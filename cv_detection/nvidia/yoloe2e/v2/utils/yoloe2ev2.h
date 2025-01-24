/*
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/v2/utils/yoloe2ev2.h
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-07 16:08:16
 */
#ifndef YOLOE2EV2MODELMANAGER_H
#define YOLOE2EV2MODELMANAGER_H

#include "cuda.h"
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "logging.h"
#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace nvinfer1;


// YOLOE2Ev2模型管理器类
// 负责管理YOLOE2Ev2模型的加载、推理和资源管理
class YOLOE2Ev2ModelManager {
public:
    // 构造函数
    YOLOE2Ev2ModelManager();
    
    // 析构函数
    ~YOLOE2Ev2ModelManager();
    
    // 加载TensorRT引擎模型
    // @param engine_name: 引擎文件路径
    // @return: 成功返回true，失败返回false
    bool loadModel(const std::string engine_name);
    
    // 反序列化TensorRT引擎
    // @param engine_name: 引擎文件路径
    // @return: 成功返回true，失败返回false
    bool deserializeEngine(const std::string engine_name);
    
    // 单张图像推理
    // @param frame: 输入图像
    // @param detBoxs: 输出检测结果
    // @return: 成功返回true，失败返回false
    bool inference(cv::Mat frame, std::vector<DetBox>& detBoxs);
    
    // 批量图像推理
    // @param frames: 输入图像列表
    // @param batchBoxes: 输出批量检测结果
    // @return: 成功返回true，失败返回false
    bool batchinference(std::vector<cv::Mat> frames, std::vector<std::vector<DetBox>>& batchBoxes);

private:
    // 模型参数
    size_t  m_kBatchSize;  // 批处理大小
    int m_channel;         // 输入图像通道数
    int m_kInputH;         // 输入图像高度
    int m_kInputW;         // 输入图像宽度
    int m_maxObject;       // 最大检测目标数

    // 图像预处理
    // @param img: 输入图像
    // @param data: 输出预处理后的数据
    void preprocess(cv::Mat& img, std::vector<float>& data);
    
    // 执行推理
    // @return: 成功返回true，失败返回false
    bool doInference();
    
    // 将预测框回归到原始图像尺寸
    // @param pred_box: 预测框
    // @param detBoxs: 输出回归后的检测框
    // @param width: 原始图像宽度
    // @param height: 原始图像高度
    void rescale_box(std::vector<DetBox>& pred_box, std::vector<DetBox>& detBoxs, int width, int height);
    // TensorRT相关资源
    IRuntime* runtime;     // TensorRT运行时
    ICudaEngine* engine;   // TensorRT引擎
    IExecutionContext* context; // 执行上下文
    cudaStream_t stream;   // CUDA流

    // 输入输出数据
    std::vector<float> input;  // 输入数据
    int* host_output0;         // 输出0：检测框数量
    float* host_output1;       // 输出1：检测框坐标
    float* host_output2;       // 输出2：检测框置信度
    float* host_output3;       // 输出3：检测框类别
    void* buffers[5];          // 设备端缓冲区

};

#endif // YOLOE2EV2MODEL_H
