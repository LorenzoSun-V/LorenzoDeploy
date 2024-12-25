/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 09:56:53
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-24 16:53:56
 * @Description: YOLOv8OBB模型前处理、推理、CUDA加速后处理代码
 */

#ifndef YOLOV8OBBMODEL_H
#define YOLOV8OBBMODEL_H

#include "cuda.h"
#include "NvInfer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "common.h" // Assuming DetBox is defined here
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Use the TensorRT namespace
using namespace nvinfer1;

// Logger class for TensorRT
class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Suppress info-level messages and log warnings and errors
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// Structure to hold TensorRT parameters
typedef struct {
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
} trt_param_t;

// YOLOV8OBBModel class definition
class YOLOV8OBBModel {
public:
    // Constructor
    YOLOV8OBBModel();

    // Destructor
    ~YOLOV8OBBModel();

    /**
     * @brief Loads the TensorRT engine from a serialized file.
     * 
     * @param engine_name Path to the serialized TensorRT engine file.
     * @return true if the model is loaded successfully.
     * @return false otherwise.
     */
    bool loadModel(const std::string engine_name);

    /**
     * @brief Performs inference on a single image and retrieves detection results.
     * 
     * @param frame Input image in OpenCV Mat format.
     * @param result Output vector to store detection results.
     * @return true if inference and post-processing are successful.
     * @return false otherwise.
     */
    bool inference(cv::Mat frame, std::vector<DetBox>& result);

    /**
     * @brief Performs batch inference on multiple images and retrieves detection results.
     * 
     * @param batch_images Vector of input images in OpenCV Mat format.
     * @param batch_result Output vector to store detection results for each image.
     * @return true if inference and post-processing are successful.
     * @return false otherwise.
     */
    bool batch_inference(std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>>& batch_result);

private:
    /**
     * @brief Deserializes the TensorRT engine from the provided file.
     * 
     * @param engine_name Path to the serialized TensorRT engine file.
     * @return true if deserialization is successful.
     * @return false otherwise.
     */
    bool deserializeEngine(const std::string engine_name);

    /**
     * @brief Performs inference on a batch of images and retrieves detection results using CUDA post-processing.
     * 
     * @param img_batch Vector of input images in OpenCV Mat format.
     * @param batch_result Output vector to store detection results for each image.
     * @return true if inference and post-processing are successful.
     * @return false otherwise.
     */
    bool doInference(std::vector<cv::Mat> img_batch, std::vector<std::vector<DetBox>>& batch_result);

    // Model and TensorRT parameters
    model_param_t m_model;      // Model parameters defined in postprocess.h
    trt_param_t m_trt;          // TensorRT parameters

    // Input and output device pointers
    float* inputSrcDevice;      // Device pointer for input data
    float* outputSrcDevice;     // Device pointer for raw output data

    // Host-side data buffers
    std::vector<float> inputData;    // Host buffer for input data
    std::vector<float> output_data;  // Host buffer for raw output data

    // Constants
    int m_kOutputSize;          // Size of the output buffer
    int m_kInputSize;           // Size of the input buffer
    int kMaxInputImageSize;     // Maximum input image size (e.g., 9000x9000)

    // Additional member variables for CUDA post-processing
    // (If any, currently none are needed beyond existing members)
};

#endif // YOLOV8OBBMODEL_H
