#include "opencv2/opencv.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace fs = std::filesystem;

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

struct DetResult {
    cv::Rect bbox;
    float conf;
    int label;
    DetResult(cv::Rect bbox,float conf,int label):bbox(bbox),conf(conf),label(label){}
};

class Yolov10 {
public:
    Yolov10(const std::string& engine_path) {
        context = createContext(engine_path);
    }

    void inferImage(const std::string& image_file);
    void inferVideo(const std::string& video_file);
    void inferFolder(const std::string& source_folder);
    void infer(const std::string& source_folder, const std::string& output_folder, float threshold);
    
private:
    void preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data);
    std::vector<DetResult> postProcess(float* result, float factor, int outputLength);
    void drawBbox(cv::Mat& img, std::vector<DetResult>& res);
    std::shared_ptr<nvinfer1::IExecutionContext> createContext(const std::string& model_path);
    std::shared_ptr<nvinfer1::IExecutionContext> context;
};

void Yolov10::preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data) {
    cv::Mat mat;
    int rh = img->rows;
    int rw = img->cols;
    int rc = img->channels();
    cv::cvtColor(*img, mat, cv::COLOR_BGR2RGB);
    int maxImageLength = rw > rh ? rw : rh;
    cv::Mat maxImage = cv::Mat::zeros(maxImageLength, maxImageLength, CV_8UC3);
    maxImage = maxImage * 255;
    cv::Rect roi(0, 0, rw, rh);
    mat.copyTo(cv::Mat(maxImage, roi));
    cv::Mat resizeImg;
    cv::resize(maxImage, resizeImg, cv::Size(length, length), 0.0f, 0.0f, cv::INTER_LINEAR);

    *factor = (float)((float)maxImageLength / (float)length);
    resizeImg.convertTo(resizeImg, CV_32FC3, 1 / 255.0);
    rh = resizeImg.rows;
    rw = resizeImg.cols;
    rc = resizeImg.channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(resizeImg, cv::Mat(rh, rw, CV_32FC1, data.data() + i * rh * rw), i);
    }
}

std::vector<DetResult> Yolov10::postProcess(float* result, float factor, int outputLength) {
    std::vector<cv::Rect> positionBoxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < outputLength; i++) {
        int s = 6 * i;
        // Due to that YOLOv10 adopts the different training strategy with others, e.g. YOLOv8, 
        // it may thus have different favorable confidence threshold to detect objects. 
        // Besides, different thresholds will have no impact on the inference latency of YOLOv10 because it does not rely on NMS. 
        // Therefore, we suggest that a smaller threshold can be freely set or tuned for YOLOv10 to detect smaller objects or objects in the distance.
        // ref: https://github.com/THU-MIG/yolov10/issues/136, set conf > 0.05 rather than 0.25 for samller objects
        if ((float)result[s + 4] > 0.05) {
            float cx = result[s + 0];
            float cy = result[s + 1];
            float dx = result[s + 2];
            float dy = result[s + 3];
            int x = (int)((cx)* factor);
            int y = (int)((cy)* factor);
            int width = (int)((dx - cx) * factor);
            int height = (int)((dy - cy) * factor);
            cv::Rect box(x, y, width, height);
            positionBoxes.push_back(box);
            classIds.push_back((int)result[s + 5]);
            confidences.push_back((float)result[s + 4]);
        }
    }
    std::vector<DetResult> re;
    for (int i = 0; i < positionBoxes.size(); i++) {
        DetResult det(positionBoxes[i], confidences[i], classIds[i]);
        re.push_back(det);
    }
    return re;
}

void Yolov10::drawBbox(cv::Mat& img, std::vector<DetResult>& res) {
    for (size_t j = 0; j < res.size(); j++) {
        cv::rectangle(img, res[j].bbox, cv::Scalar(255, 0, 255), 2);
        cv::putText(img, std::to_string(res[j].label) + "-" + std::to_string(res[j].conf), 
            cv::Point(res[j].bbox.x, res[j].bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 
            1.2, cv::Scalar(0, 0, 255), 2);
    }
}

std::shared_ptr<nvinfer1::IExecutionContext> Yolov10::createContext(const std::string& model_path) {
    std::ifstream filePtr(model_path, std::ios::binary);
    if (!filePtr.good()) {
        std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
        return nullptr;
    }
    size_t size = 0;
    filePtr.seekg(0, filePtr.end);
    size = filePtr.tellg();
    filePtr.seekg(0, filePtr.beg);
    char* modelStream = new char[size];
    filePtr.read(modelStream, size);
    filePtr.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream, size);
    return std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
}

void Yolov10::inferImage(const std::string& image_file) {
    cv::Mat image = cv::imread(image_file);
    if (image.empty()) {
        std::cout << "Image not found: " << image_file << std::endl;
        return;
    }
    float factor = 0;
    std::vector<float> inputData(640 * 640 * 3);
    preProcess(&image, 640, &factor, inputData);
    void* inputSrcDevice;
    void* outputSrcDevice;
    cudaMalloc(&inputSrcDevice, 3 * 640 * 640 * sizeof(float));
    cudaMalloc(&outputSrcDevice, 1 * 300 * 6 * sizeof(float));
    cudaMemcpy(inputSrcDevice, inputData.data(), 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice);
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    context->enqueue(1, (void**)bindings, nullptr, nullptr);
    std::vector<float> output_data(300 * 6);
    cudaMemcpy(output_data.data(), outputSrcDevice, 300 * 6 * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<DetResult> result = postProcess(output_data.data(), factor, 300);
    drawBbox(image, result);
    cv::imwrite("/lorenzo/deploy/LorenzoDeploy/result.jpg", image);
}

void Yolov10::inferVideo(const std::string& video_file) {
    cv::VideoCapture capture(video_file);
    if (!capture.isOpened()) {
        std::cerr << "ERROR: 视频无法打开" << std::endl;
        return;
    }
    cv::VideoWriter wri("./result.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30,
        cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH),capture.get(cv::CAP_PROP_FRAME_HEIGHT)), true);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* inputSrcDevice;
    void* outputSrcDevice;
    cudaMalloc(&inputSrcDevice, 3 * 640 * 640 * sizeof(float));
    cudaMalloc(&outputSrcDevice, 1 * 300 * 6 * sizeof(float));
    std::vector<float> output_data(300 * 6);
    std::vector<float> inputData(640 * 640 * 3);
    float total_infs[3];
    while (true) {
        cv::Mat frame;
        if (!capture.read(frame)) {
            break;
        }
        float factor = 0;
        auto t_beg = std::chrono::steady_clock::now();
        preProcess(&frame, 640, &factor, inputData);
        cudaMemcpyAsync(inputSrcDevice, inputData.data(), 3 * 640 * 640 * sizeof(float), 
            cudaMemcpyHostToDevice, stream);
        void* bindings[] = { inputSrcDevice, outputSrcDevice };
        auto t_end = std::chrono::steady_clock::now();
        total_infs[0] = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        std::cout << "The preprocess time is: " << total_infs[0] << "ms" << std::endl;
        t_beg = std::chrono::steady_clock::now();
        context->enqueue(1, (void**)bindings, stream, nullptr);
        t_end = std::chrono::steady_clock::now();
        total_infs[1] = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        std::cout << "The infer time is: " << total_infs[1] << "ms" << std::endl;
        t_beg = std::chrono::steady_clock::now();
        cudaMemcpyAsync(output_data.data(), outputSrcDevice, 300 * 6 * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::vector<DetResult> result = postProcess(output_data.data(), factor, 300);
        t_end = std::chrono::steady_clock::now();
        total_infs[2] = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        std::cout << "The postprocess time is: " << total_infs[2] << "ms" << std::endl;
        drawBbox(frame, result);
        std::string fps = "PreProcess: " + std::to_string(1000.0 / total_infs[0]) + "FPS\r\n"
            + "Inference: " + std::to_string(1000.0 / total_infs[1]) + "FPS\r\n"
            + "PostProcess: " + std::to_string(1000.0 / total_infs[3]) + "FPS\r\n"
            + "Total: " + std::to_string(1000.0 / (total_infs[0] + total_infs[1] + total_infs[2])) + "FPS\r\n";
        cv::putText(frame, "PreProcess: " + std::to_string(1000.0 / total_infs[0]) + "FPS",
            cv::Point(20, 40), 1, 2, cv::Scalar(255, 0, 255), 2);
        cv::putText(frame, "Inference: " + std::to_string(1000.0 / total_infs[1]) + "FPS",
            cv::Point(20, 70), 1, 2, cv::Scalar(255, 0, 255), 2);
        cv::putText(frame, "PostProcess: " + std::to_string(1000.0 / total_infs[2]) + "FPS",
            cv::Point(20, 100), 1, 2, cv::Scalar(255, 0, 255), 2);
        cv::putText(frame, "Total: " + std::to_string(1000.0 / (total_infs[0] + total_infs[1] + total_infs[2])) + "FPS",
            cv::Point(20, 130), 1, 2, cv::Scalar(255, 0, 255), 2);
        wri << frame;
    }
    wri.release();
    cudaFree(inputSrcDevice);
    cudaFree(outputSrcDevice);
    cudaStreamDestroy(stream);
}

void Yolov10::inferFolder(const std::string& source_folder) {
    for (const auto& entry : fs::directory_iterator(source_folder)) {
        if (fs::is_regular_file(entry.path())) {
            auto extension = fs::path(entry.path()).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
                inferImage(entry.path().string());
            } else if (extension == ".mp4" || extension == ".avi") {
                inferVideo(entry.path().string());
            }
        }
    }
}

void Yolov10::infer(const std::string& source_folder, const std::string& output_folder, float threshold) {
    if (!fs::exists(source_folder)) {
        std::cout << "Source folder not found: " << source_folder << std::endl;
        return;
    }
    fs::create_directories(output_folder);
    if (threshold < 1 && threshold > 0) {
        fs::path threshold_filter = fs::path(output_folder) / "threshold_filter";
        fs::create_directories(threshold_filter);
    }
    if (fs::is_directory(source_folder)) {
        inferFolder(source_folder);
    } else {
        auto extension = fs::path(source_folder).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
            inferImage(source_folder);
        } else if (extension == ".mp4" || extension == ".avi") {
            inferVideo(source_folder);
        }
    }
}

bool onnxToEngine(const std::string& onnxFile, int memorySize) {
    // 将路径作为参数传递给函数
    std::string path(onnxFile);
    std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
    std::string modelPath = path.substr(0, iPos); // 获取文件路径
    std::string modelName = path.substr(iPos, path.length() - iPos); // 获取带后缀的文件名
    std::string modelName_ = modelName.substr(0, modelName.rfind(".")); // 获取不带后缀的文件名
    std::string engineFile = modelPath + modelName_ + ".engine";

    // 构建器，获取cuda内核目录以获取最快的实现
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // 定义网络属性
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 解析onnx网络文件
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(), 2)) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    std::cout << "TensorRT loads mask onnx model successfully!" << std::endl;

    // 创建推理引擎
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to create engine." << std::endl;
        return false;
    }
    std::cout << "Try to save engine file now~~~" << std::endl;
    std::ofstream filePtr(engineFile, std::ios::binary);
    if (!filePtr) {
        std::cerr << "Could not open plan output file" << std::endl;
        return false;
    }
    nvinfer1::IHostMemory* modelStream = engine->serialize();
    filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    std::cout << "Convert onnx model to TensorRT engine model successfully!" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: main path/to/onnx/or/engine/model path/to/image/or/video" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string engine_path;

    if (model_path.substr(model_path.find_last_of(".") + 1) == "onnx") {
        engine_path = model_path.substr(0, model_path.find_last_of(".")) + ".engine";
        if (!fs::exists(engine_path)) {
            if (!onnxToEngine(model_path, 50)) {
                std::cerr << "Failed to convert ONNX to engine." << std::endl;
                return -1;
            }
            std::cout << "Engine file created: " << engine_path << std::endl;
        } else{
            std::cout << "Engine file already exists: " << engine_path << std::endl;
        }
    } else if (model_path.substr(model_path.find_last_of(".") + 1) == "engine") {
        engine_path = model_path;
        if (!fs::exists(engine_path)) {
            std::cerr << "Engine file not found: " << engine_path << " Please generate engine file first!" << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Unsupported model file format." << std::endl;
        return -1;
    }

    Yolov10 yolov10(engine_path);
    auto extension = fs::path(input_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
        yolov10.inferImage(input_path);
    } else if (extension == ".mp4" || extension == ".avi") {
        yolov10.inferVideo(input_path);
    } else {
        std::cerr << "Unsupported input file format." << std::endl;
        return -1;
    }

    return 0;
}
