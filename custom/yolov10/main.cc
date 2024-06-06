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

void preProcess(cv::Mat *img, int length, float* factor, std::vector<float>& data) {
	cv::Mat mat;
    int rh = img->rows;
    int rw = img->cols;
    int rc = img->channels();
	cv::cvtColor(*img, mat, cv::COLOR_BGR2RGB);
    int maxImageLength = rw > rh ? rw : rh;
    cv::Mat maxImage = cv::Mat::zeros(maxImageLength, maxImageLength,CV_8UC3);
    maxImage = maxImage * 255;
    cv::Rect roi (0, 0, rw, rh);
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


std::vector<DetResult> postProcess(float* result, float factor, int outputLength) {
    std::vector<cv::Rect> positionBoxes;
    std::vector <int> classIds;
    std::vector <float> confidences;
    // Preprocessing output results
    for (int i = 0; i < outputLength; i++)
    {
        int s = 6 * i;
        if ((float)result[s + 4] > 0.2)
        {
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
    for (int i = 0; i < positionBoxes.size(); i++)
    {
        DetResult det(positionBoxes[i], confidences[i], classIds[i]);
        re.push_back(det);
    }
    return re;

}
void drawBbox(cv::Mat& img, std::vector<DetResult>& res) {
    for (size_t j = 0; j < res.size(); j++) {
        cv::rectangle(img, res[j].bbox, cv::Scalar(255, 0, 255), 2);
        cv::putText(img, std::to_string(res[j].label) + "-" + std::to_string(res[j].conf), 
            cv::Point(res[j].bbox.x, res[j].bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 
            1.2, cv::Scalar(0, 0, 255), 2);
    }
}

void onnxToEngine(const char* onnxFile, int memorySize) {

    // 将路径作为参数传递给函数
    std::string path(onnxFile);
    std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
    std::string modelPath = path.substr(0, iPos);//获取文件路径
    std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
    std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
    std::string engineFile = modelPath + modelName_ + ".engine";

    // 构建器，获取cuda内核目录以获取最快的实现
    // 用于创建config、network、engine的其他对象的核心类
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // 定义网络属性
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 解析onnx网络文件
    // tensorRT模型类
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // onnx文件解析类
    // 将onnx文件解析，并填充rensorRT网络结构
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    // 解析onnx文件
    parser->parseFromFile(onnxFile, 2);
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
    }
    printf("tensorRT load mask onnx model successfully!!!...\n");

    // 创建推理引擎
    // 创建生成器配置对象。
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // 设置最大工作空间大小。
    config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
    // 设置模型输出精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // 创建推理引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // 将推理银枪保存到本地
    std::cout << "Try to save engine file." << std::endl;
    std::ofstream filePtr(engineFile, std::ios::binary);
    if (!filePtr) {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    // 将模型转化为文件流数据
    nvinfer1::IHostMemory* modelStream = engine->serialize();
    // 将文件保存到本地
    filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    // 销毁创建的对象
    modelStream->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}

std::shared_ptr<nvinfer1::IExecutionContext> creatContext(std::string modelPath) {
    // 以二进制方式读取问价
    std::ifstream filePtr(modelPath, std::ios::binary);
    if (!filePtr.good()) {
        std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
        return std::shared_ptr<nvinfer1::IExecutionContext>();
    }

    size_t size = 0;
    filePtr.seekg(0, filePtr.end);	// 将读指针从文件末尾开始移动0个字节
    size = filePtr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
    filePtr.seekg(0, filePtr.beg);	// 将读指针从文件开头开始移动0个字节
    char* modelStream = new char[size];
    filePtr.read(modelStream, size);
    // 关闭文件
    filePtr.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream, size);
    return std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
}

void inferImage(const std::string& image_file) {
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
    std::shared_ptr<nvinfer1::IExecutionContext> context = creatContext("/lorenzo/deploy/LorenzoDeploy/weights/yolov10/yolov10s.engine");
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    context->enqueue(1, (void**)bindings, nullptr, nullptr);
    std::vector<float> output_data(300 * 6);
    cudaMemcpy(output_data.data(), outputSrcDevice, 300 * 6 * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<DetResult> result = postProcess(output_data.data(), factor, 300);
    drawBbox(image, result);
    cv::imwrite("/lorenzo/deploy/LorenzoDeploy/result.jpg", image);
}

void yolov10Infer() {
    const char* videoPath = "/lorenzo/deploy/LorenzoDeploy/test_files/30952-383991415_tiny.mp4";
    const char* enginePath = "/lorenzo/deploy/LorenzoDeploy/weights/yolov10/yolov10s.engine";

    std::shared_ptr<nvinfer1::IExecutionContext> context = creatContext(enginePath);
    cv::VideoCapture capture(videoPath);
    // 检查摄像头是否成功打开
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

    // std::chrono::time_point<std::chrono::steady_clock> t_beg;
    // std::chrono::time_point<std::chrono::steady_clock> t_end;
    float total_infs[3];
    while (true)
    {
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

    	// imshow("读取视频", frame);
        wri << frame;

        // cv::waitKey(10);	//延时30
    }
    wri.release();  //释放对象
    // cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2 ){
        std::cout << "Usage: main path/to/onnx/model, "
                     "e.g ./main ./yolov10s.onnx"
                  << std::endl;
        return -1;
    }

    const char* onnx_path = argv[1]; 
    // std::cout << onnx_path << std::endl;
    onnxToEngine(onnx_path, 50);
    // yolov10Infer();
	return 0;
}