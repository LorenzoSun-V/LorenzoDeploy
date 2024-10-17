/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 09:34:15
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-17 13:03:43
 * @Description: 特征提取代码
 */
#include "featextract.h"

FeatExtractModel::FeatExtractModel()
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(0),
      inputSrcDevice(nullptr), outputSrcDevice(nullptr) {
}

FeatExtractModel::~FeatExtractModel() {
    if (inputSrcDevice) {
        cudaFree(inputSrcDevice);
        inputSrcDevice = nullptr;
    }
    if (outputSrcDevice) {
        cudaFree(outputSrcDevice);
        outputSrcDevice = nullptr;
    }
    if (context) {
        cudaFreeHost(context);
        context = nullptr;
    }
    if (engine) {
        cudaFreeHost(engine);
        engine = nullptr;
    }
    if (runtime) {
        cudaFreeHost(runtime);
        runtime = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

bool FeatExtractModel::loadModel(const std::string engine_name){
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }

    auto inputDims = engine->getBindingDimensions(0);
    m_kBatchSize = inputDims.d[0];
    m_channel = inputDims.d[1];
    m_kInputH = inputDims.d[2];
    m_kInputW = inputDims.d[3];
    std::cout << "m_kBatchSize: " << m_kBatchSize << " m_channel: " << m_channel << " m_kInputH: " << m_kInputH << " m_kInputW: " << m_kInputW << std::endl;

    auto out_dims = engine->getBindingDimensions(1);
    m_featdim = out_dims.d[1];
    std::cout << "m_featdim: " << m_featdim << std::endl;

    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return false;
    }
    if (cudaMalloc(&inputSrcDevice, m_kBatchSize * m_channel * m_kInputH * m_kInputW * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input." << std::endl;
        return false;
    }
    if (cudaMalloc(&outputSrcDevice, m_kBatchSize * m_featdim * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output." << std::endl;
        return false;
    }

    inputData.resize(m_kBatchSize * m_channel * m_kInputH * m_kInputW);
    output_data.resize(m_kBatchSize * m_featdim);
    return true;
}

bool FeatExtractModel::deserializeEngine(const std::string engine_name){
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return false;
    }
    
    cudaSetDevice(0);
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    if (NULL == runtime) {
        std::cerr << "Failed to create Infer Runtime." << std::endl;
        goto FAILED;
    }
    engine = runtime->deserializeCudaEngine(serialized_engine, size);
    if (NULL == engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        goto FAILED;
    }   
    context = engine->createExecutionContext();
    if (NULL == context) {
        std::cerr << "Failed to create execution context." << std::endl;
        goto FAILED;
    }
    delete[] serialized_engine;
    return true;

FAILED: 
        delete[] serialized_engine;
        return false;
}

void FeatExtractModel::preProcess(std::vector<cv::Mat> img_batch, std::vector<float>& data) {
    int data_index = 0;
    // 创建一个缓存图像用于复制和调整大小操作
    cv::Mat resizeImg;
    // 遍历每张图像
    for (const auto &frame : img_batch) {
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        // 将方形图像调整为指定大小
        cv::resize(rgb_frame, resizeImg, cv::Size(m_kInputW, m_kInputH), 0, 0, cv::INTER_LINEAR);
        // 归一化图像像素值
        resizeImg.convertTo(resizeImg, CV_32FC3, 1.0 / 255.0);
        // 将图像的每个通道数据提取到data向量中
        std::vector<cv::Mat> channels(m_channel);
        cv::split(resizeImg, channels);
        for (int i = 0; i < m_channel; ++i) {
            std::memcpy(data.data() + data_index + i * m_kInputH * m_kInputW, channels[i].data, m_kInputH * m_kInputW * sizeof(float));
        }
        data_index += m_channel * m_kInputH * m_kInputW;
    }
}

bool FeatExtractModel::doInference(const std::vector<cv::Mat> img_batch, std::vector<std::vector<float>>& feats) {
    preProcess(img_batch, inputData);
    if (cudaMemcpyAsync(inputSrcDevice, inputData.data(), img_batch.size() * m_channel * m_kInputH * m_kInputW * sizeof(float), 
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "Failed to copy input data to device." << std::endl;
        return false;
    }
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    if (!context->enqueueV2(bindings, stream, nullptr)) {
        std::cerr << "Failed to enqueue inference." << std::endl;
        return false;
    }
    
    if (cudaMemcpyAsync(output_data.data(), outputSrcDevice, img_batch.size() * m_featdim * sizeof(float), 
            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "Failed to copy output data to host." << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    for (int i = 0; i < output_data.size(); i+=m_featdim) {
        std::vector<float> feat(m_featdim);
        std::memcpy(feat.data(), output_data.data() + i, m_featdim * sizeof(float));
        feats.emplace_back(feat);
    }

    return true;
}

bool FeatExtractModel::inference(cv::Mat frame, std::vector<float>& feat) {
    std::vector<cv::Mat> img_batch = {frame};
    std::vector<std::vector<float>> feats;
    if (!doInference(img_batch, feats)) {
        return false;
    }
    feat = feats[0];
    return true;
}

bool FeatExtractModel::batchInference(std::vector<cv::Mat> img_frames, std::vector<std::vector<float>>& feats) {
    if (img_frames.empty()) {
        std::cerr << "Input batch is empty." << std::endl;
        return false;
    }

    for (size_t i = 0; i < img_frames.size(); i += m_kBatchSize) {
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + m_kBatchSize && j < img_frames.size(); j++) {
            img_batch.emplace_back(img_frames[j]);
        }

        if (!doInference(img_batch, feats)) {
            return false;
        }
    }

    return true;
}