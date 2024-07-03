/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 14:19:07
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-03 09:46:53
 * @Description: YOLOv10模型前处理、推理、后处理代码
 */
#include "yolov10.h"

YOLOV10ModelManager::YOLOV10ModelManager()
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(0),
      inputSrcDevice(nullptr), outputSrcDevice(nullptr) {
}

YOLOV10ModelManager::~YOLOV10ModelManager() {
    if (inputSrcDevice) cudaFree(inputSrcDevice);
    if (outputSrcDevice) cudaFree(outputSrcDevice);

    if (context) cudaFreeHost(context);
    if (engine) cudaFreeHost(engine);
    if (runtime) cudaFreeHost(runtime);
    if (stream) cudaStreamDestroy(stream);
}

// Load the model from the serialized engine file
bool YOLOV10ModelManager::loadModel(const std::string engine_name) {
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }
    
    const int inputIndex = engine->getBindingIndex("images");
    auto inputDims = engine->getBindingDimensions(inputIndex);
    m_kBatchSize = inputDims.d[0];
    m_channel = inputDims.d[1];
    m_kInputH = inputDims.d[2];
    m_kInputW = inputDims.d[3];
    std::cout << "m_kBatchSize: " << m_kBatchSize << " m_channel: " << m_channel << " m_kInputH: " << m_kInputH << " m_kInputW: " << m_kInputW << std::endl;

    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return false;
    }
    if (cudaMalloc(&inputSrcDevice, m_kBatchSize * m_channel * m_kInputH * m_kInputW * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input." << std::endl;
        return false;
    }
    if (cudaMalloc(&outputSrcDevice, m_kBatchSize * kOutputSize * 6 * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output." << std::endl;
        return false;
    }

    inputData.resize(m_kBatchSize * m_channel * m_kInputH * m_kInputW);
    output_data.resize(m_kBatchSize * kOutputSize * 6);
    return true;
}

// Deserialize the engine from file
bool YOLOV10ModelManager::deserializeEngine(const std::string engine_name) {
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

void YOLOV10ModelManager::preProcess(std::vector<cv::Mat> img_batch, std::vector<float>& data) {
    int data_index = 0;
    // 创建一个缓存图像用于复制和调整大小操作
    cv::Mat maxImage;
    cv::Mat resizeImg;
    int length = std::max(m_kInputW, m_kInputH);
    factor.clear();
    // 遍历每张图像
    for (const auto &frame : img_batch) {
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        // 获取输入图像的大小
        int rh = rgb_frame.rows;
        int rw = rgb_frame.cols;

        // 创建一个边长为输入图像最大边长的方形图像
        int maxImageLength = std::max(rh, rw);
        maxImage = cv::Mat::zeros(maxImageLength, maxImageLength, rgb_frame.type());

        // 将输入图像复制到方形图像的左上角
        rgb_frame.copyTo(maxImage(cv::Rect(0, 0, rw, rh)));

        // 将方形图像调整为指定大小
        cv::resize(maxImage, resizeImg, cv::Size(m_kInputW, m_kInputH), 0, 0, cv::INTER_LINEAR);

        // 计算缩放因子
        factor.emplace_back(static_cast<float>(maxImageLength) / length);

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

// postprocess the output data
bool YOLOV10ModelManager::postProcess(float* result, std::vector<std::vector<DetBox>>& detBoxs) {
    if(NULL == result) {
        std::cerr << "result data is NULL." << std::endl;
        return false;
    }
    int index = 0;
    for (int i = 0; i < output_data.size(); i+=kOutputSize*6) {
        std::vector<DetBox> detresult;
        for (int j = 0; j < kOutputSize*6; j+=6){
            float score = (float)result[i + j + 4];
            // Due to that YOLOv10 adopts the different training strategy with others, e.g. YOLOv8, 
            // it may thus have different favorable confidence threshold to detect objects. 
            // Besides, different thresholds will have no impact on the inference latency of YOLOv10 because it does not rely on NMS. 
            // Therefore, we suggest that a smaller threshold can be freely set or tuned for YOLOv10 to detect smaller objects or objects in the distance.
            // ref: https://github.com/THU-MIG/YOLOv10/issues/136, set conf > 0.05 rather than 0.25 for samller objects
            if (score > conf) {
                // x1 = topleft_x, y1 = topleft_y, x2 = bottomright_x, y2 = bottomright_y
                float x1 = result[i + j]* factor[index];
                float y1 = result[i + j + 1]* factor[index];
                float x2 = result[i + j + 2]* factor[index];
                float y2 = result[i + j + 3]* factor[index];
                int label_id = (int)result[i + j + 5];
                x1 = std::max(0, static_cast<int>(x1));
                y1 = std::max(0, static_cast<int>(y1));
                // Create a DetBox object and fill its fields
                DetBox box;
                box.x = x1;
                box.y = y1;
                box.w = x2 - x1; // width
                box.h = y2 - y1; // height
                box.confidence = score;
                box.classID = label_id;
                // Add the DetBox to the detBoxs vector
                detresult.emplace_back(box);
            }
        }
        detBoxs.emplace_back(detresult);
        index += 1;
    }
    return true;
}

bool YOLOV10ModelManager::doInference(const std::vector<cv::Mat> img_batch, std::vector<std::vector<DetBox>>& batchDetBoxes) {
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
    
    if (cudaMemcpyAsync(output_data.data(), outputSrcDevice, img_batch.size() * kOutputSize * 6 * sizeof(float), 
            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "Failed to copy output data to host." << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);
    return postProcess(output_data.data(), batchDetBoxes);
}

// Perform inference on the input frame
bool YOLOV10ModelManager::inference(cv::Mat frame, std::vector<DetBox>& detBoxs) {
    std::vector<cv::Mat> img_batch = {frame};
    std::vector<std::vector<DetBox>> batchDetBoxes;
    if (!doInference(img_batch, batchDetBoxes)) {
        return false;
    }
    detBoxs = batchDetBoxes[0];
    return true;
}

// Perform batch inference on the input frames
bool YOLOV10ModelManager::batchInference(std::vector<cv::Mat> img_frames, std::vector<std::vector<DetBox>>& batchDetBoxes) {
    if (img_frames.empty()) {
        std::cerr << "Input batch is empty." << std::endl;
        return false;
    }

    for (size_t i = 0; i < img_frames.size(); i += m_kBatchSize) {
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + m_kBatchSize && j < img_frames.size(); j++) {
            img_batch.emplace_back(img_frames[j]);
        }

        if (!doInference(img_batch, batchDetBoxes)) {
            return false;
        }
    }

    return true;
}
