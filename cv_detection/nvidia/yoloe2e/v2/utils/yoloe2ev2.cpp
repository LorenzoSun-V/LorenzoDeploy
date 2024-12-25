#include "yoloe2ev2.h"


YOLOE2Ev2ModelManager::YOLOE2Ev2ModelManager()
    : m_kBatchSize(1), m_channel(3), m_kInputH(640), m_kInputW(640),
    m_maxObject(20),runtime(nullptr), engine(nullptr), context(nullptr), stream(0),
    host_output0(nullptr),host_output1(nullptr),host_output2(nullptr),host_output3(nullptr) {

    }

YOLOE2Ev2ModelManager::~YOLOE2Ev2ModelManager() {
    if (context) {
        context->destroy();
        context = nullptr;
    }
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    for (int i = 0; i < 5; ++i) {
        if (buffers[i]) {
            cudaFree(buffers[i]);
            buffers[i] = nullptr;
        }
    }
    // 释放主机内存
    if (host_output0) {
        delete[] host_output0;
        host_output0=NULL;
    } 
    if(host_output1){
        delete[] host_output1;
        host_output1=NULL;
    }
    if(host_output2){
        delete[] host_output2;
        host_output2=NULL;
    }
    if(host_output3){
        delete[] host_output3;
        host_output3=NULL;
    }
}

bool YOLOE2Ev2ModelManager::loadModel(const std::string engine_name){
    struct stat buffer;
    if (!stat(engine_name.c_str(), &buffer) == 0) {
        std::cerr << "Error: File " << engine_name << " does not exist!" << std::endl;
        return false;
    }
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }

    if(engine->getNbBindings() != 5){
        std::cerr << "Please input correct network model." << std::endl;
        return false;
    }

    //const int inputIndex = engine->getBindingIndex("images");
    auto inputDims = engine->getBindingDimensions(0);

    m_kBatchSize = inputDims.d[0];
    m_channel = inputDims.d[1];
    m_kInputH = inputDims.d[2];
    m_kInputW = inputDims.d[3];

    std::cout << "m_kBatchSize: " << m_kBatchSize << " m_channel: " << m_channel << " m_kInputH: " << m_kInputH << " m_kInputW: " << m_kInputW << std::endl;
    
    int maxObjectNumbers = engine->getBindingDimensions(4).d[1];//获得目标框数量
    std::cout << "max out Object Numbers: "<< maxObjectNumbers << std::endl;
    if(maxObjectNumbers >= 20) m_maxObject = maxObjectNumbers; //根据网络设置最大检测对象数量分配空间

    input.resize(m_kBatchSize*m_kInputH * m_kInputW * 3);//保存图像数据
    if (cudaMalloc(&buffers[0], m_kBatchSize * m_kInputH * m_kInputW * 3 * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input buffer." << std::endl;
        return false;
    }

    //<- num_detections
    if (cudaMalloc(&buffers[1], m_kBatchSize * sizeof(int)) != cudaSuccess) { 
        std::cerr << "Failed to allocate device memory for output buffer 1." << std::endl;
        return false;
    }
    //<- nmsed_boxes
    if (cudaMalloc(&buffers[2], m_kBatchSize * m_maxObject * 4 * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output buffer 2." << std::endl;
        return false;
    }
    //<- nmsed_scores
    if (cudaMalloc(&buffers[3], m_kBatchSize * m_maxObject * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output buffer 3." << std::endl;
        return false;
    }
    //<- nmsed_classes
    if (cudaMalloc(&buffers[4], m_kBatchSize * m_maxObject * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output buffer 4." << std::endl;
        return false;
    }

    // 同步并从设备(GPU)内存中拷贝数据到主机(CPU)
    host_output0 = new int[m_kBatchSize];  // 预测框的数量
    host_output1 = new float[m_kBatchSize * m_maxObject * 4];  // 预测框坐标
    host_output2 = new float[m_kBatchSize * m_maxObject];  // 预测框置信度
    host_output3 = new float[m_kBatchSize * m_maxObject];  // 预测框类别

    return true;
}

// Deserialize the engine from file
bool YOLOE2Ev2ModelManager::deserializeEngine(const std::string engine_name) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open engine file." << std::endl;
        return false;
    }

    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Logger gLogger;
    initLibNvInferPlugins(&gLogger, "");
    runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create Infer Runtime." << std::endl;
        return false;
    }

    engine = runtime->deserializeCudaEngine(data.data(), data.size(), nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        return false;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return false;
    }

    return true; 
}

void YOLOE2Ev2ModelManager::preprocess(cv::Mat& img, std::vector<float>& data){
    int w, h, x, y;
	float r_w = m_kInputW / (img.cols*1.0f);
	float r_h = m_kInputH / (img.rows*1.0f);
	if (r_h > r_w) {
		w = m_kInputW;
		h = r_w * img.rows;
		x = 0;
		y = (m_kInputH - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = m_kInputH;
		x = (m_kInputW - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	//cudaResize(img, re);
	cv::Mat out(m_kInputH, m_kInputW, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < m_kInputH; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < m_kInputW; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + m_kInputH * m_kInputW] = (float)uc_pixel[1] / 255.0f;
			data[i + 2 * m_kInputH * m_kInputW] = (float)uc_pixel[0] / 255.0f;
			uc_pixel += 3;
			++i;
		}
	}
}

void YOLOE2Ev2ModelManager::rescale_box(std::vector<DetBox>& pred_box, std::vector<DetBox>& detBoxs, int width, int height){
    float l_length = std::max(m_kInputH, m_kInputW);
    float gain = l_length / std::max(width, height);
	float pad_x = (l_length - width * gain) / 2;
	float pad_y = (l_length - height * gain) / 2;

    for (const auto& box : pred_box) {
        DetBox scaled_box;
        scaled_box.x = (box.x - pad_x) / gain;
        scaled_box.y = (box.y - pad_y) / gain;
        scaled_box.w = box.w / gain;
        scaled_box.h = box.h / gain;
        scaled_box.x = scaled_box.x - scaled_box.w/2;
        scaled_box.y = scaled_box.y - scaled_box.h/2;
        scaled_box.confidence = box.confidence;
        scaled_box.classID = box.classID;
        detBoxs.push_back(scaled_box);
    }
}

bool YOLOE2Ev2ModelManager::doInference()
{
    if (!context->enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "Failed to enqueue inference." << std::endl;
        return false;
    }

    cudaMemcpyAsync(host_output0, buffers[1], m_kBatchSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(host_output1, buffers[2], m_kBatchSize * m_maxObject * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(host_output2, buffers[3], m_kBatchSize * m_maxObject * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(host_output3, buffers[4], m_kBatchSize * m_maxObject * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    return true;
}

bool YOLOE2Ev2ModelManager::inference(cv::Mat frame, std::vector<DetBox>& detBoxs)
{
    preprocess(frame, input);
    cudaMemcpyAsync(buffers[0], input.data(), m_kInputH * m_kInputW * 3 * sizeof(float), cudaMemcpyHostToDevice);
    doInference();
    detBoxs.clear();//清空detBoxs，确保其为空
    std::vector<DetBox> pred_box;
    for (int i = 0; i < host_output0[0]; i++){
        DetBox box;
	    box.x = (host_output1[i * 4 + 2] + host_output1[i * 4]) / 2.0;
		box.y = (host_output1[i * 4 + 3] + host_output1[i * 4 + 1]) / 2.0;
		box.w = host_output1[i * 4 + 2] - host_output1[i * 4];
		box.h = host_output1[i * 4 + 3] - host_output1[i * 4 + 1];
		box.confidence = host_output2[i];
		box.classID = (int)host_output3[i];
        pred_box.push_back(box);
    }

    rescale_box(pred_box, detBoxs, frame.cols, frame.rows);
    return true;
}

bool YOLOE2Ev2ModelManager::batchinference(std::vector<cv::Mat> frames, std::vector<std::vector<DetBox>>& batchBoxes) {
    if (frames.empty()) {
        std::cerr << "Input batch is empty." << std::endl;
        return false;
    }

    // 清空 batchBoxes，确保其为空
    batchBoxes.clear();

    // 按批次处理图像
    for (size_t i = 0; i < frames.size(); i += m_kBatchSize) {
        // 确定实际的批次大小，处理最后一个可能不足的批次
        size_t batchSize = std::min(static_cast<size_t>(m_kBatchSize), frames.size() - i);

        // 存储批次的输入数据，确保在异步拷贝期间数据有效
        std::vector<std::vector<float>> inputs(batchSize);
        for (size_t j = 0; j < batchSize; j++) {
            // 调整 inputs[j] 的大小以容纳预处理后的图像数据
            inputs[j].resize(m_kInputH * m_kInputW * 3);

            // 预处理图像
            preprocess(frames[i + j], inputs[j]);

            // 将预处理后的数据复制到设备输入缓冲区
            cudaMemcpyAsync(static_cast<float*>(buffers[0]) + j * m_kInputH * m_kInputW * 3, inputs[j].data(),
                m_kInputH * m_kInputW * 3 * sizeof(float),cudaMemcpyHostToDevice,stream);
        }

        // 运行推理
        context->enqueueV2(buffers, stream, nullptr);

        // 从设备拷贝输出到主机
        cudaMemcpyAsync(host_output0, buffers[1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_output1, buffers[2], batchSize * m_maxObject * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_output2, buffers[3], batchSize * m_maxObject * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(host_output3, buffers[4], batchSize * m_maxObject * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 同步流，确保所有操作完成
        cudaStreamSynchronize(stream);

        // 解析每张图像的输出
        for (size_t j = 0; j < batchSize; j++) {
            std::vector<DetBox> detBoxs;  // 当前图像的检测框
            std::vector<DetBox> pred_box; // 预测框，待回归到原始尺寸

            int num_detections = host_output0[j];
            //std::cout << "Image " << i + j << " num_detections: " << num_detections << std::endl;

            // 遍历当前图像的所有检测结果
            for (int index = 0; index < num_detections; index++) {
                // 计算当前检测结果在输出数组中的索引
                size_t idx = j * m_maxObject + index;  // 假设每张图片的最大检测数量为 m_maxObject

                // 提取检测结果数据
                DetBox box;
                box.x = (host_output1[idx * 4 + 2] + host_output1[idx * 4]) / 2.0f;
                box.y = (host_output1[idx * 4 + 3] + host_output1[idx * 4 + 1]) / 2.0f;
                box.w = host_output1[idx * 4 + 2] - host_output1[idx * 4];
                box.h = host_output1[idx * 4 + 3] - host_output1[idx * 4 + 1];
                box.confidence = host_output2[idx];
                box.classID = static_cast<int>(host_output3[idx]);

                pred_box.push_back(box);
            }

            // 将预测框回归到原始图像尺寸
            rescale_box(pred_box, detBoxs, frames[i + j].cols, frames[i + j].rows);

            // 将当前图像的检测结果添加到 batchBoxes
            batchBoxes.push_back(detBoxs);
        }
    }

    return true;
}
