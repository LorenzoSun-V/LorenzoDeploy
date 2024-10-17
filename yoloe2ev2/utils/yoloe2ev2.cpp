#include <stdio.h>
#include <string>
#include <iostream>
#include <memory>
#include <utility>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <iterator>

#include "yoloe2ev2.h"
#include "logging.h"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <npp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

YOLOE2Ev2ModelManager::YOLOE2Ev2ModelManager()
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(0){

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
}

bool YOLOE2Ev2ModelManager::loadModel(const std::string engine_name){
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

    input.resize(m_kInputH * m_kInputW * 3);
    if (cudaMalloc(&buffers[0], m_kInputH * m_kInputW * 3 * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input buffer." << std::endl;
        return false;
    }
    // cudaMalloc(&buffers[0], m_kInputH * m_kInputW * 3 * sizeof(float));  //<- input
    cudaMalloc(&buffers[1], 1 * sizeof(int)); //<- num_detections
	cudaMalloc(&buffers[2], 1 * 200 * 4 * sizeof(float)); //<- nmsed_boxes
	cudaMalloc(&buffers[3], 1 * 200 * sizeof(float)); //<- nmsed_scores
	cudaMalloc(&buffers[4], 1 * 200 * sizeof(float)); //<- nmsed_classes

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

bool YOLOE2Ev2ModelManager::inference(cv::Mat frame, std::vector<DetBox>& detBoxs){
    preprocess(frame, input);
    cudaMemcpyAsync(buffers[0], input.data(), m_kInputH * m_kInputW * 3 * sizeof(float), cudaMemcpyHostToDevice);
    context->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(output0, buffers[1], 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(output1, buffers[2], 1 * 200 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(output2, buffers[3], 1 * 200 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(output3, buffers[4], 1 * 200 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    
    std::vector<DetBox> pred_box;
  
    for (int i = 0; i < output0[0]; i++){
        DetBox box;
	    box.x = (output1[i * 4 + 2] + output1[i * 4]) / 2.0;
		box.y = (output1[i * 4 + 3] + output1[i * 4 + 1]) / 2.0;
		box.w = output1[i * 4 + 2] - output1[i * 4];
		box.h = output1[i * 4 + 3] - output1[i * 4 + 1];
		box.confidence = output2[i];
		box.classID = (int)output3[i];
        pred_box.push_back(box);
    }

    rescale_box(pred_box, detBoxs, frame.cols, frame.rows);
    return true;
}