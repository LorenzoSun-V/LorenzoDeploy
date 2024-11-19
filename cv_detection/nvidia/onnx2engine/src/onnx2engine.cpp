/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-28 11:11:54
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-16 16:08:48
 * @Description: 
 */
#include <fstream>
#include <iostream>
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// judge file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

bool onnxToEngine(const std::string& onnxFile, int memorySize) {
    Logger gLogger;
    std::string path(onnxFile);
    std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
    std::string modelPath = path.substr(0, iPos); // get the path of onnx model
    std::string modelName = path.substr(iPos, path.length() - iPos); // get the name of onnx model
    std::string modelName_ = modelName.substr(0, modelName.rfind(".")); // get the name of onnx model without suffix
    std::string engineFile = modelPath + modelName_ + ".bin";

    // phase onnx model
    std::cout << "Try to convert onnx model to TensorRT engine model..." << std::endl;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxFile.c_str(), 2)) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    std::cout << "TensorRT loads mask onnx model successfully!" << std::endl;
    
    // create engine
    std::cout << "Try to create engine..." << std::endl;
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);  // FP16 infer
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to create engine." << std::endl;
        return false;
    }
    std::cout << "Create engine successfully!" << std::endl;

    // save engine file
    std::cout << "Try to save engine file..." << std::endl;
    std::ofstream filePtr(engineFile, std::ios::binary);
    if (!filePtr) {
        std::cerr << "Could not open plan output file!" << std::endl;
        return false;
    }
    nvinfer1::IHostMemory* modelStream = engine->serialize();
    filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    // Destroy the engine
    modelStream->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    std::cout << "Convert onnx model to TensorRT engine model successfully!" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "example: ./binary yolov10.onnx" << std::endl;
        return -1;
    }

    std::string onnx_path = argv[1];
    if (!fileExists(onnx_path)) {
        std::cerr << "File not found: " << onnx_path << std::endl;
        return -1;
    }
    // the path of engine is as the same as the onnx model, but the suffix is ".engine"
    if (!onnxToEngine(onnx_path, 50)){
        std::cerr << "Convert onnx model to TensorRT engine model failed!" << std::endl;
        return -1;
    }
    return 0;
}
