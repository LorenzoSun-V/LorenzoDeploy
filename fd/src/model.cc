#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision.h"
#include "model.h"

namespace fs = std::filesystem;

// 基类Model的构造函数，加载配置文件并打印配置信息
Model::Model(const std::string& config_path){
    if (!fs::exists(config_path)){
        std::cerr << "Config file not found: " << config_path << std::endl;
    }
    cfg = load_config(config_path);
    print_config(cfg);
}

// 基类Model的configureRuntimeOptions函数，根据配置文件设置运行时选项
fastdeploy::RuntimeOption Model::configureRuntimeOptions() const {
    auto option = fastdeploy::RuntimeOption();
    if (cfg.run_option == 1 || cfg.run_option == 2) {
        option.UseGpu();
        if (cfg.run_option == 2) {
            option.UseTrtBackend();
            option.SetTrtInputShape("images", {cfg.bs, 3, cfg.img_size, cfg.img_size});
        }
    }
    return option;
}

void Model::InferImage(const std::string& image_file){
    auto image = cv::imread(image_file);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_file << std::endl;
        return;
    }
    fastdeploy::vision::DetectionResult res;
    if (!Predict(image, &res)) {
        std::cerr << "Failed to predict image: " << image_file << std::endl;
        return;
    }
    std::cout << "Image: " << image_file << std::endl;
    std::cout << res.Str() << std::endl;
    auto vis_image = fastdeploy::vision::VisDetection(image, res);
    fs::path output_path = fs::path(cfg.output_folder) / ("vis_" + fs::path(image_file).filename().string());

    // 如果设定的threshold<1，保存满足阈值的原图
    if (cfg.threshold < 1.0){
        fs::path output_path_ori = fs::path(cfg.output_folder) / ("ori_" + fs::path(image_file).filename().string());
        for (int i = 0; i < res.boxes.size(); i++){
            if (res.scores[i] > cfg.threshold){
                cv::imwrite(output_path_ori.string(), image);
                break;
            }
        }
    }
    // 保存所有有结果的图
    cv::imwrite(output_path.string(), vis_image);
}

// YOLOv8Model的构造函数，调用基类Model的构造函数并初始化模型
void YOLOv8Model::InitModel(const Config& cfg) {
    auto option = configureRuntimeOptions();
    model = std::make_shared<fastdeploy::vision::detection::YOLOv8>(cfg.model_path, "", option);
    model->GetPreprocessor().SetSize({cfg.img_size, cfg.img_size});
    model->GetPostprocessor().SetConfThreshold(cfg.conf);
    model->GetPostprocessor().SetNMSThreshold(cfg.nms_iou);
    if (!model->Initialized()) {
        std::cerr << "Failed to initialize model." << std::endl;
    }
}

bool YOLOv8Model::Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) {
    if (!model->Predict(image, res)) {
        return false;
    }
    return true;
}