# pragma once

#include <string>
#include "fastdeploy/vision.h"
#include "common.h"

class Model{
protected:
    Config cfg;
    fastdeploy::RuntimeOption configureRuntimeOptions() const;

public:
    Model(const std::string& config_file);
    virtual void InitModel(const Config& cfg) = 0;
    virtual bool Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) = 0;
    void InferImage(const std::string& image_file);
    // void InferVideo(const std::string& video_file, const std::string& output_folder) = 0;
    // void InferFolder(const std::string& folder_path, const std::string& output_folder, ) = 0;

};

class YOLOv8Model : public Model{
protected:
    std::shared_ptr<fastdeploy::vision::detection::YOLOv8> model;

public:
    YOLOv8Model(const std::string& config_path) : Model(config_path) {InitModel(cfg);}

    void InitModel(const Config& cfg) override;
    bool Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) override;

};