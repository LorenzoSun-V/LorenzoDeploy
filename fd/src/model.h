/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-04-26 14:21:20
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-05-06 15:11:12
 * @Description: 
 */
# pragma once

#include <string>
#include "fastdeploy/vision.h"
#include "common.h"

class Model{
private:
    void ProcessBatchImage(std::vector<cv::Mat>& batch_images, std::vector<std::string>& batch_names, std::vector<fastdeploy::vision::DetectionResult>* batch_results);
    void ProcessBatchVideo(std::vector<cv::Mat>& batch_frames, std::vector<fastdeploy::vision::DetectionResult>* batch_results, const std::string& video_file, cv::VideoWriter& video_writer, int* num);
    void SaveOriginalImage(const cv::Mat& image, const std::string& save_path, const fastdeploy::vision::DetectionResult& res);

protected:
    Config cfg;
    bool save_ori = false;  // 如果满足cfg中的threshold，保存原图
    fastdeploy::RuntimeOption configureRuntimeOptions() const;

public:
    Model(const std::string& config_file);
    virtual void InitModel(const Config& cfg) = 0;
    virtual bool Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) = 0;
    virtual bool BatchPredict(const std::vector<cv::Mat>& batch_images, std::vector<fastdeploy::vision::DetectionResult>* batch_results) = 0;
    void InferImage(const std::string& image_file);
    void InferImagesBatch(const std::vector<std::string>& batch_files);
    void InferVideo(const std::string& video_file);
    void InferFolder();

};


class YOLOv5Model : public Model{
protected:
    std::shared_ptr<fastdeploy::vision::detection::YOLOv5> model;

public:
    YOLOv5Model(const std::string& config_path) : Model(config_path) {InitModel(cfg);}

    void InitModel(const Config& cfg) override;
    bool Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) override;
    bool BatchPredict(const std::vector<cv::Mat>& batch_images, std::vector<fastdeploy::vision::DetectionResult>* batch_results) override;
};


class YOLOv8Model : public Model{
protected:
    std::shared_ptr<fastdeploy::vision::detection::YOLOv8> model;

public:
    YOLOv8Model(const std::string& config_path) : Model(config_path) {InitModel(cfg);}

    void InitModel(const Config& cfg) override;
    bool Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) override;
    bool BatchPredict(const std::vector<cv::Mat>& batch_images, std::vector<fastdeploy::vision::DetectionResult>* batch_results) override;
};

cv::Mat VisDetectionCustom(const cv::Mat& im, const fastdeploy::vision::DetectionResult& result,
                     float score_threshold, int line_size, float font_size);