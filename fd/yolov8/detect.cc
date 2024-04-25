#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "fastdeploy/vision.h"
#include "common.h"

namespace fs = std::filesystem;

void InferImage(fastdeploy::vision::detection::YOLOv8& model, const std::string& image_file, const std::string& output_folder, const float threshold);
void InferVideo(fastdeploy::vision::detection::YOLOv8& model, const std::string& video_file, const std::string& output_folder, const float threshold);

void InferFolder(const Config& cfg) {
  auto option = fastdeploy::RuntimeOption();
  if (cfg.run_option == 1 || cfg.run_option == 2) {
    option.UseGpu();
    if (cfg.run_option == 2) {
      option.UseTrtBackend();
      option.SetTrtInputShape("images", {1, 3, cfg.img_size, cfg.img_size});
    }
  }
  auto model = fastdeploy::vision::detection::YOLOv8(cfg.model_path, "", option);
  model.GetPreprocessor().SetSize({cfg.img_size, cfg.img_size});
  model.GetPostprocessor().SetConfThreshold(cfg.conf);
  model.GetPostprocessor().SetNMSThreshold(cfg.nms_iou);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize model." << std::endl;
    return;
  }

  // 确保输出文件夹存在
  fs::create_directories(cfg.output_folder);

  for (const auto& entry : fs::directory_iterator(cfg.source_folder)) {
    if (entry.is_regular_file()) {
      std::string file_path = entry.path().string();
      std::string extension = entry.path().extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

      if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
        InferImage(model, file_path, cfg.output_folder, cfg.threshold);
      } else if (extension == ".mp4" || extension == ".avi") {
        InferVideo(model, file_path, cfg.output_folder, cfg.threshold);
      }
    }
  }
}

void InferImage(fastdeploy::vision::detection::YOLOv8& model, const std::string& image_file, const std::string& output_folder, const float threshold = 1.0) {
  auto image = cv::imread(image_file);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << image_file << std::endl;
    return;
  }

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(image, &res)) {
    std::cerr << "Failed to predict image: " << image_file << std::endl;
    return;
  }
  std::cout << "Image: " << image_file << std::endl;
  std::cout << res.Str() << std::endl;
  auto vis_image = fastdeploy::vision::VisDetection(image, res);
  fs::path output_path = fs::path(output_folder) / ("vis_" + fs::path(image_file).filename().string());
  fs::path output_path_ori = fs::path(output_folder) / ("ori_" + fs::path(image_file).filename().string());

  // 如果设定的threshold<1，保存满足阈值的原图
  if (threshold < 1.0){
    for (int i = 0; i < res.boxes.size(); i++){
      if (res.scores[i] > threshold){
        cv::imwrite(output_path_ori.string(), image);
        break;
      }
    }
  }
  // 保存所有有结果的图
  cv::imwrite(output_path.string(), vis_image);
}

void InferVideo(fastdeploy::vision::detection::YOLOv8& model, const std::string& video_file, const std::string& output_folder, const float threshold = 1.0) {

  cv::VideoCapture cap(video_file, cv::CAP_FFMPEG);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video file: " << video_file << std::endl;
    return;
  }

  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);

  fs::path output_video_path = fs::path(output_folder) / ("vis_" + fs::path(video_file).filename().string());
  cv::VideoWriter video_writer(output_video_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

  cv::Mat frame;
  int num = 0;
  while (true) {
    cap >> frame;
    if (frame.empty()) break;

    fastdeploy::vision::DetectionResult res;

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
    if (!model.Predict(frame, &res)) {
      std::cerr << "Failed to predict video frame." << std::endl;
      continue;
    }
    // 停止计时
    auto end = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << "Prediction time: " << duration.count() << " ms" << std::endl;
    
    auto vis_frame = fastdeploy::vision::VisDetection(frame, res);
    video_writer.write(vis_frame);
    if (res.boxes.size() > 0){
      std::cout << res.Str() << std::endl;
      // 如果设定的threshold<1，保存满足阈值的原图
      if (threshold < 1.0){
        for (int i = 0; i < res.boxes.size(); i++){
          if (res.scores[i] > threshold){
            fs::path output_path_ori = fs::path(output_folder) / (fs::path(video_file).stem().string() + "_" + std::to_string(num) + ".jpg");
            fs::path output_path_vis = fs::path(output_folder) / (fs::path(video_file).stem().string() + "_" + std::to_string(num) + "_vis.jpg");
            cv::imwrite(output_path_ori.string(), frame);
            cv::imwrite(output_path_vis.string(), vis_frame);
            num++;
            break;
          }
        }
      }
    }
  }

  cap.release();
  video_writer.release();
}

Config load_config(const std::string& config_file){
  YAML::Node config = YAML::LoadFile(config_file);
  Config cfg;
  try {
    cfg.model_path = config["model_path"].as<std::string>();
    cfg.source_folder = config["source_folder"].as<std::string>();
    cfg.output_folder = config["output_folder"].as<std::string>();
    cfg.run_option = config["run_option"].as<int>();
    cfg.img_size = config["img_size"].as<int>();
    cfg.conf = config["conf"].as<float>();
    cfg.nms_iou = config["nms_iou"].as<float>();
    cfg.threshold = config["threshold"].as<float>();
  } catch (YAML::Exception& e) {
    std::cerr << "Error parsing YAML: " << e.what() << std::endl;
    std::cerr << "Please check yaml config! Missing field!" << std::endl;
    throw; // 或者处理错误，比如返回一个错误码或默认配置
  }
  return cfg;
}

int main(int argc, char* argv[]) {
  if (argc!=2){
    std::cerr << "Usage: ./detect path/to/yaml" << std::endl;
    return -1;
  }
  std::string yaml_path = argv[1];
  Config cfg = load_config(yaml_path);
  InferFolder(cfg);
  return 0;
}