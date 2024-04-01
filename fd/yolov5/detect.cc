#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision.h"

namespace fs = std::filesystem;
// using namespace std::filesystem;

void InferImage(fastdeploy::vision::detection::YOLOv5& model, const std::string& image_file, const std::string& output_folder);
void InferVideo(fastdeploy::vision::detection::YOLOv5& model, const std::string& video_file, const std::string& output_folder);

void InferFolder(const std::string& model_file, const std::string& source_folder, int run_option, const std::string& output_folder = ".", const int img_size = 640) {
  auto option = fastdeploy::RuntimeOption();
  if (run_option == 1 || run_option == 2) {
    option.UseGpu();
    if (run_option == 2) {
      option.UseTrtBackend();
      option.SetTrtInputShape("images", {1, 3, img_size, img_size});
    }
  }
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  model.GetPreprocessor().SetSize({img_size, img_size});
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize model." << std::endl;
    return;
  }

  // 确保输出文件夹存在
  fs::create_directories(output_folder);

  for (const auto& entry : fs::directory_iterator(source_folder)) {
    if (entry.is_regular_file()) {
      std::string file_path = entry.path().string();
      std::string extension = entry.path().extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

      if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
        InferImage(model, file_path, output_folder);
      } else if (extension == ".mp4" || extension == ".avi") {
        InferVideo(model, file_path, output_folder);
      }
    }
  }
}

void InferImage(fastdeploy::vision::detection::YOLOv5& model, const std::string& image_file, const std::string& output_folder) {
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
  std::cout << res.Str() << std::endl;
  auto vis_image = fastdeploy::vision::VisDetection(image, res);
  fs::path output_path = fs::path(output_folder) / ("vis_" + fs::path(image_file).filename().string());
  cv::imwrite(output_path.string(), vis_image);
}

void InferVideo(fastdeploy::vision::detection::YOLOv5& model, const std::string& video_file, const std::string& output_folder) {

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

    if (res.boxes.size() > 0){
      std::cout << res.Str() << std::endl;
    }
    auto vis_frame = fastdeploy::vision::VisDetection(frame, res);
    video_writer.write(vis_frame);
  }

  cap.release();
  video_writer.release();
}

int main(int argc, char* argv[]) {
  std::string output_folder = ".";
  int img_size = 640;
  if (argc < 4) {
    std::cout << "Usage: infer_folder path/to/model path/to/folder run_option [path/to/output_folder]" << std::endl;
    return -1;
  }
  if (argc >= 5) {
    output_folder = argv[4];
  }
  if (argc == 6){
    img_size = std::atoi(argv[5]);
  }

  InferFolder(argv[1], argv[2], std::atoi(argv[3]), output_folder, img_size);
  return 0;
}