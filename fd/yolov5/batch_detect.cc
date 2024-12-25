#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision.h"

namespace fs = std::filesystem;
// using namespace std::filesystem;

void InferImagesBatch(fastdeploy::vision::detection::YOLOv5& model, const std::vector<std::string>& batch_files, const std::string& output_folder);
void InferVideo(fastdeploy::vision::detection::YOLOv5& model, const std::string& video_file, const std::string& output_folder, const int batch_size);

void InferFolder(
    const std::string& model_file, 
    const std::string& source_folder, 
    int run_option, 
    const std::string& output_folder = ".", 
    const int img_size = 640,
    const int bs = 16
) {
    auto option = fastdeploy::RuntimeOption();
    if (run_option == 1 || run_option == 2) {
        option.UseGpu();
        if (run_option == 2) {
        option.UseTrtBackend();
        option.SetTrtInputShape("images", {bs, 3, img_size, img_size});
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

    std::vector<std::string> batch_files;
    batch_files.reserve(bs); // 预分配内存以提高性能

    for (const auto& entry : fs::directory_iterator(source_folder)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") {
                batch_files.push_back(file_path);
                if (batch_files.size() == bs) {
                    InferImagesBatch(model, batch_files, output_folder);
                    batch_files.clear(); // 清空批次列表以用于下一批次
                }
            }
            else if (extension == ".mp4" || extension == ".avi") {
                InferVideo(model, file_path, output_folder, bs);
            }
        }
    }

    // 处理剩余的图片（如果有）
    if (!batch_files.empty()) {
        InferImagesBatch(model, batch_files, output_folder);
    }
}

void InferImagesBatch(fastdeploy::vision::detection::YOLOv5& model, const std::vector<std::string>& batch_files, const std::string& output_folder) {
    std::vector<cv::Mat> batch_images;
    batch_images.reserve(batch_files.size());

    for (const auto& file_path : batch_files) {
        auto image = cv::imread(file_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << file_path << std::endl;
            continue; // Skip this image and continue with the next
        }
        batch_images.push_back(image);
    }

    if (batch_images.empty()) {
        std::cerr << "No valid images in the batch." << std::endl;
        return;
    }

    std::vector<fastdeploy::vision::DetectionResult> batch_results;
    if (!model.BatchPredict(batch_images, &batch_results)) {
        std::cerr << "Failed to predict batch." << std::endl;
        return;
    }

    for (size_t i = 0; i < batch_images.size(); ++i) {
        std::cout << batch_results[i].Str() << std::endl;
        auto vis_image = fastdeploy::vision::VisDetection(batch_images[i], batch_results[i]);
        fs::path output_path = fs::path(output_folder) / ("vis_" + fs::path(batch_files[i]).filename().string());
        cv::imwrite(output_path.string(), vis_image);
    }
}

void InferVideo(fastdeploy::vision::detection::YOLOv5& model, const std::string& video_file, const std::string& output_folder, const int batch_size) {
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file: " << video_file << std::endl;
        return;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    fs::path output_video_path = fs::path(output_folder) / ("vis_" + fs::path(video_file).filename().string());
    cv::VideoWriter video_writer(output_video_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

    std::vector<cv::Mat> batch_frames;
    std::vector<fastdeploy::vision::DetectionResult> batch_results;
    batch_frames.reserve(batch_size);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            // If there are frames in the batch, process them before breaking
            if (!batch_frames.empty()) {
                if (!model.BatchPredict(batch_frames, &batch_results)) {
                    std::cerr << "Failed to predict batch." << std::endl;
                    return;
                }
                for (const auto& res : batch_results) {
                    auto vis_frame = fastdeploy::vision::VisDetection(batch_frames[&res - &batch_results[0]], res);
                    video_writer.write(vis_frame);
                }
            }
            break;
        }

        batch_frames.push_back(frame.clone());
        if (batch_frames.size() == batch_size) {
            if (!model.BatchPredict(batch_frames, &batch_results)) {
                    std::cerr << "Failed to predict batch." << std::endl;
                    return;
            }

            for (const auto& res : batch_results) {
                auto vis_frame = fastdeploy::vision::VisDetection(batch_frames[&res - &batch_results[0]], res);
                video_writer.write(vis_frame);
            }

            batch_frames.clear();
            batch_results.clear();
        }
    }

    cap.release();
    video_writer.release();
}

int main(int argc, char* argv[]) {
  std::string output_folder = ".";
  int img_size = 640;
  int bs = 16;
  if (argc < 4) {
    std::cout << "Usage: infer_folder path/to/model path/to/folder run_option [path/to/output_folder] [batch_size, default to 16]" << std::endl;
    return -1;
  }
  if (argc >= 5) {
    output_folder = argv[4];
  }
  if (argc >= 6){
    img_size = std::atoi(argv[5]);
    if (img_size <= 0) {
        std::cerr << "Invalid image size provided. Using default value: 640" << std::endl;
        img_size = 640;
        }
  }
  if (argc >= 7) {
    bs = std::atoi(argv[6]);
    if (bs <= 0) {
        std::cerr << "Invalid batch size provided. Using default value: 16" << std::endl;
        bs = 16;
    }
  }

  InferFolder(argv[1], argv[2], std::atoi(argv[3]), output_folder, img_size, bs);
  return 0;
}