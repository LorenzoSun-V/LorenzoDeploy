// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/vision.h"
#include <chrono>


std::string save_res2txt(const fastdeploy::vision::DetectionResult& res){
    std::string data;
    for (size_t i = 0; i < res.boxes.size(); ++i) {
        data = data + std::to_string(res.boxes[i][0]) + " " +
          std::to_string(res.boxes[i][1]) + " " + std::to_string(res.boxes[i][2]) +
          " " + std::to_string(res.boxes[i][3]) + " " +
          std::to_string(res.scores[i]) + " " + std::to_string(res.label_ids[i]);
        data += "\n";
    }
    return data;
}

void CpuInfer(const std::string& model_file, 
              const std::string& img_dir, 
              const float& conf_threshold, 
              const float& nms_threshold,
              const int& img_size) {
  auto option = fastdeploy::RuntimeOption();  
  option.UseOpenVINOBackend();  // UseOpenVINOBackend  UseOrtBackend
  auto model = fastdeploy::vision::detection::YOLOv8(model_file, "", option);
//   auto model = fastdeploy::vision::detection::YOLOv8(model_file);
  model.GetPreprocessor().SetSize( {img_size, img_size} );  // {img_size,img_size} is init list, type: const std::vector<int>&
  model.GetPostprocessor().SetConfThreshold(conf_threshold);
  model.GetPostprocessor().SetNMSThreshold(nms_threshold);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<cv::String> image_dirs;
  cv::glob(img_dir, image_dirs);
  system(("mkdir vis"));
  system(("mkdir preds"));

  double duration = 0.0;

  for (const auto& image_dir : image_dirs){
    std::cout << image_dir << std::endl;
    size_t last_slash = image_dir.find_last_of('/');
    std::string basename = image_dir.substr(last_slash + 1);
    std::string basename_nosuffix = basename.substr(0, basename.length()-4);

    auto im = cv::imread(image_dir);

    fastdeploy::vision::DetectionResult res;
    auto start = std::chrono::steady_clock::now();
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    auto end = std::chrono::steady_clock::now();
    duration += std::chrono::duration<double>(end-start).count();

    std::string data = save_res2txt(res);
    std::string txt_path = "preds/"+basename_nosuffix+".txt";
    std::ofstream output(txt_path);
    if (output.is_open()){
        output << data;
        output.close();
    }
    
    std::cout << res.Str() << std::endl;
    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis/"+basename, vis_im); 
  }
  std::cout << "Time taken by inference: " << duration << "ms of " 
            << image_dirs.size() << " images" 
            << "(" << image_dirs.size()/duration*1000 << " fps)" <<std::endl;
  std::cout << "Visualized result saved in ./vis" << std::endl;
}

void GpuInfer(const std::string& model_file, 
              const std::string& img_dir, 
              const float& conf_threshold, 
              const float& nms_threshold,
              const int& img_size) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::detection::YOLOv8(model_file, "", option);
  model.GetPreprocessor().SetSize( {img_size, img_size} ); 
  model.GetPostprocessor().SetConfThreshold(conf_threshold);
  model.GetPostprocessor().SetNMSThreshold(nms_threshold);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<cv::String> image_dirs;
  cv::glob(img_dir, image_dirs);
  system(("mkdir vis"));
  system(("mkdir preds"));

  double duration = 0.0;

  for (const auto& image_dir : image_dirs){
    std::cout << image_dir << std::endl;
    size_t last_slash = image_dir.find_last_of('/');
    std::string basename = image_dir.substr(last_slash + 1);
    std::string basename_nosuffix = basename.substr(0, basename.length()-4);

    auto im = cv::imread(image_dir);

    fastdeploy::vision::DetectionResult res;
    auto start = std::chrono::steady_clock::now();
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    auto end = std::chrono::steady_clock::now();
    duration += std::chrono::duration<double>(end-start).count();

    std::string data = save_res2txt(res);
    std::string txt_path = "preds/"+basename_nosuffix+".txt";
    std::ofstream output(txt_path);
    if (output.is_open()){
        output << data;
        output.close();
    }

    std::cout << res.Str() << std::endl;
    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis/"+basename, vis_im); 
  }
  std::cout << "Time taken by inference: " << duration << "ms of " 
            << image_dirs.size() << " images" 
            << "(" << image_dirs.size()/duration*1000 << " fps)" <<std::endl;
  std::cout << "Visualized result saved in ./vis" << std::endl;
}

void TrtInfer(const std::string& model_file, 
              const std::string& img_dir, 
              const float& conf_threshold, 
              const float& nms_threshold,
              const int& img_size) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  option.SetTrtInputShape("images", {1, 3, img_size, img_size});
  auto model = fastdeploy::vision::detection::YOLOv8(model_file, "", option);
  model.GetPreprocessor().SetSize( {img_size, img_size} ); 
  model.GetPostprocessor().SetConfThreshold(conf_threshold);
  model.GetPostprocessor().SetNMSThreshold(nms_threshold);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<cv::String> image_dirs;
  cv::glob(img_dir, image_dirs);
  system(("mkdir vis"));
  system(("mkdir preds"));

  double duration = 0.0;

  for (const auto& image_dir : image_dirs){
    std::cout << image_dir << std::endl;
    size_t last_slash = image_dir.find_last_of('/');
    std::string basename = image_dir.substr(last_slash + 1);
    std::string basename_nosuffix = basename.substr(0, basename.length()-4);

    auto im = cv::imread(image_dir);

    fastdeploy::vision::DetectionResult res;
    auto start = std::chrono::steady_clock::now();
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    auto end = std::chrono::steady_clock::now();
    duration += std::chrono::duration<double>(end-start).count();

    std::string data = save_res2txt(res);
    std::string txt_path = "preds/"+basename_nosuffix+".txt";
    std::ofstream output(txt_path);
    if (output.is_open()){
        output << data;
        output.close();
    }

    std::cout << res.Str() << std::endl;

    auto vis_im = fastdeploy::vision::VisDetection(im, res);
    cv::imwrite("vis/"+basename, vis_im); 
  }
  std::cout << "Time taken by inference: " << duration << "ms of " 
            << image_dirs.size() << " images" 
            << "(" << image_dirs.size()/duration*1000 << " fps)" <<std::endl;
  std::cout << "Visualized result saved in ./vis" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "Usage: multi_infer_model path/to/model path/to/image run_option conf_threshold nms_threshold img_size, "
                 "e.g ./multi_infer_model ./yolov8s.onnx ./test_dir 0 0.45 0.6 320 "
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2], std::atof(argv[4]), std::atof(argv[5]), std::atoi(argv[6]));
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2], std::atof(argv[4]), std::atof(argv[5]), std::atoi(argv[6]));
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2], std::atof(argv[4]), std::atof(argv[5]), std::atoi(argv[6]));
  }
  return 0;
}