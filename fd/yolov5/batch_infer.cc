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
#include <sys/time.h>


double GetCurrentTimeStampMS()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
//export LD_LIBRARY_PATH=/home/ubuntu/lingpu/fastdeploy-linux-x64-1.0.6

void BatchCpuInfer(const std::string& model_file, const std::string& image_file) {
  auto model = fastdeploy::vision::detection::YOLOv5(model_file);
  model.GetPreprocessor().SetSize( {640, 640} );
  model.GetPreprocessor().SetStride(64);


  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  double t_detect_start = GetCurrentTimeStampMS();
  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  double t_detect_end = GetCurrentTimeStampMS();  
  fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);


  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void BatchGpuInfer(const std::string& model_file, const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<cv::String> image_dirs;
  std::vector<cv::Mat> images;
  cv::glob(image_file, image_dirs);
  for (const auto& image_dir : image_dirs){
    auto im = cv::imread(image_dir);
    // cv::resize(im, im, cv::Size(640, 640));
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    int w = im.rows;
    int h = im.cols;
    int c = im.channels();
    std::cout << "w: " << w << " h: " << h << " c: " << c << " path: " << image_dir << std::endl;
    images.emplace_back(im);
  }

  std::vector<fastdeploy::vision::DetectionResult> reses;
  if (!model.BatchPredict(images, &reses)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  std::cout << reses.size() << std::endl;
  for (int i=0; i<reses.size(); i++){
    std::cout << reses[i].Str() << std::endl;
    auto vis_im = fastdeploy::vision::VisDetection(images[i], reses[i]);
    cv::imwrite("vis_result" + std::to_string(i) + ".jpg", vis_im);
  }
//   auto vis_im = fastdeploy::vision::VisDetection(im, res);
//   cv::imwrite("vis_result.jpg", vis_im);
//   std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void BatchTrtInfer(const std::string& model_file, const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  option.SetTrtInputShape("images", {1, 3, 640, 640});
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option, "
                 "e.g ./infer_model ./yolov5.onnx ./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    BatchCpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    BatchGpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 2) {
    BatchTrtInfer(argv[1], argv[2]);
  }
  return 0;
}
