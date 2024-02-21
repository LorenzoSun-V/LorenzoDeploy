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

#include "fastdeploy/vision/faceid/contrib/insightface/base.h"

namespace fastdeploy {
namespace vision {
namespace faceid {

InsightFaceRecognitionBase::InsightFaceRecognitionBase(
    const std::string& model_file, const std::string& params_file,
    const fastdeploy::RuntimeOption& custom_option,
    const fastdeploy::ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
  }
  valid_rknpu_backends = {Backend::RKNPU2};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
}

bool InsightFaceRecognitionBase::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool InsightFaceRecognitionBase::Predict(const cv::Mat& im,
                                         FaceRecognitionResult* result) {
  std::vector<FaceRecognitionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool InsightFaceRecognitionBase::BatchPredict(
    const std::vector<cv::Mat>& images,
    std::vector<FaceRecognitionResult>* results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  FDASSERT(images.size() == 1, "Only support batch = 1 now.");
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy
