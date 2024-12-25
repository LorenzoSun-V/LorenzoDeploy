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

#include "fastdeploy/pybind/main.h"

namespace fastdeploy {
void BindAdaFace(pybind11::module& m) {
  pybind11::class_<vision::faceid::AdaFacePreprocessor>(
      m, "AdaFacePreprocessor")
      .def(pybind11::init())
      .def("run", [](vision::faceid::AdaFacePreprocessor& self,
                     std::vector<pybind11::array>& im_list) {
        std::vector<vision::FDMat> images;
        for (size_t i = 0; i < im_list.size(); ++i) {
          images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
        }
        std::vector<FDTensor> outputs;
        if (!self.Run(&images, &outputs)) {
          throw std::runtime_error("Failed to preprocess the input data in AdaFacePreprocessor.");
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
          outputs[i].StopSharing();
        }
        return outputs;
      })
      .def_property("permute", &vision::faceid::AdaFacePreprocessor::GetPermute,
                    &vision::faceid::AdaFacePreprocessor::SetPermute)
      .def_property("alpha", &vision::faceid::AdaFacePreprocessor::GetAlpha,
                    &vision::faceid::AdaFacePreprocessor::SetAlpha)
      .def_property("beta", &vision::faceid::AdaFacePreprocessor::GetBeta,
                    &vision::faceid::AdaFacePreprocessor::SetBeta)
      .def_property("size", &vision::faceid::AdaFacePreprocessor::GetSize,
                    &vision::faceid::AdaFacePreprocessor::SetSize);

  pybind11::class_<vision::faceid::AdaFacePostprocessor>(
      m, "AdaFacePostprocessor")
      .def(pybind11::init())
      .def("run", [](vision::faceid::AdaFacePostprocessor& self, std::vector<FDTensor>& inputs) {
        std::vector<vision::FaceRecognitionResult> results;
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error("Failed to postprocess the runtime result in AdaFacePostprocessor.");
        }
        return results;
      })
      .def("run", [](vision::faceid::AdaFacePostprocessor& self, std::vector<pybind11::array>& input_array) {
        std::vector<vision::FaceRecognitionResult> results;
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error("Failed to postprocess the runtime result in AdaFacePostprocessor.");
        }
        return results;
      })
      .def_property("l2_normalize", &vision::faceid::AdaFacePostprocessor::GetL2Normalize,
                    &vision::faceid::AdaFacePostprocessor::SetL2Normalize);

  pybind11::class_<vision::faceid::AdaFace, FastDeployModel>(
      m, "AdaFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption, ModelFormat>())
      .def("predict", [](vision::faceid::AdaFace& self, pybind11::array& data) {
        cv::Mat im = PyArrayToCvMat(data);
        vision::FaceRecognitionResult result;
        self.Predict(im, &result);
        return result;
      })
      .def("batch_predict", [](vision::faceid::AdaFace& self, std::vector<pybind11::array>& data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        std::vector<vision::FaceRecognitionResult> results;
        self.BatchPredict(images, &results);
        return results;
      })
      .def_property_readonly("preprocessor", &vision::faceid::AdaFace::GetPreprocessor)
      .def_property_readonly("postprocessor", &vision::faceid::AdaFace::GetPostprocessor);
}
}  // namespace fastdeploy
