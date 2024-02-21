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

#pragma once
#include "fastdeploy/vision/common/processors/manager.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace perception {
/*! @brief Preprocessor object for Petr serials model.
 */
class FASTDEPLOY_DECL PetrPreprocessor : public ProcessorManager  {
 public:
  PetrPreprocessor() = default;
  /** \brief Create a preprocessor instance for Petr model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g smoke/infer_cfg.yml
   */
  explicit PetrPreprocessor(const std::string& config_file);

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the preprocess successed, otherwise false
   */
  bool Apply(FDMatBatch* image_batch, std::vector<FDTensor>* outputs);

 protected:
  bool BuildPreprocessPipelineFromConfig();
  std::vector<std::shared_ptr<Processor>> processors_;

  bool disable_permute_ = false;

  bool initialized_ = false;

  std::string config_file_;

  float scale_ = 1.0f;
  std::vector<float> mean_;
  std::vector<float> std_;

  std::vector<float> input_k_data_{
    -1.40307297e-03, 9.07780395e-06, 4.84838307e-01, -5.43047376e-02,
    -1.40780103e-04, 1.25770375e-05, 1.04126692e+00, 7.67668605e-01,
    -1.02884378e-05, -1.41007011e-03, 1.02823459e-01, -3.07415128e-01,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    -9.39000631e-04, -7.65239349e-07, 1.14073277e+00, 4.46270645e-01,
    1.04998052e-03, 1.91798881e-05, 2.06218868e-01, 7.42717385e-01,
    1.48074005e-05, -1.40855671e-03, 7.45946690e-02, -3.16081315e-01,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    -7.0699735e-04, 4.2389297e-07,  -5.5183989e-01, -5.3276348e-01,
    -1.2281288e-03, 2.5626015e-05, 1.0212017e+00, 6.1102939e-01,
    -2.2421273e-05, -1.4170362e-03, 9.3639769e-02, -3.0863306e-01,
    0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
    2.2227580e-03, 2.5312484e-06, -9.7261822e-01, 9.0684637e-02,
    1.9360810e-04, 2.1347081e-05, -1.0779887e+00, -7.9227984e-01,
    4.3742721e-06, -2.2310747e-03, 1.0842450e-01, -2.9406491e-01,
    0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
    5.97175560e-04, -5.88774265e-06, -1.15893924e+00, -4.49921310e-01,
    -1.28312141e-03, 3.58297058e-07, 1.48300052e-01, 1.14334166e-01,
    -2.80917516e-06, -1.41527120e-03, 8.37693438e-02, -2.36765608e-01,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    3.6048229e-04, 3.8333174e-06, 7.9871160e-01, 4.3321830e-01,
    1.3671946e-03, 6.7484652e-06, -8.4722507e-01, 1.9411178e-01,
    7.5027779e-06, -1.4139183e-03, 8.2083985e-02, -2.4505949e-01,
    0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00};
};

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
