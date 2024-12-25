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

#include "fastdeploy/vision/perception/paddle3d/caddn/postprocessor.h"

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace perception {

CaddnPostprocessor::CaddnPostprocessor() {}

bool CaddnPostprocessor::Run(const std::vector<FDTensor>& tensors,
                             std::vector<PerceptionResult>* results) {
  results->resize(1);
  (*results)[0].Clear();
  (*results)[0].Reserve(tensors[0].shape[0]);
  if (tensors[0].dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  const float* data_0 = reinterpret_cast<const float*>(tensors[0].Data());
  auto result = &(*results)[0];
  for (int i = 0; i < tensors[0].shape[0] * tensors[0].shape[1]; i += 7) {
    // item 1 ~ 3   :  box3d bottom center x, y, z
    // item 4 ~ 6   :  box3d w, h, l
    // item 7       :  box3d yaw angle
    std::vector<float> vec(data_0 + i, data_0 + i + 7);
    result->boxes.emplace_back(
        std::array<float, 7>{0, 0, 0, 0, vec[3], vec[4], vec[5]});
    result->center.emplace_back(std::array<float, 3>{vec[0], vec[1], vec[2]});
    result->yaw_angle.push_back(vec[6]);
  }
  const float* data_1 = reinterpret_cast<const float*>(tensors[2].Data());
  for (int i = 0; i < tensors[2].shape[0]; i += 1) {
    std::vector<float> vec(data_1 + i, data_1 + i + 1);
    result->scores.push_back(vec[0]);
  }
  const float* data_2 = reinterpret_cast<const float*>(tensors[1].Data());
  for (int i = 0; i < tensors[1].shape[0]; i++) {
    std::vector<float> vec(data_2 + i, data_2 + i + 1);
    result->label_ids.push_back(vec[0]);
  }

  result->valid.push_back(true);   // 0 scores
  result->valid.push_back(true);   // 1 label_ids
  result->valid.push_back(true);   // 2 boxes
  result->valid.push_back(true);   // 3 center
  result->valid.push_back(false);  // 4 observation_angle
  result->valid.push_back(true);   // 5 yaw_angle
  result->valid.push_back(false);  // 6 velocity

  return true;
}

}  // namespace perception
}  // namespace vision
}  // namespace fastdeploy
