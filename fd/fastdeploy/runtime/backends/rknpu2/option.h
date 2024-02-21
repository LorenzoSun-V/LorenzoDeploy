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

namespace fastdeploy {
namespace rknpu2 {
typedef enum _rknpu2_cpu_name {
  RK356X = 0, /* run on RK356X. */
  RK3588 = 1, /* default,run on RK3588. */
  UNDEFINED,
} CpuName;

/* The specification of NPU core setting.It has the following choices :
 * RKNN_NPU_CORE_AUTO : Referring to automatic mode, meaning that it will
 * select the idle core inside the NPU.
 * RKNN_NPU_CORE_0 : Running on the NPU0 core.
 * RKNN_NPU_CORE_1: Runing on the NPU1 core.
 * RKNN_NPU_CORE_2: Runing on the NPU2 core.
 * RKNN_NPU_CORE_0_1: Running on both NPU0 and NPU1 core simultaneously.
 * RKNN_NPU_CORE_0_1_2: Running on both NPU0, NPU1 and NPU2 simultaneously.
 */
typedef enum _rknpu2_core_mask {
  RKNN_NPU_CORE_AUTO = 0,
  RKNN_NPU_CORE_0 = 1,
  RKNN_NPU_CORE_1 = 2,
  RKNN_NPU_CORE_2 = 4,
  RKNN_NPU_CORE_0_1 = RKNN_NPU_CORE_0 | RKNN_NPU_CORE_1,
  RKNN_NPU_CORE_0_1_2 = RKNN_NPU_CORE_0_1 | RKNN_NPU_CORE_2,
  RKNN_NPU_CORE_UNDEFINED,
} CoreMask;
}  // namespace rknpu2

struct RKNPU2BackendOption {
  rknpu2::CpuName cpu_name = rknpu2::CpuName::RK3588;
  rknpu2::CoreMask core_mask = rknpu2::CoreMask::RKNN_NPU_CORE_AUTO;
};
}  // namespace fastdeploy
