/*
 * @FilePath: /jack/github/bt_alg_api/cv_classification/nvidia/classification/utils/postprocess.h
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-20 15:06:34
 */

#ifndef POSTPROCESS_H
#define POSTPROCESS_H
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include "common.h"

// 定义结构
struct element_t {
    float value;
    int index;
};


void softmax(std::vector<std::vector<float>> output, int classnum, int batchsize, int top_k, std::vector<ClsResult>& cls_rets);

#endif