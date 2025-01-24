#ifndef YOLOV5MODEL_H
#define YOLOV5MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <vector>
#include "rknn_api.h"
#include "postprocess.h"
#include "common.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class YOLOV5Model {
public:
    YOLOV5Model();
    ~YOLOV5Model();

    ENUM_ERROR_CODE loadModel(const char* pWeightfile, int class_num);
    ENUM_ERROR_CODE inference(cv::Mat frame, std::vector<DetBox>& detBoxs);
private:
    int target_size;
    int width;
    int height;
    int channel;
    int classnum;
    rknn_context ctx;
    rknn_input_output_num io_num;
    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
};

#endif // YOLOV5MODEL_H
