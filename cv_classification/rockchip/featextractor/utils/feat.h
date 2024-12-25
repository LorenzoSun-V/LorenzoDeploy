#ifndef FEATMODEL_H
#define FEATMODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <vector>
#include "rknn_api.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class FeatExtractModel {
public:
    FeatExtractModel();
    ~FeatExtractModel();

    int loadModel(const char* pWeightfile);
    int inference(cv::Mat frame, std::vector<float>& features);
private:
    int m_iwidth;
    int m_iheight;
    int m_channel;
    int classnum;
    rknn_context ctx;
    rknn_input_output_num io_num;
    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
};

#endif // FeatExtractModel_H
