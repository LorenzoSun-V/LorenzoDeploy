/*
 * @FilePath: /bt_alg_api/test-yolov5/test-infer/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 16:50:13
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "common.h"
#include "yolov5infer.h"
#include "opencv2/opencv.hpp"
#include "utils.h"

using namespace cv;
using namespace std;

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
	if(argc < 2) {
          std::cout<<"example: ./binary imagedir .bin"<<std::endl;
          exit(-1);
    }

    const char* pimagedir = argv[2];
    const char* pWeightsfile = argv[1];
          
    if(pimagedir == NULL || pWeightsfile == NULL)
    {
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }
	
    std::vector<cv::Mat> batchframes; 
    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
  	cv::Mat frame; 
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        batchframes.push_back(frame);
    }

    void * pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK && NULL == pDNNInstance){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 
    
    std::cout<<"Init Finshed!"<<std::endl;  
    int frame_num = static_cast<int>(batchframes.size());
    double total_time = 0.0;
    int index=1;
    std::vector<DetBox> detBoxs;
    // for( auto& frame: batchframes)
    for (int i=0; i < frame_num; i++)
    {   
        cv::Mat frame = batchframes[i];
        detBoxs.clear();
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDNNInstance, frame, detBoxs);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;
        // 生成新的图像名称，原始名称 + 下横线 + 序号
        std::string baseImageName = getBaseFileName(imagePaths[i]);
        std::string imagename = "_" + baseImageName.substr(0, baseImageName.find_last_of('.')) + ".jpg";
        DrawRectDetectResultForImage(frame, detBoxs);   
        cv::imwrite(imagename, frame);
        index++;
      }

    DestoryDeepmodeInstance(&pDNNInstance);	      
    std::cout << "Total detection time: " << total_time << "ms" <<"frame_num: "<<frame_num<< std::endl;
    std::cout << "Average fps: " << 1000 /(total_time / frame_num) << std::endl;     
    std::cout << "Finish !"<<std::endl;
    return 0;
}
