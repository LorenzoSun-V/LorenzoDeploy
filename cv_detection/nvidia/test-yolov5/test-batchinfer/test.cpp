/*
 * @FilePath: /bt_alg_api/test-yolov5/test-batchinfer/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 16:49:52
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>

#include "common.h"
#include "yolov5infer.h"
#include "utils.h"

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
    const char* pWeightsfile= argv[1];
      
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

    //加载模型 	    
    void * pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK && NULL == pDNNInstance){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    }

    std::vector<std::vector<DetBox>> batchDetBoxs;
    std::cout<<"Init Finshed!"<<std::endl;  

    //推理获得结果   
    double t_detect_start = GetCurrentTimeStampMS();
    BatchInferenceGetDetectResult(pDNNInstance, batchframes, batchDetBoxs); 
    double t_detect_end = GetCurrentTimeStampMS();  
    //将结果进行保存   
    int frame_num = static_cast<int>(batchframes.size());  
    for(int i=0; i < frame_num; i++)
    {
        std::string imagename = "image"+int2string(i)+".jpg";
        DrawRectDetectResultForImage(batchframes[i], batchDetBoxs[i]);   
        cv::imwrite(imagename, batchframes[i] );
    }
    double total_time = t_detect_end - t_detect_start;
    DestoryDeepmodeInstance(&pDNNInstance);	  
    fprintf(stdout, "Total detection time %.02lfms\n", total_time);
    std::cout << "Average fps: " << 1000 /(total_time / frame_num) << std::endl;             
    std::cout << "Finish !"<<std::endl;
    return 0;
}

