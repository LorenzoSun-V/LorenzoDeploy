/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-23 09:03:15
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-24 10:18:12
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include "utils.h"
#include "common.h"
#include "yolodnninfer.h"

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
          std::cout<<"example: ./binary imagedir .onnx"<<std::endl;
          exit(-1);
    }

    const char* pimagedir = argv[1];
    const char* pWeightsfile = argv[2];
          
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
    ENUM_ERROR_CODE eOK =  LoadDNNModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK && NULL == pDNNInstance){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 
    
    std::cout<<"Init Finshed!"<<std::endl;
    // std::vector<DetBox> detBoxs;
    // InferenceGetDetectResult(pDNNInstance, frame, detBoxs);  
    int frame_num = static_cast<int>(batchframes.size());
    double total_time = 0.0;
    int index=1;
    std::vector<DetBox> detBoxs;
    for(int i=0; i<frame_num; i++)
    {
        frame = batchframes[i];
        detBoxs.clear();
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceDNNGetDetectResult(pDNNInstance, frame, detBoxs);
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

    DestoryDNNModeInstance(&pDNNInstance);	  
    std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    std::cout << "Average fps: " << frame_num / total_time * 1000 << std::endl;         
    std::cout << "Finish !"<<std::endl;
    return 0;
}

