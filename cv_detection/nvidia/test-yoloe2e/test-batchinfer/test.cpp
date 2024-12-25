#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "utils.h"
#include "common.h"
#ifdef E2E_V1
    #include "yoloe2einfer.h"
#else
    #include "yoloe2ev2infer.h"
#endif


std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
        std::cout<<"example: ./binary image_folder .engine"<<std::endl;
        exit(-1);
    }

    const char* pWeightsfile = argv[1];
    const char* pimagedir = argv[2];

    if(pimagedir == NULL || pWeightsfile == NULL) {
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> vframes; 
    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
    cv::Mat frame; 
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        vframes.push_back(frame);
    }

    void * pDeepInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDeepInstance);
    if(eOK != ENUM_OK) {
        std::cout<<"can not get pDeepInstance!"<<std::endl;
        return -1;
    } 

    std::cout<<"Init Finshed!"<<std::endl;  

    int frame_num = static_cast<int>(vframes.size());
    std::vector<std::vector<DetBox>> batchdetBoxs;
    double t_detect_start = GetCurrentTimeStampMS();
    BatchInferenceGetDetectResult(pDeepInstance, vframes, batchdetBoxs);
    double t_detect_end = GetCurrentTimeStampMS();  
    fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
    std::cout << "size="<<batchdetBoxs.size()<<std::endl;
    std::cout << "Total images: " << batchdetBoxs.size() << std::endl;
    for (size_t i = 0; i < batchdetBoxs.size(); ++i) {
        std::cout << "Image " << i << " has " << batchdetBoxs[i].size() << " detections." << std::endl;
    }

    for (int i=0; i <frame_num && batchdetBoxs.size()>0; i++){
        std::string imagename = "image"+int2string(i)+".jpg";
        DrawRectDetectResultForImage(vframes[i], batchdetBoxs[i]);   
        cv::imwrite(imagename, vframes[i]);
    }
   
    std::cout << "Finish !"<<std::endl;
    DestoryDeepmodeInstance(&pDeepInstance);
    return 0;
}
