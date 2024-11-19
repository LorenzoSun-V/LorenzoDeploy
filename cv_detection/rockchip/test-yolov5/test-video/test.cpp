/**
* @file      test.cpp
*
* @brief     精度验证推理代码
*
* @copyright 无锡宝通智能科技股份有限公司
*
* @author  图像算法组-贾俊杰
*
* All Rights Reserved.
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rkinfer.h"
#include "utils.h"

using namespace cv;
using namespace std;

#define SAVE_OUTPUT_FILE
//#define SHOW_DETECT_IMAGE

int main(int argc, char* argv[]) {
    if(argc < 3) {
        cout << "Usage: " << argv[0] << " <model_file> <video_file>" << endl;
        return -1;
    }

    const char* modelPath = argv[1];
    const char* videoPath = argv[2];

    // 初始化模型推理实例
    void* pRkInferInstance = NULL;
    ENUM_ERROR_CODE eRet = InitRKInferenceInstance(modelPath, 2, &pRkInferInstance);
    if(pRkInferInstance == NULL || eRet != ENUM_OK){
        cout<<"can not get pRkInferInstance!"<<endl;
        return -1;
    }

    // 打开视频文件
    VideoCapture cap(videoPath);
    if(!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    // 获取视频帧的尺寸和帧率
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    // 创建视频写入器
    VideoWriter videoWriter("output_video.mp4", VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width, frame_height));

    Mat frame;
    while(true) {
        bool isSuccess = cap.read(frame); // 读取新帧
        if(!isSuccess) break; // 如果读取失败或到达视频末尾，则停止

        std::vector<DetBox> detResult;
        InferenceGetDetectResult(pRkInferInstance, frame, detResult); // 对帧进行推理
        std::cout<<"detResult: "<<detResult.size()<<std::endl;
        DrawRectDetectResultForImage(frame, detResult); // 绘制推理结果

        videoWriter.write(frame); // 将处理后的帧写入视频

#ifdef SHOW_DETECT_IMAGE
        imshow("Frame", frame);
        if(waitKey(25) >= 0) break;
#endif
    }

    // 释放资源
    cap.release();
    videoWriter.release();

#ifdef SHOW_DETECT_IMAGE
    destroyAllWindows();
#endif

    cout << "Finished processing video." << endl;
    return 0;
}



