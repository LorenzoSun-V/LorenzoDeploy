/*
 * @FilePath: /bt_alg_api/test-rkinfer/test-infer/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 18:06:05
 */
/**
* @file      test.cpp
*
* @brief     单batch测试代码
*
* @copyright 无锡宝通智能科技股份有限公司
*
* @author  图像算法组-贾俊杰
*
* All Rights Reserved.
*/
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "common.h"
#include "featinfer.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 3){
		cout<<"param1: the make binary;"<<endl;
		cout<<"param2: model path is the model and prototxt path;"<<endl;
		cout<<"param3: input the detect image file;"<<endl;
		cout<<"example: ./binary weightsfile imagepath"<<endl;
		exit(-1);
 	}

    const char* pWeightsfile = argv[1];
    const char* pVideofile = argv[2];

    //读图像
	cv::Mat frame = cv::imread(pVideofile);
	if( frame.empty() )
	{
		std::cout<<"image is empty!"<<std::endl;
		return -1;
	}
    
    //初始化句柄    
    void * pRkInferInstance = NULL;
    ENUM_ERROR_CODE eRet = LoadDeepModelModules(pWeightsfile, &pRkInferInstance);
    if(pRkInferInstance == NULL || eRet != ENUM_OK){
        cout<<"can not get pRkInferInstance!"<<endl;
        return -1;
    }

    std::vector<float> detResult;
    cout<<"Init Finshed!"<<endl;  
    detResult.clear();
    //推理获得检测的目标及数量
    InferenceGetFeature(pRkInferInstance, frame, detResult);


    DestoryDeepmodeInstance(&pRkInferInstance);           
    std::cout << "Finish !"<<endl;
    return 0;
}
