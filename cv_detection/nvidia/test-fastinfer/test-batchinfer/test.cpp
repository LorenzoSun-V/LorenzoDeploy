#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "fastinfer.h"

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
	if(argc < 3) {
		std::cout<<"example: ./binary imagelist.txt .onnx .trt"<<std::endl;
        std::cout<<" imagelist.txt 格式: "<<std::endl;
		std::cout<<" /root/path/1.jpg"<<std::endl; 
        std::cout<<" /root/path/2.jpg"<<std::endl; 
        std::cout<<" ...."<<std::endl; 
		exit(-1);
	}

	const char* pimagelist = argv[1];
	const char* pWeightsfile = argv[2];
	const char* pSerializefile = argv[3];
		
	if(pimagelist==NULL || pWeightsfile == NULL)
	{
	    std::cout<<"input param error!"<<std::endl;
	    return -1;
	}
	
	std::vector<cv::Mat> vframes; 
	std::ifstream inputFile( pimagelist );   
	if (!inputFile) {
        std::cout << "无法打开文件！" << std::endl;
        return 1;
    }
    cv::Mat frame; 
    std::string address;
    while (std::getline(inputFile, address)) {
       std::cout<<address<<std::endl;  

       ReadFrameFromPath(address.c_str(), frame);
       vframes.push_back(frame); // 将图像地址添加到 vector 容器中 
    }
    inputFile.close(); // 关闭文件
    
  	    
	void * pDNNInstance = LoadDeepModelModules(pWeightsfile, pSerializefile, 1, MODEL_CODE_2, RUN_GPU);
    if(pDNNInstance == NULL ){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    }
    
    int framecount = vframes.size();
    
    BatchFrames* pBatchframes = new BatchFrames[framecount];
    if (NULL == pBatchframes) {
        std::cout<<"pBatchframes is NULL!"<<std::endl;
        return -1;
    }
    
    for(int i =0; i < framecount; i++)
    {
        pBatchframes[i].frame =  vframes[i];
    }
    
    
    BatchDetBox *pBatchDetBoxs = new BatchDetBox[framecount];
    if (NULL == pBatchDetBoxs) {
        std::cout<<"pBatchDetBoxs is NULL!"<<std::endl;
        return -1;
    }
        
    for(int i =0; i < framecount; i++)
    {
        DetBox *pdetbox = new DetBox[100];
        if (NULL == pdetbox) {
            std::cout<<"pdetbox is NULL!"<<std::endl;
            return -1;
        }
        memset(pdetbox, 0, sizeof(DetBox)*100);    	 
        pBatchDetBoxs[i].pdetbox = pdetbox;
    }
    
    std::cout<<"Init Finshed!"<<std::endl;  
    while (true)
    {	    
	    double t_detect_start = GetCurrentTimeStampMS();
	    BatchInferenceGetDetectResult(pDNNInstance, pBatchframes, framecount, pBatchDetBoxs); 
	    double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
	    for(int i=0; i < framecount;i++)
	    {
	        std::string imagename = "image"+int2string(i)+".jpg";
	        RectDetectResultForImage(pBatchframes[i].frame, pBatchDetBoxs[i].pdetbox, pBatchDetBoxs[i].ndetcount);   
	        cv::imwrite(imagename, pBatchframes[i].frame);
        }
        break;
    }
    DestoryDeepmodeInstance(&pDNNInstance);	           
    std::cout << "Finish !"<<std::endl;
    return 0;
}
