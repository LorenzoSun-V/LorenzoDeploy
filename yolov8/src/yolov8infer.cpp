/*
 * @FilePath: /bt_zs_4x_api/yolov8/src/yolov8infer.cpp
 * @Description:  yolov8模型推理接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-07-04 11:47:17
 */

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "cv_utils.h"
#include "utils.h"
#include "yolov8.h"
#include "yolov8infer.h"


class YOLOV8ModelParam {
public:
    bool bParamIsOk;

    YOLOV8ModelManager yolov8model;

    YOLOV8ModelParam() 
    {
	    bParamIsOk = false;	
    }

};

class YOLOV8ModelInstance {
public:
   std::shared_ptr<YOLOV8ModelParam> _param;

   YOLOV8ModelInstance() {
        _param = std::make_shared<YOLOV8ModelParam>();
    }
};

bool checkFileExists(const char* pWeightsfile) {
    struct stat buffer;
    return (stat(pWeightsfile, &buffer) == 0);
}

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if (pWeightsfile == NULL)
    {
        std::cerr << "input pWeightsfile is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }
    if (!checkFileExists(pWeightsfile)) {
        std::cerr << "Error: File " << pWeightsfile << " does not exist!" << std::endl;
        return ERR_MODEL_INPUTPATH_NOT_EXIST;
    }

    //load dnn model     
    YOLOV8ModelInstance* _instance =  new YOLOV8ModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init YOLOV8ModelInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    bool bOk = _instance->_param->yolov8model.loadModel(pWeightsfile);
    if( !bOk ) {
        delete _instance;
        pDeepInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }

    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    YOLOV8ModelInstance* _instance = static_cast<YOLOV8ModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "InferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty()) {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    bool bOK = _instance->_param->yolov8model.inference(frame, detBoxs);
    if(!bOK) {
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}  


ENUM_ERROR_CODE BatchInferenceGetDetectResult(void* pDeepInstance, std::vector<cv::Mat> img_batch, std::vector<std::vector<DetBox>> &batchDetBoxs )
{
    YOLOV8ModelInstance* _instance = static_cast<YOLOV8ModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "BatchInferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if ( img_batch.empty() )
    {
        std::cout << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    _instance->_param->yolov8model.batchInference(img_batch, batchDetBoxs);

    return ENUM_OK;
}  

  
ENUM_ERROR_CODE DestoryDeepmodeInstance( void *pDeepInstance)
{
    YOLOV8ModelInstance* _instance = static_cast<YOLOV8ModelInstance*>(pDeepInstance); 	 
	if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "DestoryDeepmodeInstance pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        pDeepInstance = NULL;
    }
         

    return ENUM_OK;
}

