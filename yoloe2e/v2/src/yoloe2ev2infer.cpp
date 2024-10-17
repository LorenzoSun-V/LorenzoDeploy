#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "yoloe2ev2.h"
#include "yoloe2ev2infer.h"

class YOLOE2Ev2ModelParam {
public:
    bool bParamIsOk;

    YOLOE2Ev2ModelManager yoloe2ev2model;

    YOLOE2Ev2ModelParam() 
    {
	    bParamIsOk = false;	
    }
};

class YOLOE2Ev2ModelInstance {
public:
   std::shared_ptr<YOLOE2Ev2ModelParam> _param;

   YOLOE2Ev2ModelInstance() {
        _param = std::make_shared<YOLOE2Ev2ModelParam>();
    }
};

bool checkFileExists(const char* pWeightsfile) {
    struct stat buffer;
    return (stat(pWeightsfile, &buffer) == 0);
}

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if ( nullptr == pWeightsfile)
    {
        std::cout << "LoadDeepModelModules input weights file is nullptr !" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    if (!checkFileExists(pWeightsfile)) {
        std::cerr << "Error: File " << pWeightsfile << " does not exist!" << std::endl;
        return ERR_MODEL_INPUTPATH_NOT_EXIST;
    }

    //load dnn model     
    YOLOE2Ev2ModelInstance* _instance = new YOLOE2Ev2ModelInstance();
    if (nullptr  == _instance) {
         std::cout <<  "Init YOLOE2Ev2ModelcInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }

    if (!_instance->_param->yoloe2ev2model.loadModel(pWeightsfile)){
        delete _instance;
        pDeepInstance = nullptr ;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }
    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    YOLOE2Ev2ModelInstance* _instance = static_cast<YOLOE2Ev2ModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "InferenceGetDetectResult pDeepInstance is nullptr " << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty())
    {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    if (!_instance->_param->yoloe2ev2model.inference(frame, detBoxs)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}
  
ENUM_ERROR_CODE DestoryDeepmodeInstance(void **pDeepInstance)
{
    YOLOE2Ev2ModelInstance* _instance = static_cast<YOLOE2Ev2ModelInstance*>(*pDeepInstance); 	 
	if (_instance == nullptr || !_instance->_param->bParamIsOk) {
	    std::cout << "DestoryDeepmodeInstance pDeepInstance is nullptr " << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        *pDeepInstance = nullptr;   
    }
        
    return ENUM_OK;
}

