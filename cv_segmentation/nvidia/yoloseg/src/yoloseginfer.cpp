#include "yoloseginfer.h"
#include "yoloseg.h"

class YOLOSegModelParam {
public:
    bool bParamIsOk;

    YOLOSegModel yolosegmodel;
    YOLOSegModelParam() {
    }

    ~YOLOSegModelParam() {
        bParamIsOk =false;
    }
};

class YOLOSegModelInstance {
public:
   std::shared_ptr<YOLOSegModelParam> _param;

   YOLOSegModelInstance() {
        _param = std::make_shared<YOLOSegModelParam>();
    }
};

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if (pWeightsfile == NULL)
    {
        std::cerr << "input pWeightsfile is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    YOLOSegModelInstance* _instance =  new YOLOSegModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init VideoStreamInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }

    bool bOK = _instance->_param->yolosegmodel.loadModel(pWeightsfile);
    if( !bOK ) {
        delete _instance;
        pDeepInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }
    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  
    return ENUM_OK;
}

//GPU推理获得检测结果
ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<SegBox>& detBoxs, std::vector<cv::Mat>& masks)
{
    YOLOSegModelInstance* _instance = static_cast<YOLOSegModelInstance*>(pDeepInstance);          
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "InferenceGetDetectResult pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }

    if (frame.empty())
    {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    if (!_instance->_param->yolosegmodel.inference(frame, detBoxs, masks)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    
    return ENUM_OK;
}  


//多batch推理获得检测结果
ENUM_ERROR_CODE BatchInferenceGetDetectResult(void* pDeepInstance, std::vector<cv::Mat> img_batch, std::vector<std::vector<SegBox>> &batchDetBoxs, std::vector<std::vector<cv::Mat>> &batchMasks)
{
    YOLOSegModelInstance* _instance = static_cast<YOLOSegModelInstance*>(pDeepInstance);  
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "BatchInferenceGetDetectResult pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if ( img_batch.empty() )
    {
        std::cout << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    bool bOK = _instance->_param->yolosegmodel.batch_inferenece(img_batch, batchDetBoxs, batchMasks);
    if(!bOK)
    {
        return ERR_DETECT_OBJECT_EMPTY; 
    }
    return ENUM_OK;
}  

  
ENUM_ERROR_CODE DestoryDeepmodeInstance( void **pDeepInstance)
{
    YOLOSegModelInstance* _instance = static_cast<YOLOSegModelInstance*>(*pDeepInstance);  	 
	    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "DestoryDeepmodeInstance pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }

  	if (_instance)
    {
        delete _instance;
        pDeepInstance = NULL;
    }

    return ENUM_OK;
}