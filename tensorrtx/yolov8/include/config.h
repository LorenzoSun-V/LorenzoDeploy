/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 13:45:53
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-04 08:45:41
 * @Description: 
 */
#define USE_FP16
//#define USE_INT8

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";
const static int kNumClass = 2;
const static int kBatchSize = 8;
const static int kGpuId = 0;
const static int kInputH = 640;
const static int kInputW = 640;
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.25f;
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;
