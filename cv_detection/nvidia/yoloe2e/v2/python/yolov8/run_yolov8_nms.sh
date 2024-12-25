#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/yoloe2e/v2/python/yolov8/run_yolov8_nms.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-11-04 17:33:20
### 

input_model=/home/sysadmin/jack/ultralytics/runs/detect/v8s_hw_cls2_320_v0.17_1104/weights/model8_b16s_20241104_cls2_320_v0.17.onnx
outputmodel=/home/sysadmin/jack/ultralytics/runs/detect/v8s_hw_cls2_320_v0.17_1104/weights/model8_b16s_20241104_cls2_320_v0.17_1.onnx
numcls=2
keepTopK=200

python yolov8_add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python yolov8_add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
