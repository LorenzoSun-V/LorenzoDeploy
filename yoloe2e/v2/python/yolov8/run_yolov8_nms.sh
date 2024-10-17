#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/yoloe2e/v2/python/yolov8/run_yolov8_nms.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-10-17 10:58:44
### 

input_model=/home/sysadmin/lorenzo/bt_repo/ultralytics/weights/yolov8m_b4.onnx
outputmodel=/home/sysadmin/lorenzo/bt_repo/ultralytics/weights/yolov8m_b4_1.onnx
numcls=80
keepTopK=300

python yolov8_add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python yolov8_add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
