#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/yoloe2e/v2/python/yolov8/run_yolov8_nms.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2025-01-09 14:09:18
### 

input_model=/home/sysadmin/lorenzo/bt_repo/ultralytics/runs/xray/model11_b64m_20250109_cls5_xray_v1.0/weights/best.onnx
outputmodel=/home/sysadmin/lorenzo/bt_repo/ultralytics/runs/xray/model11_b64m_20250109_cls5_xray_v1.0/weights/best_1.onnx
numcls=5
keepTopK=300

python yolov8_add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python yolov8_add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
