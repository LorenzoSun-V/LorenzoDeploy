#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-10-17 15:15:04
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2025-01-22 08:21:45
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/v2/python/yolov5/run_yolo_nms.sh
### 

input_model=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/runs/xray/model5_b128m_20250121_cls4_xray_v0.6/weights/best.onnx
outputmodel=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/runs/xray/model5_b128m_20250121_cls4_xray_v0.6/weights/best_b1.onnx
numcls=4
keepTopK=300

python add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
