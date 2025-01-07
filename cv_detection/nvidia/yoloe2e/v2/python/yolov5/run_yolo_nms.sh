#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-10-17 15:15:04
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-11-07 14:16:32
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/v2/python/yolov5/run_yolo_nms.sh
### 

input_model=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/runs/xray/model5_b64m_20250106_cls4_xray_v0.5/weights/best.onnx
outputmodel=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/runs/xray/model5_b64m_20250106_cls4_xray_v0.5/weights/best_1.onnx
numcls=4
keepTopK=300

python add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
