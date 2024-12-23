#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-10-17 15:15:04
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-11-07 14:16:32
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/v2/python/yolov5/run_yolo_nms.sh
### 

input_model=models/yolov5m_b1.onnx
outputmodel=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/weights/yolov5m_b4_1.onnx
numcls=80
keepTopK=300

python add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
