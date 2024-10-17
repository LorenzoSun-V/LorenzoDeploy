#! /bin/bash

input_model=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/weights/yolov5m_b4.onnx
outputmodel=/home/sysadmin/lorenzo/bt_repo/yolov5/yolov5-v7.0/weights/yolov5m_b4_1.onnx
numcls=80
keepTopK=300

python add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
