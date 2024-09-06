#! /bin/bash

input_model=/home/jack/data1/project/yolov9/yolov9-m-converted.onnx
outputmodel=/home/jack/data1/project/yolov9/yolov9-m-converted_1.onnx
numcls=80

python yolov8_add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python yolov8_add_nms.py --model ${outputmodel} --numcls ${numcls}
