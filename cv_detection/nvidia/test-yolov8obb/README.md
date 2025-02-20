<!--
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/test-yolov8obb/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2025-02-20 13:15:43
-->

## 导出ONNX模型

运行前需要将pt模型导出为onnx格式再转为engine，YOLOv8的OBB Detect Head的输出维度需要permute：

通过`pip install ultralytics`安装ultralytics库，直接修改对应conda环境下`ultralytics/nn/modules/head.py`中的`class OBB(Detect):`类中的`forward`函数:

```
class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        #! source code:
        # return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))
        #! for adapting deploy code
        return torch.cat([x, angle], 1).permute(0, 2, 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))
```

修改完成后，运行`yolo export model=yolov8s-obb.pt format=onnx`导出onnx模型。

## 设置环境变量

如果提示`error while loading shared libraries: libopencv_imgproc.so.408: cannot open shared object file: No such file or directory`，则需要：

```
export LD_LIBRARY_PATH=${TensorRT_Path}/lib
```

## 测试单batch推理

```
./test-infer ${engine_path} ${image_folder}
```

- engine_path：engine文件路径

- image_folder：图片文件夹路径

## 测试多batch推理

```
./test-batchinfer ${engine_path} ${image_folder}
```

- engine_path：engine文件路径

- image_folder：图片文件夹路径

## 验证模型精度

```
./test-precious ${engine_path} ${image_folder}
```

- engine_path：engine文件路径

- image_folder：验证集/测试集图片文件夹路径