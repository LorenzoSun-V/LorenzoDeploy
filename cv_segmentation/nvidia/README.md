<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2025-01-06 10:04:40
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-02-20 13:28:05
 * @Description: 
-->

## 英伟达实例分割

### 1. yoloseg和test-yoloseg

#### 导出ONNX模型

基于[tensorrtx](https://github.com/wang-xinyu/tensorrtx)库开发的通用YOLO实例分割代码。运行前需要将pt模型导出为onnx格式再转为engine，同时YOLOv8、YOLOv11等Detect Head的输出维度需要permute，从[1, 8400, n] -> [1, n, 8400]，和YOLOv5对齐。测试代码支持单图和多图推理。

通过`pip install ultralytics`安装ultralytics库，直接修改对应conda环境下`ultralytics/nn/modules/head.py`中的`class Segment(Detect):`类中的`forward`函数:

```
class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        #! source code:
        # return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
        #! for adapting deploy
        return (torch.cat([x, mc], 1).permute(0, 2, 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
```

修改完成后，运行`yolo export model=yolov8s-seg.pt format=onnx`导出onnx模型。

#### 设置环境变量

如果提示`error while loading shared libraries: libopencv_imgproc.so.408: cannot open shared object file: No such file or directory`，则需要：

```
export LD_LIBRARY_PATH=3rdlibs/opencv4.8_x86_64/lib
```

#### 测试单batch推理

```
./build/test-infer/test-infer ${image_folder} ${engine_path} 0
```

- image_folder：图片文件夹路径

- engine_path：engine文件路径

- YOLOv5等anchor-based分割参数为1，YOLOv8/yolov11分割参数为0。

#### 测试多batch推理

```
./build/test-batchinfer/test-batchinfer ${image_folder} ${engine_path} 0
```

- image_folder：图片文件夹路径

- engine_path：多batch engine文件路径

- YOLOv5等anchor-based分割参数为1，YOLOv8/yolov11分割参数为0。