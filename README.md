# tensorrt-deploy
使用tensorrt部署onnx模型    
以部署yolov5示例，展示从导出onnx开始，到部署流程，部署方法为生产者消费者模型
# 部署流程
    * 导出onnx模型
    * pipeline编写
    * tensorrt加载模型
    * 优化tensorrt

## 1. 导出onnx模型
以目标检测中的[yoloV5](https://github.com/ultralytics/yolov5)为例，将pt文件导出为onnx文件，模型文件使用 nano model yolov5n.pt

* input size: 1x3x640x640
* output size: 1x25200x85
* dataset: coco  

```python
python export.py --weights yolov5n.pt --include onnx  --opset=11 --batch-size=5
```


## 2. pipeline编写
## 3. tensorrt加载模型
## 4. 优化tensorrt