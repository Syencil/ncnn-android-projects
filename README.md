# ncnn-android-projects
## Introduction
Android Demon of mobilev2-yolo5s and retinaface。深度学习模型工程化优化的最后一步。<br>
模型端轻量化，剪枝，蒸馏 ===>[https://github.com/Syencil/mobile-yolov5-pruning-distillation](https://github.com/Syencil/mobile-yolov5-pruning-distillation)<br>
云端和嵌入式端模型转换 ===> [https://github.com/Syencil/tensorRT](https://github.com/Syencil/tensorRT)<br>

### Achieved
1. 实现检测模型的demo，模型为[mobilev2-yolo5s](https://github.com/Syencil/mobile-yolov5-pruning-distillation)和[retinaface](https://github.com/biubug6/Pytorch_Retinaface)
2. 实现模型之间的一键转换，图像大小的一键转换（640 ===> 320）。返回的Toast时间为是从java代码请求C++开始到得到结果的总时间，而benchmark返回的fps仅为infer的时间
3. 目前仅测试过华为P40 pro的CPU端

### Attention
此android只是一个方便展示的前端界面。C++均为本人所写，JAVA大部分参考了其他git。具体优化方式和内容请移步[mobilev2-yolo5s](https://github.com/Syencil/mobile-yolov5-pruning-distillation)
