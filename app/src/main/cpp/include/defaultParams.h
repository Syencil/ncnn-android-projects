// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/10/15

#ifndef NCNN_DEFAULTPARAMS_H
#define NCNN_DEFAULTPARAMS_H

#include "common.h"

std::vector<common::Anchor> initRetinafaceAnchors(){
    std::vector<common::Anchor> anchors;
    common::Anchor anchor;
//    [16, 32, 64, 128, 256, 512]
    anchor.width = anchor.height = 16;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 32;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 64;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 128;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 256;
    anchors.emplace_back(anchor);
    anchor.width = anchor.height = 512;
    anchors.emplace_back(anchor);
    return anchors;
}

std::vector<common::Anchor> initYolov5Anchors(){
    std::vector<common::Anchor> anchors;
    common::Anchor anchor;
    // 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326
    anchor.width = 10;
    anchor.height = 13;
    anchors.emplace_back(anchor);
    anchor.width = 16;
    anchor.height = 30;
    anchors.emplace_back(anchor);
    anchor.width = 32;
    anchor.height = 23;
    anchors.emplace_back(anchor);
    anchor.width = 30;
    anchor.height = 61;
    anchors.emplace_back(anchor);
    anchor.width = 62;
    anchor.height = 45;
    anchors.emplace_back(anchor);
    anchor.width = 59;
    anchor.height = 119;
    anchors.emplace_back(anchor);
    anchor.width = 116;
    anchor.height = 90;
    anchors.emplace_back(anchor);
    anchor.width = 156;
    anchor.height = 198;
    anchors.emplace_back(anchor);
    anchor.width = 373;
    anchor.height = 326;
    anchors.emplace_back(anchor);
    return anchors;
}

void RetinafaceDefaultParams(common::InputParams &inputParams, common::NcnnParams &ncnnParams, common::DetectParams &detectParams){
    inputParams.ImgH = 640;
    inputParams.ImgW = 640;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.rgb = false;
    inputParams.InputTensorNames = std::vector<std::string>{"input0"};
    inputParams.OutputTensorNames = std::vector<std::string>{"output0", "588", "587"};
    inputParams.Mean = std::vector<float>{104.f, 117.f, 123.f};
    inputParams.Var = std::vector<float>{1.f, 1.f, 1.f};

    ncnnParams.FP32 = true;
    ncnnParams.Int8 = false;
    ncnnParams.worker = 1;
    ncnnParams.NcnnWorker = 4;
    ncnnParams.ParamPath = "retinaface.param";
    ncnnParams.ModelPath = "retinaface.bin";

    detectParams.Strides = std::vector<int> {8, 16, 32};
    detectParams.AnchorPerScale = 2;
    detectParams.NumClass = 2;
    detectParams.NMSThreshold = 0.5;
    detectParams.PostThreshold = 0.6;
    detectParams.Anchors = initRetinafaceAnchors();
}

void Mobilev2Yolo5sDefaultParams(common::InputParams &inputParams, common::NcnnParams &ncnnParams, common::DetectParams &detectParams){
    inputParams.ImgH = 320;
    inputParams.ImgW = 320;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    inputParams.rgb = true;
    inputParams.InputTensorNames = std::vector<std::string>{"images"};
    inputParams.OutputTensorNames = std::vector<std::string>{"output", "821", "776"};
    inputParams.Mean = std::vector<float>{0.f, 0.f, 0.f};
    inputParams.Var = std::vector<float>{0.003922, 0.003922, 0.003922};

    ncnnParams.FP32 = true;
    ncnnParams.Int8 = false;
    ncnnParams.worker = 1;
    ncnnParams.NcnnWorker = 4;
    ncnnParams.ParamPath = "mobilev2-yolo5s-fp32.param";
    ncnnParams.ModelPath = "mobilev2-yolo5s-fp32.bin";

    detectParams.Strides = std::vector<int> {8, 16, 32};
    detectParams.AnchorPerScale = 3;
    detectParams.NumClass = 20;
    detectParams.NMSThreshold = 0.5;
    detectParams.PostThreshold = 0.45;
    detectParams.Anchors = initYolov5Anchors();
}


#endif //NCNN_DEFAULTPARAMS_H
