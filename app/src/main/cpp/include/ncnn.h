// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/3

#ifndef NCNN_NCNN_H
#define NCNN_NCNN_H

#include <string>
#include <chrono>
#include <vector>
#include <iostream>
#include <memory>
#include <android/log.h>
#include <arm_neon.h>

#include "ncnn/net.h"
#include "thread_safety_stl.h"
#include "common.h"
#include "utils.h"

class Ncnn{
protected:
    std::shared_ptr<ncnn::Net> mNet;
    common::InputParams mInputParams;
    common::NcnnParams mNcnnParams;
//    tss::thread_pool mThreadPool;

protected:
    Ncnn(common::InputParams inputParams, common::NcnnParams ncnnParams);

    bool constructNetwork(const std::string& param, const std::string& model);

    bool constructNetwork(AAssetManager* mgr, const std::string& param, const std::string& model);

    virtual common::Image preProcess(JNIEnv* env, const jobject &image, bool rgb);

    virtual float infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas);

};

class Detection : protected Ncnn{
protected:
    common::DetectParams mDetectParams;
public:
    void transformBbox(const int &oh, const int &ow, const int &nh, const int &nw, std::vector<common::Bbox> &bboxes);

    Detection(common::InputParams inputParams, common::NcnnParams ncnnParams, common::DetectParams detectParams);

    virtual common::Image preProcess(JNIEnv* env, const jobject &image, bool rgb);

    virtual float infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas);

    virtual std::vector<common::Bbox> postProcess(const std::vector<ncnn::Mat> &outputDatas, int H, int W, float postThres, float nmsThres) = 0;

    virtual bool initSession(AAssetManager* mgr);

    virtual std::vector<common::Bbox> predOneImage(JNIEnv* env, const jobject &image, float postThres, float nmsThres);

    virtual bool setNumThread(int num);

    virtual bool setNcnnWorkThread(int num);

    virtual float benchmark();

    virtual int convertSize();
};


#endif //NCNN_NCNN_H
