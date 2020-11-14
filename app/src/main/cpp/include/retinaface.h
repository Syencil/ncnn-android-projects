// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/3

#ifndef NCNN_RETINAFACE_H
#define NCNN_RETINAFACE_H

#include <cassert>
#include <thread>
#include <mutex>

#include "utils.h"
#include "ncnn.h"


class Retinaface : public Detection{
private:
    std::mutex mMutex;

    void safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox);

    void postProcessParall(unsigned long start_h, unsigned long length, unsigned long width,
                           unsigned long s, unsigned long pos, const float *loc, const float *conf, const float *land, float postThres,
                           std::vector<common::Bbox> *bboxes);

public:
    Retinaface(common::InputParams inputParams, common::NcnnParams ncnnParams, common::DetectParams detectParams);

    common::Image preProcess(JNIEnv* env, const jobject &image, bool rgb) override ;

    float infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas) override ;

    std::vector<common::Bbox> postProcess(const std::vector<ncnn::Mat> &outputDatas, int H, int W, float postThres, float nmsThres) override;

    bool initSession(AAssetManager* mgr) override;

    std::vector<common::Bbox> predOneImage(JNIEnv* env, const jobject &image, float postThres=-1, float nmsThres=-1) override;

    bool setNumThread(int num) override;

    bool setNcnnWorkThread(int num) override;

    float benchmark() override ;

};


#endif //NCNN_RETINAFACE_H
