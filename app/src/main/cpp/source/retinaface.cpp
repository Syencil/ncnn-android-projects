// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/4

#include "retinaface.h"

Retinaface::Retinaface(common::InputParams inputParams, common::NcnnParams ncnnParams,
                       common::DetectParams detectParams) : Detection(std::move(inputParams), std::move(ncnnParams), std::move(detectParams)){
    
}

common::Image Retinaface::preProcess(JNIEnv* env, const jobject &image, bool rgb) {
    return Detection::preProcess(env, image, rgb);
}

float Retinaface::infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas) {
    return Detection::infer(inputDatas, outputDatas);
}

std::vector<common::Bbox>
Retinaface::postProcess(const std::vector<ncnn::Mat> &outputDatas, int H, int W, float postThres, float nmsThres) {
    assert(mInputParams.OutputTensorNames.size()==3);
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    assert(mInputParams.BatchSize==1);
    std::vector<common::Bbox> bboxes;
    const float *loc = outputDatas[0].channel(0);
    const float *conf = outputDatas[1].channel(0);
    const float *land = outputDatas[2].channel(0);

    unsigned long pos = 0;
    for(unsigned long s=0; s<mDetectParams.Strides.size(); ++s){
        unsigned long stride = mDetectParams.Strides[s];
        unsigned long height = (H - 1) / stride + 1;
        unsigned long width = (W -1) / stride + 1;
        // 并发
//        unsigned long min_threads;
//        if (mNcnnParams.worker<0){
//            const unsigned long min_length = 8;
//            min_threads = (height - 1) / min_length + 1;
//        }else if(mNcnnParams.worker==0){
//            min_threads = 1;
//        }else{
//            min_threads = mNcnnParams.worker;
//        }
//        const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
//        const unsigned long num_threads = std::min(cpu_max_threads !=0 ? cpu_max_threads : 1, min_threads);
//        const unsigned long block_size = height / num_threads;
//        std::vector<std::future<void>> futures(num_threads-1);
        unsigned long block_start = 0;
//        for(auto & future : futures){
//            future = mThreadPool.submit(&Retinaface::postProcessParall, this,  block_start, block_size, width, s, pos, loc, conf, land, postThres, &bboxes);
//            block_start += block_size;
//        }
        this->postProcessParall(block_start, height - block_start, width, s, pos, loc, conf, land, postThres, &bboxes);
//        for(auto & future : futures){
//            future.get();
//        }
        pos += height * width * mDetectParams.AnchorPerScale;
    }
    nms_cpu(bboxes, nmsThres);
    return bboxes;
}

bool Retinaface::initSession(AAssetManager* mgr) {
    return Detection::initSession(mgr);
}

std::vector<common::Bbox> Retinaface::predOneImage(JNIEnv* env, const jobject &image, float postThres, float nmsThres) {
    return Detection::predOneImage(env, image, postThres, nmsThres);
}

void Retinaface::postProcessParall(unsigned long start_h, unsigned long length, unsigned long width,
                                   unsigned long s, unsigned long pos, const float *loc, const float *conf, const float *land, float postThres,
                                   std::vector<common::Bbox> *bboxes) {
    // CHW
    int stride = mDetectParams.Strides[s];
    common::Bbox bbox;
    pos += start_h * width * mDetectParams.AnchorPerScale;
    for(unsigned long h=start_h; h<start_h+length; ++h){
        for(unsigned long w = 0; w<width; ++w){
            for(unsigned long a=0; a<mDetectParams.AnchorPerScale; ++a){
                float score = conf[pos*2+1];
                if(score>=postThres){
                    // bbox
                    float cx_a = (w + 0.5) * stride;
                    float cy_a = (h + 0.5) * stride;
                    float w_a = mDetectParams.Anchors[mDetectParams.AnchorPerScale * s + a].width;
                    float h_a = mDetectParams.Anchors[mDetectParams.AnchorPerScale * s + a].height;
                    float cx_b = cx_a + loc[pos * 4 + 0] * 0.1 * w_a;
                    float cy_b = cy_a + loc[pos * 4 + 1] * 0.1 * h_a;
                    float w_b = w_a * expf(loc[pos * 4 + 2] * 0.2);
                    float h_b = h_a * expf(loc[pos * 4 + 3] * 0.2);
                    bbox.xmin = (cx_b - w_b / 2);
                    bbox.ymin = (cy_b - h_b / 2);
                    bbox.xmax = (cx_b + w_b / 2);
                    bbox.ymax = (cy_b + h_b / 2);
                    bbox.score = score;
                    bbox.cid = 0;
                    this->safePushBack(bboxes, &bbox);
                }
                ++pos;
            }
        }
    }
}

void Retinaface::safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}

bool Retinaface::setNumThread(int num) {
    return Detection::setNumThread(num);
}

bool Retinaface::setNcnnWorkThread(int num) {
    return Detection::setNcnnWorkThread(num);
}

float Retinaface::benchmark() {
    return Detection::benchmark();
}

