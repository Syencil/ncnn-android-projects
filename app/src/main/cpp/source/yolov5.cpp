// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/10/15

#include "yolov5.h"

Yolov5::Yolov5(common::InputParams inputParams, common::NcnnParams ncnnParams, common::DetectParams detectParams)
        : Detection(std::move(inputParams), std::move(ncnnParams), std::move(detectParams)) {

}

void Yolov5::safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}

void Yolov5::postProcessParall(unsigned long start, unsigned long length, unsigned long height, unsigned long width, int nh, int nw,
                               int scale_idx, float postThres, const float *origin_output,
                               std::vector<common::Bbox> *bboxes) {
    common::Bbox bbox;
    float cx, cy, w_b, h_b, score;
    int cid;
    unsigned long pos = start * width * mDetectParams.AnchorPerScale * (5+mDetectParams.NumClass);
    const float *ptr = origin_output + pos;
    for(unsigned long h=start; h<start+length; ++h){
        for(unsigned long w=0; w<width; ++w){
            for(unsigned long a=0; a<mDetectParams.AnchorPerScale; ++a){
                const float *cls_ptr =  ptr + 5;
                cid = argmax(cls_ptr, cls_ptr+mDetectParams.NumClass);
                score = sigmoid(ptr[4]) * sigmoid(cls_ptr[cid]);
                if(score>=postThres){
                    cx = (sigmoid(ptr[0]) * 2.f - 0.5f + static_cast<float>(w)) * static_cast<float>(mDetectParams.Strides[scale_idx]);
                    cy = (sigmoid(ptr[1]) * 2.f - 0.5f + static_cast<float>(h)) * static_cast<float>(mDetectParams.Strides[scale_idx]);
                    w_b = powf(sigmoid(ptr[2]) * 2.f, 2) * mDetectParams.Anchors[scale_idx * mDetectParams.AnchorPerScale + a].width;
                    h_b = powf(sigmoid(ptr[3]) * 2.f, 2) * mDetectParams.Anchors[scale_idx * mDetectParams.AnchorPerScale + a].height;
                    bbox.xmin = clip(cx - w_b / 2, 0.f, static_cast<float>(nw - 1));
                    bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(nh - 1));
                    bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(nw - 1));
                    bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(nh - 1));
                    bbox.score = score;
                    bbox.cid = cid;
                    this->safePushBack(bboxes, &bbox);
                }
                ptr += 5 + mDetectParams.NumClass;
            }
        }
    }
}



common::Image Yolov5::preProcess(JNIEnv* env, const jobject &image, bool rgb) {
    return Detection::preProcess(env, image, rgb);
}

float Yolov5::infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas) {
    return Detection::infer(inputDatas, outputDatas);
}

std::vector<common::Bbox>
Yolov5::postProcess(const std::vector<ncnn::Mat> &outputDatas, int H, int W, float postThres, float nmsThres) {
    if(postThres<0){
        postThres = mDetectParams.PostThreshold;
    }
    if(nmsThres<0){
        nmsThres = mDetectParams.NMSThreshold;
    }
    assert(mInputParams.BatchSize==1);
    std::vector<common::Bbox> bboxes;
    // 并发执行
    for (int scale_idx=0; scale_idx<mInputParams.OutputTensorNames.size(); ++scale_idx) {
        const int stride = mDetectParams.Strides[scale_idx];
//        const int width = (W +stride-1)/ stride;
//        const int height = (H +stride-1) / stride;
        const int height = outputDatas[scale_idx].h;
        const int width = outputDatas[scale_idx].w;

        __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", "%d, %d, %d ", outputDatas[scale_idx].c, outputDatas[scale_idx].h , outputDatas[scale_idx].w );
//        std::cout << outputDatas[scale_idx].c << outputDatas[scale_idx].h << outputDatas[scale_idx].w << std::endl;

        int nw = W;
        int nh = H;
        ncnn::Mat output_mat = outputDatas[scale_idx];
        float *ptr[output_mat.c];
        for(int i=0; i<output_mat.c; ++i){
            ptr[i] = output_mat.channel(i);
        }
        for(int h=0; h<height; ++h){
            int shift = h * width;
            for(int w=0; w<width; ++w){
                int pos = shift + w;
                for(int a=0; a<mDetectParams.AnchorPerScale; ++a){
                    int a_s = a * (5 + mDetectParams.NumClass);
                    float s = sigmoid(ptr[a_s+4][pos]);
                    for(int c=5; c<(5+ mDetectParams.NumClass); ++c){
                        float score = s * sigmoid(ptr[a_s+c][pos]);
                        if(score>=postThres){
                            common::Bbox bbox;
                            float cx = ((sigmoid(ptr[a_s][pos])) * 2.f - 0.5f + w) * stride;
                            float cy = ((sigmoid(ptr[a_s+1][pos])) * 2.f - 0.5f + h) * stride;
                            float w_b = powf(sigmoid(ptr[a_s+2][pos]) * 2.f, 2) * mDetectParams.Anchors[scale_idx * mDetectParams.AnchorPerScale + a].width;
                            float h_b = powf(sigmoid(ptr[a_s+3][pos]) * 2.f, 2) * mDetectParams.Anchors[scale_idx * mDetectParams.AnchorPerScale + a].height;
                            bbox.xmin = clip(cx - w_b / 2, 0.f, static_cast<float>(nw - 1));
                            bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(nh - 1));
                            bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(nw - 1));
                            bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(nh - 1));
                            bbox.score = score;
                            bbox.cid = c-5;
//                            bprint(bbox);
                            this->safePushBack(&bboxes, &bbox);
                        }
                    }
                }
            }
        }


        // NCNN decode
//        unsigned long min_threads;
//        if (mNcnnParams.worker < 0) {
//            const unsigned long min_length = 64;
//            min_threads = (height - 1) / min_length + 1;
//        } else if (mNcnnParams.worker == 0) {
//            min_threads = 1;
//        } else {
//            min_threads = mNcnnParams.worker;
//        }
//        const unsigned long cpu_max_threads = std::thread::hardware_concurrency();
//        const unsigned long num_threads = std::min(cpu_max_threads != 0 ? cpu_max_threads : 1, min_threads);
//        const unsigned long block_size = height / num_threads;
//        std::vector<std::future<void>> futures (num_threads - 1);
//        unsigned long block_start = 0;
//        for (auto &future : futures) {
//            future = mThreadPool.submit(&Yolov5::postProcessParall, this, block_start, block_size, height, width, H, W, scale_idx, postThres, origin_output, &bboxes);
//            block_start += block_size;
//        }
//        this->postProcessParall(block_start, height-block_start, height, width, H, W, scale_idx, postThres, origin_output, &bboxes);
//        for (auto &future : futures){
//            future.get();
//        }
    }

    nms_cpu(bboxes, nmsThres);
    return bboxes;
}

bool Yolov5::initSession(AAssetManager *mgr) {
    return Detection::initSession(mgr);
}

std::vector<common::Bbox> Yolov5::predOneImage(JNIEnv* env, const jobject &image, float postThres, float nmsThres) {
    return Detection::predOneImage(env, image, postThres, nmsThres);
}

bool Yolov5::setNumThread(int num) {
    return Detection::setNumThread(num);
}

bool Yolov5::setNcnnWorkThread(int num) {
    return Detection::setNcnnWorkThread(num);
}

float Yolov5::benchmark() {
    return Detection::benchmark();
}
