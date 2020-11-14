// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/3

#include "ncnn.h"
#include <android/log.h>
#include <string>
#include <include/ncnn/option.h>
#include <ncnn/cpu.h>

Ncnn::Ncnn(common::InputParams inputParams, common::NcnnParams ncnnParams) :
                       mInputParams(std::move(inputParams)), mNcnnParams(std::move(ncnnParams)){
    mNet = std::make_shared<ncnn::Net>();
    if (!mNet){
        std::cerr << "Create ncnn net failed!" << std::endl;
        exit(1);
    }
}

bool Ncnn::constructNetwork(const std::string& param, const std::string& model) {
    CHECK(mNet->load_param(param.c_str()));
    CHECK(mNet->load_model(model.c_str()));
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = mNcnnParams.NcnnWorker;
    this->mNet->opt = opt;
    return true;
}

bool Ncnn::constructNetwork(AAssetManager* mgr, const std::string& param, const std::string& model) {
    ncnn::Option opt;
    ncnn::set_cpu_powersave(2);
    opt.lightmode = true;
    opt.num_threads = mNcnnParams.NcnnWorker;
    this->mNet->opt = opt;
    CHECK(mNet->load_param(mgr, param.c_str()));
    CHECK(mNet->load_model(mgr, model.c_str()));
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", "%s", "INIT NCNN success");
    return true;
}

common::Image Ncnn::preProcess(JNIEnv *env, const jobject &image, bool rgb){
    AndroidBitmapInfo image_size;
    AndroidBitmap_getInfo(env, image, &image_size);
    common::Image imageDatas;
    imageDatas.oh = image_size.height;
    imageDatas.ow = image_size.width;
    float scale = std::min(static_cast<float>(mInputParams.ImgH) / static_cast<float>(imageDatas.oh),
                           static_cast<float>(mInputParams.ImgW) / static_cast<float>(imageDatas.ow));

    imageDatas.nh = static_cast<int>(scale * static_cast<float>(imageDatas.oh));
    imageDatas.nw = static_cast<int>(scale * static_cast<float>(imageDatas.ow));
    auto code = rgb ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR;
    imageDatas.image = ncnn::Mat::from_android_bitmap_resize(env, image, code, imageDatas.nw, imageDatas.nh);
    imageDatas.image.substract_mean_normalize(mInputParams.Mean.data(), mInputParams.Var.data());
    return imageDatas;
}

float Ncnn::infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas){assert(mInputParams.InputTensorNames.size()==1);

    const auto t_start = std::chrono::high_resolution_clock::now();
    auto mExtractor = mNet -> create_extractor();
    CHECK(mExtractor.input(mInputParams.InputTensorNames[0].c_str(), inputDatas));
    for (int i=0; i<mInputParams.OutputTensorNames.size(); ++i){
        CHECK(mExtractor.extract(mInputParams.OutputTensorNames[i].c_str(), outputDatas[i]));
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const float elapsed_time = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    return elapsed_time;
}


Detection::Detection(common::InputParams inputParams, common::NcnnParams ncnnParams, common::DetectParams detectParams) :
                Ncnn(std::move(inputParams), std::move(ncnnParams)), mDetectParams(std::move(detectParams)){

}


void Detection::transformBbox(const int &oh, const int &ow, const int &nh, const int &nw, std::vector<common::Bbox> &bboxes) {
    float scale = std::max(static_cast<float>(ow) / static_cast<float>(nw), static_cast<float>(oh) / static_cast<float>(nh));
    for (auto &bbox : bboxes){
        bbox.xmin = bbox.xmin * scale;
        bbox.ymin = bbox.ymin * scale;
        bbox.xmax = bbox.xmax * scale;
        bbox.ymax = bbox.ymax * scale;
    }
}

bool Detection::initSession(AAssetManager* mgr) {
    auto status =  Ncnn::constructNetwork(mgr, mNcnnParams.ParamPath, mNcnnParams.ModelPath);
    return status;
}

common::Image Detection::preProcess(JNIEnv *env, const jobject &image, bool rgb) {
    return Ncnn::preProcess(env, image, rgb);
}

float Detection::infer(const ncnn::Mat &inputDatas, std::vector<ncnn::Mat> &outputDatas) {
    return Ncnn::infer(inputDatas, outputDatas);
}

std::vector<common::Bbox> Detection::predOneImage(JNIEnv* env, const jobject &image, float postThres, float nmsThres) {
    assert(mInputParams.BatchSize==1);
    assert(mInputParams.InputTensorNames.size()==1);

    std::vector<ncnn::Mat> outputDatas(mInputParams.OutputTensorNames.size());

    auto imageDatas = this->preProcess(env, image, mInputParams.rgb);
    float elapsedTime = this->infer(imageDatas.image, outputDatas);
    std::string st = "infer time is " + std::to_string(elapsedTime);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", "%s", st.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<common::Bbox> bboxes = this->postProcess(outputDatas, imageDatas.nh, imageDatas.nw, postThres, nmsThres);
    auto t2 = std::chrono::high_resolution_clock::now();
    st = "Post time is " + std::to_string(std::chrono::duration<float, std::milli>(t2-t1).count());
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", "%s", st.c_str());
    this->transformBbox(imageDatas.oh, imageDatas.ow, imageDatas.nh, imageDatas.nw, bboxes);

    return bboxes;
}

bool Detection::setNumThread(int num) {
    mNcnnParams.worker = num;
    return true;
}

bool Detection::setNcnnWorkThread(int num) {
    mNcnnParams.NcnnWorker = num;
    return true;
}

float Detection::benchmark(){
    ncnn::Mat in(mInputParams.ImgW, mInputParams.ImgH, mInputParams.ImgC);
    in.fill(0.01f);
    std::vector<ncnn::Mat> outputDatas(mInputParams.OutputTensorNames.size());
    for(int i=0; i<2; ++i){
        infer(in, outputDatas);
    }
    float total_time = 0;
    for(int i=0; i<10; ++i){
        total_time += infer(in, outputDatas);
    }
    return total_time / 1000 / 10;
}

int Detection::convertSize() {
    if(mInputParams.ImgH == 640){
        mInputParams.ImgH = 320;
        mInputParams.ImgW = 320;
        return 320;
    }else if(mInputParams.ImgH == 320){
        mInputParams.ImgH = 640;
        mInputParams.ImgW = 640;
        return 640;
    }
}






