#include <jni.h>
#include <string>

#include <android/asset_manager_jni.h>
#include <android/log.h>

#include "ncnn/net.h"
#include "defaultParams.h"
#include "ncnn.h"
#include "retinaface.h"
#include "yolov5.h"

static std::shared_ptr<Detection> detection;
//static char *modelName = "retinaface";
static char *modelName = "mobilev2-yolo5s";


extern "C" {
    JNIEXPORT jboolean JNICALL
    Java_com_luozw_detection_Detection_init(JNIEnv *env, jclass, jobject assetManager){
        jboolean status = false;

        // TODO 依据名字选模型
        common::InputParams inputParams;
        common::NcnnParams ncnnParams;
        common::DetectParams detectParams;
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
        if (!std::strcmp(modelName, "retinaface")){
            RetinafaceDefaultParams(inputParams, ncnnParams, detectParams);
            detection = std::shared_ptr<Detection>{new Retinaface(inputParams, ncnnParams, detectParams)};
        }else if(!std::strcmp(modelName, "mobilev2-yolo5s")){
            Mobilev2Yolo5sDefaultParams(inputParams, ncnnParams, detectParams);
            detection = std::shared_ptr<Detection>{new Yolov5(inputParams, ncnnParams, detectParams)};
        }else{
            return status;
        }
        status = detection->initSession(mgr);
        __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", "%s", "INIT success");
        return status;
    }

    JNIEXPORT jboolean JNICALL
    Java_com_luozw_detection_Detection_convert(JNIEnv *env, jclass, jobject assetManager){
        jboolean status = false;
        // TODO 依据名字选模型
        common::InputParams inputParams;
        common::NcnnParams ncnnParams;
        common::DetectParams detectParams;
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
        if (!std::strcmp(modelName, "mobilev2-yolo5s")){
            RetinafaceDefaultParams(inputParams, ncnnParams, detectParams);
            detection = std::shared_ptr<Detection>{new Retinaface(inputParams, ncnnParams, detectParams)};
            modelName = "retinaface";
        }else if(!std::strcmp(modelName, "retinaface")){
            Mobilev2Yolo5sDefaultParams(inputParams, ncnnParams, detectParams);
            detection = std::shared_ptr<Detection>{new Yolov5(inputParams, ncnnParams, detectParams)};
            modelName = "mobilev2-yolo5s";
        }else{
            return false;
        }
        status = detection->initSession(mgr);
        return status;
    }


    JNIEXPORT jobjectArray JNICALL
    Java_com_luozw_detection_Detection_detect(JNIEnv *env, jclass, jobject image, jdouble threshold, jdouble nms_threshold){
        std::vector<common::Bbox>bboxes = detection->predOneImage(env, image, threshold, nms_threshold);

        auto box_cls = env->FindClass("com/luozw/detection/Box");
        auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
        jobjectArray ret = env->NewObjectArray( bboxes.size(), box_cls, nullptr);
        int i=0;
        for(auto &box : bboxes){
            env->PushLocalFrame(1);
            jobject obj = env->NewObject(box_cls, cid,box.xmin,box.ymin,box.xmax,box.ymax,box.cid,box.score);
            obj = env->PopLocalFrame(obj);
            env->SetObjectArrayElement( ret, i++, obj);
        }
        return ret;
    }

    JNIEXPORT jboolean JNICALL
    Java_com_luozw_detection_Detection_setInferThreadNum(JNIEnv *env, jclass, jint num_threads){
        return (jboolean)detection->setNcnnWorkThread(num_threads);
    }

    JNIEXPORT jboolean JNICALL
    Java_com_luozw_detection_Detection_setProcessThreadNum(JNIEnv *env, jclass, jint num_threads){
        return (jboolean)detection->setNumThread(num_threads);
    }

    JNIEXPORT jint JNICALL
    Java_com_luozw_detection_Detection_benchmark(JNIEnv *env, jclass){
        return (jint) (1 / detection->benchmark() + 0.5);
    }

    JNIEXPORT jint JNICALL
    Java_com_luozw_detection_Detection_convertSize(JNIEnv *env, jclass){
        return detection->convertSize();
    }

}
