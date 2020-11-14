// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/13

#ifndef NCNN_COMMON_H
#define NCNN_COMMON_H

#include <vector>
#include <string>
#include <iostream>
#include "ncnn/net.h"

#define CHECK( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(int err, const char *file, int line ) {
    if (err != 0) {
        printf( "Error code %d in %s at line %d\n", err,
                file, line );
        exit( EXIT_FAILURE );
    }
}

namespace common{
    // <============== Params =============>
    struct InputParams {
        // General
        int ImgH;
        int ImgW;
        int ImgC;
        int BatchSize;
        bool rgb;
        // Tensor
        std::vector<std::string> InputTensorNames;
        std::vector<std::string> OutputTensorNames;
        // Image pre-process function
        std::vector<float> Mean;
        std::vector<float> Var;
    };

    struct NcnnParams{
        bool FP32;
        bool Int8;
        int worker;
        int NcnnWorker;
        std::string ParamPath;
        std::string ModelPath;
        std::string MemPath;
    };

    struct Anchor{
        float width;
        float height;
    };

    struct DetectParams{
        // Detection/Segmentation
        std::vector<int> Strides;
        std::vector<common::Anchor> Anchors;
        int AnchorPerScale;
        int NumClass;
        float NMSThreshold;
        float PostThreshold;
    };

    struct KeypointParams{
        // Hourglass
        int HeatMapH;
        int HeatMapW;
        int NumClass;
        float PostThreshold;
    };

    struct ClassificationParams{
        int NumClass;
    };

    // <============== Outputs =============>
    struct Image{
        ncnn::Mat image;
        int ow;
        int oh;
        int c;
        int nw;
        int nh;
    };
    struct Bbox{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int cid;
    };
    
    struct Keypoint{
        float x;
        float y;
        float score;
        int cid;
    };


    // <============== Operator =============>
    struct InferDeleter{
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
}

#endif //NCNN_COMMON_H
