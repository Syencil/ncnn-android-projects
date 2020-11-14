// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/6/4

#ifndef NCNN_UTILS_H
#define NCNN_UTILS_H

//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>

#include "common.h"

//cv::Mat renderBoundingBox(cv::Mat image, const std::vector<common::Bbox> &bboxes);
//
//cv::Mat renderKeypoint(cv::Mat image, const std::vector<common::Keypoint> &keypoints);

void nms_cpu(std::vector<common::Bbox> &bboxes, float threshold);

// ===========Template Operation ==========>
template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

template <typename T>
T clip(const T &n, const T &lower, const T &upper){
    return std::max(lower, std::min(n, upper));
}

template <typename T>
T sigmoid(const T &n){
    return 1 / (1 + exp(-n));
}

template <typename T>
T fast_sigmoid(const T &n){
    return (n / (1 + abs(n))) * 0.5 + 0.5;
}

// ===========Time Fun ==========>
template <typename _ClockType>
class Clock{
private:
    std::chrono::time_point<_ClockType> start_t;
    std::chrono::time_point<_ClockType> end_t;

public:
    void tick(){
        start_t = _ClockType::now();
    }
    void tock() {
        end_t = _ClockType::now();
    }

    template <typename T>
    T duration(){
        T elapsedTime = std::chrono::duration<T, std::milli>(end_t - start_t).count();
        return elapsedTime;
    }
};

#endif //NCNN_UTILS_H
