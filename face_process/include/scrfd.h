#pragma once
#include "types.h"
#include <net.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

class SCRFD {
public:
    struct Config {
        int input_size = 320;
        float score_threshold = 0.45f;
        float nms_threshold = 0.40f;
        int num_threads = 4;
        bool use_vulkan = false;
        float mean = 127.5f;
        float norm = 1.0f / 128.0f;
    };

    explicit SCRFD(Config cfg);
    int load(const std::string& param_path, const std::string& bin_path);
    std::vector<FaceObject> detect(const cv::Mat& bgr, TimingInfo* timing = nullptr) const;
    int input_size() const { return cfg_.input_size; }

private:
    Config cfg_;
    ncnn::Net net_;

    static float intersection_area(const FaceObject& a, const FaceObject& b);
    static void nms_sorted_bboxes(const std::vector<FaceObject>& faces,
                                  std::vector<int>& picked,
                                  float nms_threshold);
};
