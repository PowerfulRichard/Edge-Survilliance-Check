#pragma once
#include <array>
#include <net.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

class MobileFaceNet {
public:
    struct Config {
        int num_threads = 4;
        bool use_vulkan = false;
        float mean = 127.5f;
        float norm = 1.0f / 128.0f;
    };

    explicit MobileFaceNet(Config cfg);
    int load(const std::string& param_path, const std::string& bin_path);

    static bool align5(const cv::Mat& src,
                       const std::array<cv::Point2f, 5>& kps,
                       cv::Mat& aligned112);

    bool extract(const cv::Mat& aligned_bgr,
                 std::vector<float>& embedding,
                 double* inference_ms = nullptr) const;

    static void l2_normalize(std::vector<float>& v);

private:
    Config cfg_;
    ncnn::Net net_;
};
