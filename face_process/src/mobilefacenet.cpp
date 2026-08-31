#include "mobilefacenet.h"
#include <chrono>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using Clock = std::chrono::steady_clock;

MobileFaceNet::MobileFaceNet(Config cfg) : cfg_(cfg) {
    net_.opt.num_threads = cfg_.num_threads;
    net_.opt.use_vulkan_compute = cfg_.use_vulkan;
    net_.opt.use_fp16_packed = true;
    net_.opt.use_fp16_storage = true;
    net_.opt.use_fp16_arithmetic = false;
}

int MobileFaceNet::load(const std::string& param_path, const std::string& bin_path) {
    int ret = net_.load_param(param_path.c_str());
    if (ret != 0) return ret;
    return net_.load_model(bin_path.c_str());
}

bool MobileFaceNet::align5(const cv::Mat& src,
                           const std::array<cv::Point2f, 5>& kps,
                           cv::Mat& aligned112) {
    static const std::vector<cv::Point2f> dst = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}
    };
    std::vector<cv::Point2f> src_pts(kps.begin(), kps.end());
    cv::Mat inliers;
    cv::Mat M = cv::estimateAffinePartial2D(src_pts, dst, inliers, cv::LMEDS);
    if (M.empty()) return false;
    cv::warpAffine(src, aligned112, M, cv::Size(112, 112), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return !aligned112.empty();
}

void MobileFaceNet::l2_normalize(std::vector<float>& v) {
    double ss = 0.0;
    for (float x : v) ss += static_cast<double>(x) * x;
    const float n = static_cast<float>(std::sqrt(ss));
    if (n < 1e-12f) return;
    for (float& x : v) x /= n;
}

bool MobileFaceNet::extract(const cv::Mat& aligned_bgr,
                            std::vector<float>& embedding,
                            double* inference_ms) const {
    if (aligned_bgr.empty()) return false;

    cv::Mat face;
    if (aligned_bgr.size() == cv::Size(112, 112)) face = aligned_bgr;
    else cv::resize(aligned_bgr, face, cv::Size(112, 112));

    ncnn::Mat in = ncnn::Mat::from_pixels(face.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
    const float mean_vals[3] = {cfg_.mean, cfg_.mean, cfg_.mean};
    const float norm_vals[3] = {cfg_.norm, cfg_.norm, cfg_.norm};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net_.create_extractor();
    if (ex.input("in0", in) != 0) return false;

    const auto t0 = Clock::now();
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) return false;
    const auto t1 = Clock::now();
    if (inference_ms) *inference_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (out.total() != 512) return false;
    embedding.resize(out.total());
    const float* ptr = reinterpret_cast<const float*>(out.data);
    std::copy(ptr, ptr + out.total(), embedding.begin());
    l2_normalize(embedding);
    return true;
}
