
#include "scrfd.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

using Clock = std::chrono::steady_clock;

SCRFD::SCRFD(Config cfg) : cfg_(cfg) {
    net_.opt.num_threads = cfg_.num_threads;
    net_.opt.use_vulkan_compute = cfg_.use_vulkan;
    net_.opt.use_fp16_packed = true;
    net_.opt.use_fp16_storage = true;
    net_.opt.use_fp16_arithmetic = false;
}

int SCRFD::load(const std::string& param_path, const std::string& bin_path) {
    int ret = net_.load_param(param_path.c_str());
    if (ret != 0) return ret;
    return net_.load_model(bin_path.c_str());
}

float SCRFD::intersection_area(const FaceObject& a, const FaceObject& b) {
    const float x1 = std::max(a.rect.x, b.rect.x);
    const float y1 = std::max(a.rect.y, b.rect.y);
    const float x2 = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
    const float y2 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
    return std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
}

void SCRFD::nms_sorted_bboxes(const std::vector<FaceObject>& faces,
                              std::vector<int>& picked,
                              float nms_threshold) {
    picked.clear();
    std::vector<float> areas(faces.size());
    for (size_t i = 0; i < faces.size(); ++i) areas[i] = faces[i].rect.area();

    for (size_t i = 0; i < faces.size(); ++i) {
        bool keep = true;
        for (int j : picked) {
            const float inter = intersection_area(faces[i], faces[j]);
            const float uni = areas[i] + areas[j] - inter;
            if (uni > 0.f && inter / uni > nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) picked.push_back(static_cast<int>(i));
    }
}

std::vector<FaceObject> SCRFD::detect(const cv::Mat& bgr, TimingInfo* timing) const {
    if (bgr.empty()) return {};

    const auto prep0 = Clock::now();
    const int target = cfg_.input_size;
    const float scale = std::min(target / static_cast<float>(bgr.cols),
                                 target / static_cast<float>(bgr.rows));
    const int resized_w = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
    const int pad_x = (target - resized_w) / 2;
    const int pad_y = (target - resized_h) / 2;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
        bgr.cols, bgr.rows, resized_w, resized_h);

    ncnn::Mat padded;
    ncnn::copy_make_border(in, padded,
                           pad_y, target - resized_h - pad_y,
                           pad_x, target - resized_w - pad_x,
                           ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {cfg_.mean, cfg_.mean, cfg_.mean};
    const float norm_vals[3] = {cfg_.norm, cfg_.norm, cfg_.norm};
    padded.substract_mean_normalize(mean_vals, norm_vals);
    const auto prep1 = Clock::now();

    ncnn::Extractor ex = net_.create_extractor();
    if (ex.input("in0", padded) != 0) {
        std::cerr << "SCRFD input blob 'in0' not found\n";
        return {};
    }

    const char* score_names[3] = {"out0", "out1", "out2"};
    const char* bbox_names[3]  = {"out3", "out4", "out5"};
    const char* kps_names[3]   = {"out6", "out7", "out8"};
    const int strides[3] = {8, 16, 32};

    ncnn::Mat scores[3], bboxes[3], kpss[3];
    const auto infer0 = Clock::now();
    for (int i = 0; i < 3; ++i) {
        if (ex.extract(score_names[i], scores[i]) != 0 ||
            ex.extract(bbox_names[i], bboxes[i]) != 0 ||
            ex.extract(kps_names[i], kpss[i]) != 0) {
            std::cerr << "SCRFD output extraction failed at stride " << strides[i] << "\n";
            return {};
        }
    }
    const auto infer1 = Clock::now();

    const auto post0 = Clock::now();
    std::vector<FaceObject> proposals;

    for (int level = 0; level < 3; ++level) {
        const int stride = strides[level];
        const int feat_w = target / stride;
        const int feat_h = target / stride;
        const int cells = feat_w * feat_h;

        const int anchors = 2;
        const int points = cells * anchors;

        const int score_total = static_cast<int>(scores[level].total());
        const int bbox_total = static_cast<int>(bboxes[level].total());
        const int kps_total = static_cast<int>(kpss[level].total());

        if (cells <= 0 ||
            score_total < points ||
            bbox_total < points * 4 ||
            kps_total < points * 10) {
            std::cerr << "Unexpected SCRFD tensor layout at stride " << stride
                      << ": expected_points=" << points
                      << " score=" << score_total
                      << " bbox=" << bbox_total
                      << " kps=" << kps_total << "\n";
            return {};
        }

        const float* sp = reinterpret_cast<const float*>(scores[level].data);
        const float* bp = reinterpret_cast<const float*>(bboxes[level].data);
        const float* kp = reinterpret_cast<const float*>(kpss[level].data);

        for (int y = 0; y < feat_h; ++y) {
            for (int x = 0; x < feat_w; ++x) {
                for (int a = 0; a < anchors; ++a) {
                    const int idx = (y * feat_w + x) * anchors + a;
                    const float conf = sp[idx];
                    if (conf < cfg_.score_threshold) continue;

                    const float cx = static_cast<float>(x * stride);
                    const float cy = static_cast<float>(y * stride);
                    const float l = bp[idx * 4 + 0] * stride;
                    const float t = bp[idx * 4 + 1] * stride;
                    const float r = bp[idx * 4 + 2] * stride;
                    const float b = bp[idx * 4 + 3] * stride;

                    float x0 = (cx - l - pad_x) / scale;
                    float y0 = (cy - t - pad_y) / scale;
                    float x1 = (cx + r - pad_x) / scale;
                    float y1 = (cy + b - pad_y) / scale;
                    x0 = std::clamp(x0, 0.f, static_cast<float>(bgr.cols - 1));
                    y0 = std::clamp(y0, 0.f, static_cast<float>(bgr.rows - 1));
                    x1 = std::clamp(x1, 0.f, static_cast<float>(bgr.cols - 1));
                    y1 = std::clamp(y1, 0.f, static_cast<float>(bgr.rows - 1));

                    FaceObject f;
                    f.det_score = conf;
                    f.rect = cv::Rect2f(x0, y0,
                                        std::max(0.f, x1 - x0),
                                        std::max(0.f, y1 - y0));

                    for (int p = 0; p < 5; ++p) {
                        float px = (cx + kp[idx * 10 + p * 2 + 0] * stride - pad_x) / scale;
                        float py = (cy + kp[idx * 10 + p * 2 + 1] * stride - pad_y) / scale;
                        px = std::clamp(px, 0.f, static_cast<float>(bgr.cols - 1));
                        py = std::clamp(py, 0.f, static_cast<float>(bgr.rows - 1));
                        f.kps[p] = cv::Point2f(px, py);
                    }

                    proposals.push_back(f);
                }
            }
        }
    }

    std::sort(proposals.begin(), proposals.end(), [](const FaceObject& a, const FaceObject& b) {
        return a.det_score > b.det_score;
    });

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, cfg_.nms_threshold);

    std::vector<FaceObject> out;
    out.reserve(picked.size());
    for (int i : picked) out.push_back(proposals[static_cast<size_t>(i)]);

    const auto post1 = Clock::now();

    if (timing) {
        timing->detector_preprocess_ms +=
            std::chrono::duration<double, std::milli>(prep1 - prep0).count();

        timing->detector_inference_ms +=
            std::chrono::duration<double, std::milli>(infer1 - infer0).count();

        timing->detector_postprocess_ms +=
            std::chrono::duration<double, std::milli>(post1 - post0).count();
    }

    return out;
}

