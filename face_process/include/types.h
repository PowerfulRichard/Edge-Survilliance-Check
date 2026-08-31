#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct FaceObject {
    cv::Rect2f rect;
    std::array<cv::Point2f, 5> kps{};
    float det_score = 0.f;
};

struct MatchResult {
    std::string name = "UNKNOWN";
    float similarity = -1.f;
    bool known = false;
};

struct RecognizedFace {
    FaceObject face;
    MatchResult match;
    std::vector<float> embedding;
};

struct TimingInfo {
    double image_load_ms = 0.0;
    double detector_preprocess_ms = 0.0;
    double detector_inference_ms = 0.0;
    double detector_postprocess_ms = 0.0;
    double alignment_ms = 0.0;
    double recognizer_inference_ms = 0.0;
    double database_match_ms = 0.0;
    double database_io_ms = 0.0;
    double total_ms = 0.0;
};

struct PipelineResult {
    std::string status = "ok";
    std::string message;
    int image_width = 0;
    int image_height = 0;
    int detector_input_size = 0;
    std::vector<RecognizedFace> faces;
    TimingInfo timing;
};
