#pragma once
#include "face_database.h"
#include "mobilefacenet.h"
#include "scrfd.h"
#include "types.h"
#include <memory>
#include <string>

struct PipelineConfig {
    std::string det_param;
    std::string det_bin;
    std::string rec_param;
    std::string rec_bin;
    std::string db_path = "faces.db";
    int det_size = 320;
    int threads = 4;
    float det_score = 0.45f;
    float nms = 0.40f;
    float threshold = 0.45f;
    bool use_vulkan = false;
};

class FacePipeline {
public:
    explicit FacePipeline(PipelineConfig cfg);
    bool load_models(std::string& error);

    PipelineResult infer(const std::string& image_path, bool collect_timing);
    PipelineResult enroll_if_unknown(const std::string& image_path,
                                     const std::string& name,
                                     bool collect_timing,
                                     bool* enrolled = nullptr);

private:
    PipelineConfig cfg_;
    std::unique_ptr<SCRFD> detector_;
    std::unique_ptr<MobileFaceNet> recognizer_;

    bool load_database(FaceDatabase& db, bool allow_missing, TimingInfo* timing, std::string& error) const;
    bool save_database(const FaceDatabase& db, TimingInfo* timing, std::string& error) const;
    PipelineResult run_image(const std::string& image_path,
                             FaceDatabase* db,
                             bool collect_timing,
                             bool include_embeddings);
};
