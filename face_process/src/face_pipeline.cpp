#include "face_pipeline.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

using Clock = std::chrono::steady_clock;

FacePipeline::FacePipeline(PipelineConfig cfg) : cfg_(std::move(cfg)) {}

bool FacePipeline::load_models(std::string& error) {
    SCRFD::Config dc;
    dc.input_size = cfg_.det_size;
    dc.score_threshold = cfg_.det_score;
    dc.nms_threshold = cfg_.nms;
    dc.num_threads = cfg_.threads;
    dc.use_vulkan = cfg_.use_vulkan;

    MobileFaceNet::Config rc;
    rc.num_threads = cfg_.threads;
    rc.use_vulkan = cfg_.use_vulkan;

    detector_ = std::make_unique<SCRFD>(dc);
    recognizer_ = std::make_unique<MobileFaceNet>(rc);

    if (detector_->load(cfg_.det_param, cfg_.det_bin) != 0) {
        error = "failed_to_load_scrfd";
        return false;
    }
    if (recognizer_->load(cfg_.rec_param, cfg_.rec_bin) != 0) {
        error = "failed_to_load_mobilefacenet";
        return false;
    }
    return true;
}

bool FacePipeline::load_database(FaceDatabase& db, bool allow_missing,
                                 TimingInfo* timing, std::string& error) const {
    const auto t0 = Clock::now();
    const bool exists = std::filesystem::exists(cfg_.db_path);
    if (!exists) {
        if (timing) timing->database_io_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (allow_missing) return true;
        error = "database_not_found";
        return false;
    }
    if (!db.load(cfg_.db_path)) {
        if (timing) timing->database_io_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        error = "failed_to_load_database";
        return false;
    }
    if (timing) timing->database_io_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return true;
}

bool FacePipeline::save_database(const FaceDatabase& db, TimingInfo* timing, std::string& error) const {
    const auto t0 = Clock::now();
    const bool ok = db.save(cfg_.db_path);
    if (timing) timing->database_io_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (!ok) error = "failed_to_save_database";
    return ok;
}

PipelineResult FacePipeline::run_image(const std::string& image_path,
                                       FaceDatabase* db,
                                       bool collect_timing,
                                       bool include_embeddings) {
    PipelineResult out;
    out.detector_input_size = cfg_.det_size;
    const auto total0 = Clock::now();

    const auto load0 = Clock::now();
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    const auto load1 = Clock::now();
    if (collect_timing) out.timing.image_load_ms = std::chrono::duration<double, std::milli>(load1 - load0).count();

    if (image.empty()) {
        out.status = "error";
        out.message = "cannot_read_image";
        return out;
    }
    out.image_width = image.cols;
    out.image_height = image.rows;

    auto faces = detector_->detect(image, collect_timing ? &out.timing : nullptr);
    if (faces.empty()) {
        out.status = "no_face";
        if (collect_timing) out.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total0).count();
        return out;
    }

    out.faces.reserve(faces.size());
    for (const FaceObject& f : faces) {
        RecognizedFace rf;
        rf.face = f;

        const auto a0 = Clock::now();
        cv::Mat aligned;
        const bool aligned_ok = MobileFaceNet::align5(image, f.kps, aligned);
        const auto a1 = Clock::now();
        if (collect_timing) out.timing.alignment_ms += std::chrono::duration<double, std::milli>(a1 - a0).count();
        if (!aligned_ok) continue;

        std::vector<float> emb;
        double rec_ms = 0.0;
        if (!recognizer_->extract(aligned, emb, collect_timing ? &rec_ms : nullptr)) continue;
        if (collect_timing) out.timing.recognizer_inference_ms += rec_ms;

        if (db) {
            const auto m0 = Clock::now();
            rf.match = db->match(emb, cfg_.threshold);
            const auto m1 = Clock::now();
            if (collect_timing) out.timing.database_match_ms += std::chrono::duration<double, std::milli>(m1 - m0).count();
        } else {
            rf.match = MatchResult{};
        }

        if (include_embeddings) rf.embedding = std::move(emb);
        out.faces.push_back(std::move(rf));
    }

    if (out.faces.empty()) {
        out.status = "error";
        out.message = "faces_detected_but_recognition_failed";
    }
    if (collect_timing) out.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total0).count();
    return out;
}

PipelineResult FacePipeline::infer(const std::string& image_path, bool collect_timing) {
    const auto overall0 = Clock::now();
    FaceDatabase db;
    std::string error;
    TimingInfo db_timing;
    if (!load_database(db, true, collect_timing ? &db_timing : nullptr, error)) {
        PipelineResult r;
        r.status = "error";
        r.message = error;
        r.detector_input_size = cfg_.det_size;
        if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
        return r;
    }
    PipelineResult r = run_image(image_path, &db, collect_timing, false);
    if (collect_timing) {
        r.timing.database_io_ms += db_timing.database_io_ms;
        r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
    }
    return r;
}

PipelineResult FacePipeline::enroll_if_unknown(const std::string& image_path,
                                               const std::string& name,
                                               bool collect_timing,
                                               bool* enrolled) {
    const auto overall0 = Clock::now();
    if (enrolled) *enrolled = false;
    FaceDatabase db;
    std::string error;
    TimingInfo io_timing;
    if (!load_database(db, true, collect_timing ? &io_timing : nullptr, error)) {
        PipelineResult r;
        r.status = "error";
        r.message = error;
        r.detector_input_size = cfg_.det_size;
        if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
        return r;
    }

    PipelineResult r = run_image(image_path, &db, collect_timing, true);
    if (collect_timing) r.timing.database_io_ms += io_timing.database_io_ms;
    if (r.status != "ok" || r.faces.empty()) {
        if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
        return r;
    }

    // Enrollment intentionally selects one face: max(area * detector confidence).
    auto best = std::max_element(r.faces.begin(), r.faces.end(), [](const RecognizedFace& a, const RecognizedFace& b) {
        return a.face.rect.area() * a.face.det_score < b.face.rect.area() * b.face.det_score;
    });

    if (best->match.known) {
        RecognizedFace chosen = *best;
        chosen.embedding.clear();
        r.faces.assign(1, std::move(chosen));
        if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
        return r;
    }

    if (name.empty()) {
        r.status = "error";
        r.message = "enroll_name_required";
        if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
        return r;
    }

    db.add_template(name, best->embedding);
    TimingInfo save_timing;
    if (!save_database(db, collect_timing ? &save_timing : nullptr, error)) {
        r.status = "error";
        r.message = error;
        if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
        return r;
    }
    if (collect_timing) r.timing.database_io_ms += save_timing.database_io_ms;

    // After insertion, the new template self-matches at cosine 1.0.
    best->match.name = name;
    best->match.similarity = 1.0f;
    best->match.known = true;
    RecognizedFace chosen = *best;
    chosen.embedding.clear();
    r.faces.assign(1, std::move(chosen));
    if (enrolled) *enrolled = true;
    if (collect_timing) r.timing.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - overall0).count();
    return r;
}
