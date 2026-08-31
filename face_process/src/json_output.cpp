#include "json_output.h"
#include <iomanip>
#include <sstream>

static std::string esc(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "?";
                else o << static_cast<char>(c);
        }
    }
    return o.str();
}

std::string result_to_json(const PipelineResult& r,
                           bool include_timing,
                           const std::string& mode,
                           const std::string& action) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(6);
    o << "{";
    o << "\"status\":\"" << esc(r.status) << "\"";
    o << ",\"mode\":\"" << esc(mode) << "\"";
    if (!action.empty()) o << ",\"action\":\"" << esc(action) << "\"";
    if (!r.message.empty()) o << ",\"message\":\"" << esc(r.message) << "\"";
    o << ",\"image\":{\"width\":" << r.image_width << ",\"height\":" << r.image_height << "}";
    o << ",\"detector_input_size\":" << r.detector_input_size;
    o << ",\"face_count\":" << r.faces.size();
    o << ",\"faces\":[";

    for (size_t i = 0; i < r.faces.size(); ++i) {
        if (i) o << ',';
        const auto& f = r.faces[i];
        o << "{";
        o << "\"username\":\"" << esc(f.match.name) << "\"";
        // 'accuracy' is kept for the Python interface requested by the project.
        // It is cosine similarity, not a calibrated probability/accuracy statistic.
        o << ",\"accuracy\":" << f.match.similarity;
        o << ",\"metric\":\"cosine_similarity\"";
        o << ",\"known\":" << (f.match.known ? "true" : "false");
        o << ",\"detection_score\":" << f.face.det_score;
        o << ",\"bbox\":["
          << f.face.rect.x << ',' << f.face.rect.y << ','
          << f.face.rect.width << ',' << f.face.rect.height << ']';
        o << ",\"landmarks\":[";
        for (size_t p = 0; p < f.face.kps.size(); ++p) {
            if (p) o << ',';
            o << '[' << f.face.kps[p].x << ',' << f.face.kps[p].y << ']';
        }
        o << "]}";
    }
    o << ']';

    if (include_timing) {
        const auto& t = r.timing;
        o << ",\"timing_ms\":{";
        o << "\"image_load\":" << t.image_load_ms;
        o << ",\"detector_preprocess\":" << t.detector_preprocess_ms;
        o << ",\"detector_inference\":" << t.detector_inference_ms;
        o << ",\"detector_postprocess\":" << t.detector_postprocess_ms;
        o << ",\"alignment\":" << t.alignment_ms;
        o << ",\"recognizer_inference\":" << t.recognizer_inference_ms;
        o << ",\"database_match\":" << t.database_match_ms;
        o << ",\"database_io\":" << t.database_io_ms;
        o << ",\"total\":" << t.total_ms;
        o << '}';
    }
    o << '}';
    return o.str();
}
