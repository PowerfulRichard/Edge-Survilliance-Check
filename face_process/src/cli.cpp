#include "cli.h"
#include <filesystem>
#include <iostream>

ArgMap parse_args(int argc, char** argv) {
    ArgMap out;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0) continue;
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) out[key] = argv[++i];
        else out[key] = "1";
    }
    return out;
}

std::string arg_get(const ArgMap& args, const std::string& key, const std::string& def) {
    const auto it = args.find(key);
    return it == args.end() ? def : it->second;
}

bool build_config(const ArgMap& args, PipelineConfig& cfg, std::string& image_path, std::string& error) {
    image_path = arg_get(args, "--image");
    cfg.det_param = arg_get(args, "--det-param");
    cfg.det_bin = arg_get(args, "--det-bin");
    cfg.rec_param = arg_get(args, "--rec-param");
    cfg.rec_bin = arg_get(args, "--rec-bin");
    cfg.db_path = arg_get(args, "--db", "faces.db");

    if (image_path.empty() || cfg.det_param.empty() || cfg.det_bin.empty() || cfg.rec_param.empty() || cfg.rec_bin.empty()) {
        error = "missing_required_argument";
        return false;
    }

    try {
        cfg.det_size = std::stoi(arg_get(args, "--det-size", "320"));
        cfg.threads = std::stoi(arg_get(args, "--threads", "4"));
        cfg.det_score = std::stof(arg_get(args, "--det-score", "0.45"));
        cfg.nms = std::stof(arg_get(args, "--nms", "0.40"));
        cfg.threshold = std::stof(arg_get(args, "--threshold", "0.45"));
        cfg.use_vulkan = arg_get(args, "--vulkan", "0") == "1";
    } catch (...) {
        error = "invalid_numeric_argument";
        return false;
    }

    if (cfg.det_size != 160 && cfg.det_size != 320 && cfg.det_size != 480 && cfg.det_size != 640) {
        error = "det_size_must_be_160_320_480_or_640";
        return false;
    }
    if (cfg.threads < 1) {
        error = "threads_must_be_positive";
        return false;
    }
    return true;
}

static void common_usage(const char* exe) {
    std::cout
        << "  " << exe << " --image IMAGE --det-param DET.param --det-bin DET.bin\\\n\n"
        << "      --rec-param REC.param --rec-bin REC.bin --db faces.db --det-size 320\\\n\n"
        << "      [--threshold 0.45] [--det-score 0.45] [--nms 0.40] [--threads 4]\n";
}

void print_usage_main() {
    std::cout << "Inference:\n";
    common_usage("face_main --mode infer");
    std::cout << "Enrollment (only adds a template when the selected face is currently UNKNOWN):\n";
    common_usage("face_main --mode enroll --name Alice");
}

void print_usage_mini() {
    std::cout << "Inference only:\n";
    common_usage("face_mini");
}
