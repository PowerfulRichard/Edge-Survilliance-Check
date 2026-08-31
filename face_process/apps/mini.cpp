#include "cli.h"
#include "face_pipeline.h"
#include "json_output.h"
#include <iostream>

int main(int argc, char** argv) {
    const ArgMap args = parse_args(argc, argv);
    if (argc < 2 || args.count("--help")) {
        print_usage_mini();
        return 0;
    }

    PipelineConfig cfg;
    std::string image_path, error;
    if (!build_config(args, cfg, image_path, error)) {
        PipelineResult r;
        r.status = "error";
        r.message = error;
        std::cout << result_to_json(r, false, "infer") << '\n';
        return 2;
    }

    FacePipeline pipeline(cfg);
    if (!pipeline.load_models(error)) {
        PipelineResult r;
        r.status = "error";
        r.message = error;
        r.detector_input_size = cfg.det_size;
        std::cout << result_to_json(r, false, "infer") << '\n';
        return 3;
    }

    const PipelineResult r = pipeline.infer(image_path, false);
    std::cout << result_to_json(r, false, "infer") << '\n';
    return r.status == "error" ? 4 : 0;
}
